# train, test
from JH_data_loader import CelebA_DATASET, CelebA_DATALOADER
from JH_model import StarGAN_Discriminator, StarGAN_Generator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
import os

# best evaluation 시점의 checkpoints를 저장한다.
class Save_Best_Checkpoints(object):
    def __init__(self, epoch, # epoch or iteration
                       generator,
                       G_optimizer,
                       save_generator_path,
                       discriminator,
                       D_optimizer,
                       save_discriminator_path,
                       loss, # dictionary -> loss = {'generator': loss_G, 'discriminator': loss_D}
                       min_loss # dictionary -> loss = {'generator': loss_G_min, 'discriminator': loss_D_min}
                       ):
        super(Save_Best_Checkpoints, self).__init__()
        self.epoch = epoch
        self.generator = generator
        self.G_optimizer = G_optimizer
        self.save_generator_path = save_generator_path
        self.discriminator = discriminator
        self.D_optimizer = D_optimizer
        self.save_discriminator_path = save_discriminator_path
        self.loss = loss # input
        self.min_loss = min_loss

    def save_checkpoints_best_eval(self):
        # generator_loss와 dictionary_loss가 가장 작을 때 같이 저장 -> generator_loss가 가장 적을 때 같이 저장
        if self.loss['generator'] < self.min_loss['generator']:
            self.min_loss['generator'] = self.loss['generator']
            self.min_loss['discriminator'] = self.loss['discriminator']

            # Generator
            torch.save({'epoch': self.epoch,
                        'model': self.generator.state_dict(),
                        'optimizer': self.G_optimizer.state_dict()}, os.path.join(self.save_generator_path, 'generator_best_epoch.pt'))

            # Discriminator
            torch.save({'epoch': self.epoch,
                        'model': self.discriminator.state_dict(),
                        'optimizer': self.G_optimizer.state_dict()}, os.path.join(self.save_discriminator_path, 'discriminator_best_epoch.pt'))
        
        return self.min_loss # dictionary 갱신 -> for문을 돌기 전 미리 정의해 둬야 한다.

# Checkpoints 저장 -> Class
# 모든 epoch를 저장한다. 가장 최근의 Checkpoints를 저장한다.
class Save_Checkpoints(object):
    def __init__(self, epoch, # epoch or iteration
                       generator, 
                       G_optimizer, 
                       save_generator_path, 
                       discriminator, 
                       D_optimizer, 
                       save_discriminator_path, 
                       select = 'epoch'): # 'epoch', 'last'
        super(Save_Checkpoints, self).__init__()
        self.epoch = epoch
        self.generator = generator
        self.G_optimizer = G_optimizer
        self.save_generator_path = save_generator_path
        self.discriminator = discriminator
        self.D_optimizer = D_optimizer
        self.save_discriminator_path = save_discriminator_path
        
        # 가장 최근의 Checkpoints VS best evaluation 시점의 Checkpoints
        if select == 'epoch':
            self.save_checkpoints_epoch() # return하는 값이 없기 때문에 class만 호출해도 된다.
        elif select == 'last':
            self.save_checkpoints_last()
            
    def save_checkpoints_epoch(self):
        # Generator
        torch.save({'epoch': self.epoch,
                    'model': self.generator.state_dict(),
                    'optimizer': self.G_optimizer.state_dict()}, os.path.join(self.save_generator_path, 'generator_epoch_{}.pt'.format(self.epoch)))

        # Discriminator
        torch.save({'epoch': self.epoch,
                    'model': self.discriminator.state_dict(),
                    'optimizer': self.G_optimizer.state_dict()}, os.path.join(self.save_discriminator_path, 'discriminator_epoch_{}.pt'.format(self.epoch)))

    def save_checkpoints_last(self):
        # Generator
        torch.save({'epoch': self.epoch,
                    'model': self.generator.state_dict(),
                    'optimizer': self.G_optimizer.state_dict()}, os.path.join(self.save_generator_path, 'generator_last_epoch.pt'))

        # Discriminator
        torch.save({'epoch': self.epoch,
                    'model': self.discriminator.state_dict(),
                    'optimizer': self.G_optimizer.state_dict()}, os.path.join(self.save_discriminator_path, 'discriminator_last_epoch.pt'))

# Gradient Penalty Loss -> Class, ***이해가 안된다.
class Gradient_Penalty_Loss(object):
    def __init__(self, y, x, device):
        super(Gradient_Penalty_Loss, self).__init__()
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        self.dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        # self.output() -> return 값이 있기 때문에 굳이 함수를 호출하지 않아도 된다.

    def output(self):
        return torch.mean((self.dydx_l2norm-1)**2) # return 값이 있기 때문에 Class를 호출할 때 함수를 같이 호출하여야 한다.

class Solver(object):
    def __init__(self, config, data_loader):
        # hyper parameter -> main에서 config 인자를 받아온다.
        # self.nb_epochs = config.nb_epochs
        self.nb_iters = config.nb_iters
        self.resume_iters = None
        self.domain_dim = config.domain_dim
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.G_learning_rate = config.G_learning_rate
        self.D_learning_rate = config.D_learning_rate
        self.data_loader = data_loader

        # self.iter = len(self.data_loader) # 전체 dataset // batch_size -> 1 번의 epoch = self.iter

        # Save iteration

        self.loss_iter = config.loss_iter
        self.result_iter = config.result_iter
        self.checkpoints_iter = config.checkpoints_iter 

        # Save path
        self.save_train_path = config.save_train_path
        self.save_test_path = config.save_test_path
        self.save_generator_path = config.save_generator_path
        self.save_discriminator_path = config.save_discriminator_path

        # Discriminator
        self.lambda_real_src = config.lambda_real_src
        self.lambda_real_cls = config.lambda_real_cls
        self.lambda_fake_src = config.lambda_fake_src
        self.lambda_gp = config.lambda_gp

        # Generator
        self.lambda_src = config.lambda_src
        self.lambda_cls = config.lambda_cls
        self.lambda_recon = config.lambda_recon

        self.data_loader = data_loader
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # device -> GPU 처리, dataset(image, label), model
        
        self.generator = StarGAN_Generator(domain_dim=self.domain_dim, batch_size=self.batch_size, image_size=self.image_size).to(self.device)
        self.discriminator = StarGAN_Discriminator(domain_dim=self.domain_dim, image_size=self.image_size).to(self.device)

        self.criterion = nn.BCEWithLogitsLoss() # Cross Entropy Loss처럼 softmax를 포함하지 않기 때문에, softmax 또는 다른 activation function을 적용해야 한다.
        # nn.BCELoss(nn.Sigmoid(input), target)

        self.G_optimizer = optim.Adam(self.generator.parameters(), self.G_learning_rate, [self.beta1, self.beta2])
        self.D_optimizer = optim.Adam(self.discriminator.parameters(), self.D_learning_rate, [self.beta1, self.beta2])

        # For visualization
        # 첫 번째 iteration에서 나오는 data
        images, labels = next(iter(self.data_loader)) # [batch_size, domain_dim]
        rand_idx = torch.randperm(self.batch_size)
        self.fixed_images = images.to(self.device)
        # self.fixed_trg_labels = labels[rand_idx].to(self.device)
        
        # 굳이 하지 않아도 된다.
        # self.train()
        # self.test()

    # checkpoints : epoch -> epoch, iteration -> iteration
    def train(self):
        # iteration으로 학습 돌리기 -> next(iter)
        start_iters = 0
        # if self.resume_iters != None: # resume_iters이 존재한다면
        #     start_iters = self.resume_iters
        #     # 저장한 모델 불러오기 -> Generator, Discriminator
        #     ckpt_G = torch.load(os.path.join(self.save_generator_path, 'generator_epoch_{}.pt'.format(self.resume_iters)))
        #     ckpt_D = torch.load(os.path.join(self.save_discriminator_path, 'discriminator_epoch_{}.pt'.format(self.resume_iters)))
        #     self.generator.load_state_dict(ckpt_G['model'])
        #     self.discriminator.load_state_dict(ckpt_D['model'])
        #     self.G_optimizer.load_state_dict(ckpt_G['optimizer'])
        #     self.D_optimizer.load_state_dict(ckpt_D['optimizer'])
            
        for iters in range(start_iters, self.nb_iters):
            images, labels = next(iter(self.data_loader))
            # self.iteration = self.epoch * self.iter + idx
            
            # data(images, labels) -> GPU 처리
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # print(images.shape) #[16, 3, 128, 128]
            # print(labels.shape) #[16, 5]

            # Discriminator 학습
            # real image -> real/fake(real로 판단해야 한다. mean 값을 최대화 or real label과의 차이가 적어야 한다.), classification -> origin domain
            out_src, out_cls = self.discriminator(images)
            
            # print(out_src.shape) # [16, 1, 2, 2]
            # print(out_cls.shape) # [16, 5, 1, 1]
            
            real_src_loss = -torch.mean(out_src) # mean 값을 최대화 -> real image를 real로 인식해야 하기 때문
            
            out_cls = out_cls.reshape(out_cls.size(0), out_cls.size(1))
            # real_cls_loss = self.criterion(out_cls, labels) / self.batch_size # batch_size 만큼 나눠줘야 한다.
            real_cls_loss = self.criterion(out_cls, labels)
            # real_cls_loss = F.binary_cross_entropy_with_logits(out_cls, labels) / self.batch_size
            # fake image -> real/fake(fake로 판단해야 한다. mean 값을 최소화 or fake label과의 차이가 적어야 한다.), classification -> target domain
            # fake image 생성을 위한 target label 생성 -> origin label의 순서를 random하게 섞는다.
            rand_idx = torch.randperm(labels.size(0))
            trg_labels = labels[rand_idx].to(self.device)
            fake_images = self.generator(images, trg_labels)
            # print(fake_images.shape) # [16, 3, 128, 128]
            
            fake_out_src, fake_out_cls = self.discriminator(fake_images)

            fake_src_loss = torch.mean(fake_out_src) # mean 값을 최소화 -> fake image를 fake로 인식해야 하기 때문
            fake_out_cls = fake_out_cls.reshape(fake_out_cls.size(0), fake_out_cls.size(1))

            # Gradient Penalty Loss -> 이해 안된다.
            alpha = torch.rand(images.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * images.data + (1 - alpha) * fake_images.data).requires_grad_(True)
            out_src, _ = self.discriminator(x_hat)
            gp_loss = Gradient_Penalty_Loss(out_src, x_hat, self.device).output()
            # print(self.lambda_real_src, self.lambda_real_cls, self.lambda_fake_src, self.lambda_gp)
            D_loss = self.lambda_real_src * real_src_loss + self.lambda_real_cls * real_cls_loss + self.lambda_fake_src * fake_src_loss + self.lambda_gp * gp_loss 
            # D_loss = self.lambda_real_src * real_src_loss + 1 * real_cls_loss + self.lambda_fake_src * fake_src_loss + self.lambda_gp * gp_loss 
            # D_loss = 1 * real_src_loss + 1 * real_cls_loss + 1 * fake_src_loss + 10 * gp_loss 
            # d_loss = d_loss_real + d_loss_fake + 1 * d_loss_cls + 10 * d_loss_gp

            self.D_optimizer.zero_grad()
            self.G_optimizer.zero_grad()
            D_loss.backward()
            self.D_optimizer.step()
            if iters % 5 == 0:

                # Generator 학습 
                fake_images = self.generator(images, trg_labels) # 게임의 공평성을 위해, Discriminator 학습에서 이용했던 동일한 fake image를 사용해야 한다.
                G_out_src, G_out_cls = self.discriminator(fake_images)

                G_out_cls = G_out_cls.reshape(G_out_cls.size(0), G_out_cls.size(1))

                G_src_loss = -torch.mean(G_out_src) # mean 값을 최대화 -> fake image를 real로 인식해야 하기 때문
                # G_cls_loss = self.criterion(G_out_cls, trg_labels) / self.batch_size
                G_cls_loss = self.criterion(G_out_cls, trg_labels)
                # G_cls_loss = F.binary_cross_entropy_with_logits(G_out_cls, trg_labels) / self.batch_size

                # Cyclic Consistency Loss
                recon_images = self.generator(fake_images, labels)
                recon_loss = torch.mean(torch.abs(recon_images - images))
                # print(recon_loss)
                
                G_loss = self.lambda_src * G_src_loss + self.lambda_cls * G_cls_loss + self.lambda_recon * recon_loss
                # G_loss = self.lambda_src * G_src_loss + 1 * G_cls_loss + self.lambda_recon * recon_loss
                # G_loss = 1 * G_src_loss + 1 * G_cls_loss + 10 * recon_loss
                # g_loss = g_loss_fake + 10 * g_loss_rec + 1 * g_loss_cls
                # print(self.lambda_src, self.lambda_cls, self.lambda_recon)

                self.G_optimizer.zero_grad()
                self.D_optimizer.zero_grad()
                G_loss.backward()
                self.G_optimizer.step()

            if iters % self.loss_iter == 0:
                D_loss_dict = {}
                D_loss_dict['real_src_loss'] = real_src_loss.item()
                D_loss_dict['real_cls_loss'] = real_cls_loss.item()
                D_loss_dict['fake_src_loss'] = fake_src_loss.item()
                D_loss_dict['gp_loss'] = gp_loss.item()
                D_loss_dict['D_loss'] = D_loss.item()

                G_loss_dict = {}
                G_loss_dict['G_src_loss'] = G_src_loss.item()
                G_loss_dict['G_cls_loss'] = G_cls_loss.item()
                G_loss_dict['recon_loss'] = recon_loss.item()
                G_loss_dict['G_loss'] = G_loss.item()

                print('iteration : {}'.format(iters), D_loss_dict, G_loss_dict)

            if iters % self.result_iter == 0: # 학습을 마치지 않았다. -> 학습 모드를 끄고 test 해야 한다.
                # image를 저장하기 전, tanh() [-1, 1] -> [0, 1]
                # (x + 1) / 2 -> 한 이미지의 모든 픽셀에 + 1 / 2
                fixed_trg_labels = torch.FloatTensor([[1, 0, 0, 1, 1]])
                fixed_trg_labels = fixed_trg_labels.repeat(repeats=[16, 1])
                fixed_trg_labels = fixed_trg_labels.to(self.device)
                result_fake_images = self.generator(self.fixed_images, fixed_trg_labels)
                result_fake_images = (result_fake_images + 1) / 2
                result_fake_images = result_fake_images.clamp_(0, 1) # Q. clamp를 하지 않으면 안되나?
                # result_fake_images = torch.clamp_(result_fake_images, 0, 1)
                # result_fake_images = result_fake_images.clamp_(result_fake_images, 0, 1) # 최소값 -> 0, 최대값 -> 1로 강제적으로 mapping
                # print(result_fake_images.shape) #[16, 3, 128, 128]
                # real image + fake image
                result_images = torch.cat([self.fixed_images, result_fake_images], dim=0)
                save_image(result_images, os.path.join(self.save_train_path, 'train_{}.png'.format(iters)))

            if iters % self.checkpoints_iter == 0 and iters > 0:
                Save_Checkpoints(iters, self.generator, self.G_optimizer, self.save_generator_path, self.discriminator, self.D_optimizer, self.save_discriminator_path, select='epoch')
                self.resume_iters = iters
                
    def test(self):
        ckpt = torch.load(os.path.join(self.save_generator_path, 'generator_epoch_{}.pt'.format(self.resume_iters))) # 가장 최근의 Generator checkpoints를 가져온다.
        self.generator.load_state_dict(ckpt['model'])
        self.generator.eval() # batchnormalization이나, dropout layer를 사용하지 않기 위해서
        with torch.no_grad(): # gradient를 계산하지 않아서, 연산량을 절약하기 위해서
            # random index 생성 -> fake image 생성
            # Test data 10개의 batch 생성
            for idx, [images, labels] in enumerate(self.data_loader):
                images = images.to(self.device)
                rand_idx = torch.randperm(self.batch_size)
                trg_labels = labels[rand_idx].to(self.device)
                fake_images = self.generator(images, trg_labels)
                result_images = torch.cat([images, fake_images], dim=0)
                save_image(result_images, os.path.join(self.save_test_path, 'test_{}.png'.format(idx + 1)))

                if idx == 10: # 10개의 image만 test
                    break