# train, test
from JH_data_loader import CelebA_DATASET, CelebA_DATALOADER
from JH_model import StarGAN_Discriminator, StarGAN_Generator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
        self.output()

    def output(self):
        return torch.mean((self.dydx_l2norm-1)**2)

class Solver(object):
    def __init__(self, config, data_loader):
        # hyper parameter -> main에서 config 인자를 받아온다.
        self.nb_epochs = config.nb_epochs
        self.domain_dim = config.domain_dim
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.G_learning_rate = config.G_learning_rate
        self.D_learning_rate = config.D_learning_rate

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

        self.train()
        self.test()

    def train(self):
        for epoch in range(self.nb_epochs):
            for idx, [images, labels] in enumerate(self.data_loader):
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
                real_cls_loss = self.criterion(out_cls, labels) / self.batch_size # batch_size 만큼 나눠줘야 한다.
                
                # fake image -> real/fake(fake로 판단해야 한다. mean 값을 최소화 or fake label과의 차이가 적어야 한다.), classification -> target domain
                # fake image 생성을 위한 target label 생성 -> origin label의 순서를 random하게 섞는다.
                rand_idx = torch.randperm(labels.size(0))
                trg_labels = labels[rand_idx].to(self.device)
                fake_images = self.generator(images, trg_labels)
                # print(fake_images.shape) # [16, 3, 128, 128]
                
                fake_out_src, fake_out_cls = self.discriminator(fake_images)

                fake_src_loss = torch.mean(fake_out_src) # mean 값을 최소화 -> fake image를 fake로 인식해야 하기 때문
                fake_out_cls = fake_out_cls.reshape(fake_out_cls.size(0), fake_out_cls.size(1))

                # Gradient Penalty Loss
                alpha = torch.rand(images.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * images.data + (1 - alpha) * fake_images.data).requires_grad_(True)
                out_src, _ = self.discriminator(x_hat)
                gp_loss = Gradient_Penalty_Loss(out_src, x_hat, self.device).output()
                
                D_loss = self.lambda_real_src * real_src_loss + self.lambda_real_cls * real_cls_loss + self.lambda_fake_src * fake_src_loss
                + self.lambda_gp * gp_loss 

                self.D_optimizer.zero_grad()
                self.G_optimizer.zero_grad()
                D_loss.backward()
                self.D_optimizer.step()

                # Generator 학습
                
                fake_images = self.generator(images, trg_labels) # 게임의 공평성을 위해, Discriminator 학습에서 이용했던 동일한 fake image를 사용해야 한다.
                G_out_src, G_out_cls = self.discriminator(fake_images)

                G_out_cls = G_out_cls.reshape(G_out_cls.size(0), G_out_cls.size(1))

                G_src_loss = -torch.mean(G_out_src) # mean 값을 최대화 -> fake image를 real로 인식해야 하기 때문
                G_cls_loss = self.criterion(G_out_cls, trg_labels) / self.batch_size
                
                # Cyclic Consistency Loss
                recon_images = self.generator(fake_images, labels)
                recon_loss = torch.mean(torch.abs(recon_images - images))
                # print(recon_loss)

                G_loss = self.lambda_src * G_src_loss + self.lambda_cls * G_cls_loss + self.lambda_recon * recon_loss
                
                self.G_optimizer.zero_grad()
                self.D_optimizer.zero_grad()
                G_loss.backward()
                self.G_optimizer.step()
                
            # n epoch 마다 loss 출력
            if epoch % self.loss_iter == 0:
                
            # n epoch 마다 result 출력
            if epoch % self.result_iter == 0:

    def test(self):
        print()