from JH_model import StarGAN_Discriminator, StarGAN_Generator
from JH_logger import Logger

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import os

# Checkpoints
class Save_Checkpoints(object):
    def __init__(self, epoch, # epoch or iteration
                       generator, 
                       save_generator_path, 
                       discriminator, 
                       save_discriminator_path, 
                       select = 'epoch'): # 'epoch' or 'last'
        super(Save_Checkpoints, self).__init__()
        self.epoch = epoch
        self.generator = generator
        self.save_generator_path = save_generator_path
        self.discriminator = discriminator
        self.save_discriminator_path = save_discriminator_path
        self.select = select
        if self.select == 'epoch':
            self.save_checkpoints_epoch()
        elif self.select == 'last':
            self.save_checkpoints_last()
            
    def save_checkpoints_epoch(self):
        # Generator
        torch.save({'epoch': self.epoch,
                    'model': self.generator.state_dict()}, os.path.join(self.save_generator_path, 'generator_epoch_{}.pt'.format(self.epoch)))
        # Discriminator
        torch.save({'epoch': self.epoch,
                    'model': self.discriminator.state_dict()}, os.path.join(self.save_discriminator_path, 'discriminator_epoch_{}.pt'.format(self.epoch)))

    def save_checkpoints_last(self):
        # Generator
        torch.save({'epoch': self.epoch,
                    'model': self.generator.state_dict()}, os.path.join(self.save_generator_path, 'generator_last_epoch.pt'))
        # Discriminator
        torch.save({'epoch': self.epoch,
                    'model': self.discriminator.state_dict()}, os.path.join(self.save_discriminator_path, 'discriminator_last_epoch.pt'))

# Gradient penalty loss
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

    def output(self):
        return torch.mean((self.dydx_l2norm-1)**2)

class Solver(object):
    def __init__(self, config, data_loader):
        # Basic
        self.data_loader = data_loader
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        
        self.domain_dim = config.domain_dim
        self.nb_iters = config.nb_iters
        self.resume_iters = config.resume_iters
        self.G_learning_rate = config.G_learning_rate
        self.D_learning_rate = config.D_learning_rate
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        
        # Save iter
        self.train_iter = config.train_iter
        self.loss_iter = config.loss_iter
        self.result_iter = config.result_iter
        self.checkpoints_iter = config.checkpoints_iter 
        self.logger_iter = config.logger_iter
        
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

        # Logger
        self.log_dir = config.log_dir
        self.logger = Logger(self.log_dir)

        # Test
        self.test_num = config.test_num

        # Basic setting
        self.basic_setting()

    def basic_setting(self):
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model
        self.generator = StarGAN_Generator(domain_dim=self.domain_dim, batch_size=self.batch_size, image_size=self.image_size).to(self.device)
        self.discriminator = StarGAN_Discriminator(domain_dim=self.domain_dim, image_size=self.image_size).to(self.device)

        # Criterion
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum')

        # Optimizer
        self.G_optimizer = optim.Adam(self.generator.parameters(), self.G_learning_rate, [self.beta1, self.beta2])
        self.D_optimizer = optim.Adam(self.discriminator.parameters(), self.D_learning_rate, [self.beta1, self.beta2])

        # For visualization
        images, labels = next(iter(self.data_loader))
        rand_idx = torch.randperm(self.batch_size)
        self.fixed_images = images
        fixed_trg_labels_A = torch.FloatTensor([[1, 0, 0, 1, 1]])
        self.fixed_trg_labels_A = fixed_trg_labels_A.repeat(repeats=[16, 1])
        rand_idx = torch.randperm(self.batch_size)
        self.fixed_trg_labels_B = labels[rand_idx]

    def train(self):
        start_iters = 0
        if self.resume_iters != None:
            start_iters = self.resume_iters
            ckpt_G = torch.load(os.path.join(self.save_generator_path, 'generator_epoch_{}.pt'.format(self.resume_iters)))
            ckpt_D = torch.load(os.path.join(self.save_discriminator_path, 'discriminator_epoch_{}.pt'.format(self.resume_iters)))
            self.generator.load_state_dict(ckpt_G['model'])
            self.discriminator.load_state_dict(ckpt_D['model'])
            
        for iters in range(start_iters, self.nb_iters):
            images, labels = next(iter(self.data_loader))
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Discriminator 학습
            # real image
            out_src, out_cls = self.discriminator(images)            
            real_src_loss = - torch.mean(out_src)
            out_cls = out_cls.reshape(out_cls.size(0), out_cls.size(1))
            real_cls_loss = self.criterion(out_cls, labels) / self.batch_size
            # fake image
            rand_idx = torch.randperm(labels.size(0))
            trg_labels = labels[rand_idx].to(self.device)
            fake_images = self.generator(images, trg_labels)            
            fake_out_src, _ = self.discriminator(fake_images)
            fake_src_loss = torch.mean(fake_out_src)

            # Gradient penalty loss
            alpha = torch.rand(images.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * images.data + (1 - alpha) * fake_images.data).requires_grad_(True)
            out_src, _ = self.discriminator(x_hat)
            gp_loss = Gradient_Penalty_Loss(out_src, x_hat, self.device).output()
            
            # Discriminator loss
            D_loss = self.lambda_real_src * real_src_loss + self.lambda_real_cls * real_cls_loss + self.lambda_fake_src * fake_src_loss + self.lambda_gp * gp_loss 

            self.D_optimizer.zero_grad()
            self.G_optimizer.zero_grad()
            D_loss.backward()
            self.D_optimizer.step()

            if iters % self.train_iter == 0:
                # Generator 학습
                fake_images = self.generator(images, trg_labels)
                G_out_src, G_out_cls = self.discriminator(fake_images)
                G_out_cls = G_out_cls.reshape(G_out_cls.size(0), G_out_cls.size(1))
                G_src_loss = - torch.mean(G_out_src)
                G_cls_loss = self.criterion(G_out_cls, trg_labels) / self.batch_size

                # Cyclic consistency loss
                recon_images = self.generator(fake_images, labels)
                recon_loss = torch.mean(torch.abs(recon_images - images))
                
                # Generator loss
                G_loss = self.lambda_src * G_src_loss + self.lambda_cls * G_cls_loss + self.lambda_recon * recon_loss

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

            # For visualization
            if iters % self.result_iter == 0:
                with torch.no_grad():
                    self.fixed_images = self.fixed_images.to(self.device)
                    self.fixed_trg_labels_A = self.fixed_trg_labels_A.to(self.device)
                    result_fake_images = self.generator(self.fixed_images, self.fixed_trg_labels_A)
                    result_fake_images = (result_fake_images + 1) / 2
                    result_images = torch.cat([self.fixed_images, result_fake_images], dim=0)
                    save_image(result_images, os.path.join(self.save_train_path, 'train_{}.png'.format(iters)))

            if iters % self.checkpoints_iter == 0 and iters > 0:
                Save_Checkpoints(iters, self.generator, self.save_generator_path, self.discriminator, self.save_discriminator_path, select='epoch')

            if iters % self.logger_iter == 0:
                for i in G_loss_dict:
                    self.logger.scalar_writer(tag='{}'.format(i), value=G_loss_dict[f'{i}'], step=iters)
                for i in D_loss_dict:
                    self.logger.scalar_writer(tag='{}'.format(i), value=D_loss_dict[f'{i}'], step=iters)
                # Print input image and fake image
                self.logger.image_writer('input_image', images)
                self.logger.image_writer('fake_image', fake_images)
                
    def test(self):
        ckpt = torch.load(os.path.join(self.save_generator_path, 'generator_epoch_{}.pt'.format(self.resume_iters)))
        self.generator.load_state_dict(ckpt['model'])
        self.generator.eval()
        with torch.no_grad():
            for idx, [images, labels] in enumerate(self.data_loader):
                images = images.to(self.device)
                rand_idx = torch.randperm(self.batch_size)
                trg_labels = labels[rand_idx].to(self.device)
                fake_images = self.generator(images, trg_labels)
                result_images = torch.cat([images, fake_images], dim=0)
                save_image(result_images, os.path.join(self.save_test_path, 'test_{}.png'.format(idx + 1)))
                if idx == self.test_num:
                    break