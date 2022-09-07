from JH_data_loader import CelebA_DATALOADER
from JH_solver import Solver

import argparse
import os
from torch.backends import cudnn

# For fast training
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='StarGAN Implementation by JH')

# DataLoader
parser.add_argument('--data_path', type=str, default='data/celeba/images/') # 상대 경로
parser.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt')
parser.add_argument('--target_attrs', nargs='+', type=str, default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'], help='length : 5')
parser.add_argument('--crop_size', type=int, default=178)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--mode', type=str, default='Train', choices=['Train', 'Test'])
parser.add_argument('--num_workers', type=int, default=1) # core 수의 절반

# Basic
parser.add_argument('--domain_dim', type=int, default=5)
parser.add_argument('--nb_iters', type=int, default=200000)
parser.add_argument('--resume_iters', type=int, default=None)
parser.add_argument('--G_learning_rate', type=float, default=0.0001)
parser.add_argument('--D_learning_rate', type=float, default=0.0001)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)

# Discriminator 
parser.add_argument('--lambda_real_src', type=float, default=1)
parser.add_argument('--lambda_real_cls', type=float, default=1)
parser.add_argument('--lambda_fake_src', type=float, default=1)
parser.add_argument('--lambda_gp', type=float, default=10)

# Generator 
parser.add_argument('--lambda_src', type=float, default=1)
parser.add_argument('--lambda_cls', type=float, default=1)
parser.add_argument('--lambda_recon', type=float, default=10)

# Save iteration
parser.add_argument('--train_iter', type=int, default=5)
parser.add_argument('--loss_iter', type=int, default=10)
parser.add_argument('--result_iter', type=int, default=1000)
parser.add_argument('--checkpoints_iter', type=int, default=10000)
parser.add_argument('--logger_iter', type=int, default=100)

# Image path
parser.add_argument('--save_results_path', type=str, default='stargan_celeba/results/')
parser.add_argument('--save_train_path', type=str, default='stargan_celeba/results/train')
parser.add_argument('--save_test_path', type=str, default='stargan_celeba/results/test')

# Model path
parser.add_argument('--save_model_path', type=str, default='stargan_celeba/models/')
parser.add_argument('--save_generator_path', type=str, default='stargan_celeba/models/generator/')
parser.add_argument('--save_discriminator_path', type=str, default='stargan_celeba/models/discriminator/')

# Logger
parser.add_argument('--log_dir', type=str, default='stargan_logger')

# Test
parser.add_argument('--test_num', type=int, default=10, help='smaller than epoch / batch_size')

config = parser.parse_args()

# Image path
if not os.path.exists(config.save_results_path):
    os.mkdir(config.save_results_path)
if not os.path.exists(config.save_train_path):
    os.mkdir(config.save_train_path)
if not os.path.exists(config.save_test_path):
    os.mkdir(config.save_test_path)

# Model path
if not os.path.exists(config.save_model_path):
    os.mkdir(config.save_model_path)
if not os.path.exists(config.save_generator_path):
    os.mkdir(config.save_generator_path)
if not os.path.exists(config.save_discriminator_path):
    os.mkdir(config.save_discriminator_path)
    
# Data loader
data_loader = CelebA_DATALOADER(data_path=config.data_path, attr_path=config.attr_path, target_attrs=config.target_attrs, crop_size=config.crop_size, image_size=config.image_size, batch_size=config.batch_size, mode=config.mode, num_workers=config.num_workers)
data_loader = data_loader.data_loader()

# Solver for Train, Test
solver = Solver(config, data_loader)

if __name__ == '__main__':
    if config.mode == 'Train':
        print(config)
        solver.train()
    elif config.mode == 'Test':
        print(config)
        solver.test()