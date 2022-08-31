# config로 hyper parameter 정의, train or test

from JH_data_loader import CelebA_DATASET, CelebA_DATALOADER
from JH_model import StarGAN_Generator, StarGAN_Discriminator
from JH_solver import Solver

import argparse

# Hyper parameter 정의

parser = argparse.ArgumentParser(description='StarGAN Study by JH') # 인자값을 받을 수 있는 인스턴스 생성

parser.add_argument('--data_path', type=str, default='data/celeba/images/') # 상대 경로
parser.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt')
parser.add_argument('--target_attrs', nargs='+', type=str, default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']) # list
parser.add_argument('--crop_size', type=int, default=178)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--mode', type=str, default='Train')
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--domain_dim', type=int, default=5)
parser.add_argument('--nb_epochs', type=int, default=200000)
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

config = parser.parse_args() # config에 위의 내용 저장

# Data Loader
data_loader = CelebA_DATALOADER(data_path=config.data_path, attr_path=config.attr_path, target_attrs=config.target_attrs, crop_size=config.crop_size, image_size=config.image_size, batch_size=config.batch_size, mode=config.mode, num_workers=config.num_workers)
data_loader = data_loader.data_loader()

# Solver
solver = Solver(config, data_loader)

if __name__ == '__main__':
    print(config)
    # train or test
    if config.mode == 'Train':
        solver.train()
    elif config.mode == 'Test':
        solver.test()