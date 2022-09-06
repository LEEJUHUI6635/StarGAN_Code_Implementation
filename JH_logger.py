from torch.utils.tensorboard import SummaryWriter
from JH_data_loader import CelebA_DATASET, CelebA_DATALOADER
from JH_model import StarGAN_Discriminator, StarGAN_Generator
from JH_main import config

import torchvision
from matplotlib.pyplot import imshow
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

log_dir = 'stargan_celeba/logs/'
writer = SummaryWriter(log_dir) # TensorBoard에 정보를 제공(write)하는 주요한 객체 정의

# image grid -> Input Image + 모델을 통과한 Image

# 모델 학습 추적 -> [Iteration - Training Loss(Discriminator Loss, Generator Loss)]

# TensorBoard에 기록하기 -> image -> Class 화
# 임의의 학습 이미지를 가져온다.
images, labels = next(iter(CelebA_DATALOADER.data_loader()))

# 이미지 그리드를 만든다.
img_grid = torchvision.utils.make_grid(images)

# 이미지를 보여준다.
imshow(img_grid, one_channel=False) # RGB

# tensorboard에 기록한다.
writer.add_image('Train_Images', img_grid)

# TensorBoard로 모델 학습 추적하기
generator = StarGAN_Generator(domain_dim=config.domain_dim, batch_size=config.batch_size, image_size=config.image_size)
discriminator = StarGAN_Discriminator(domain_dim=config.domain_dim, image_size=config.image_size)

start_iteration = 0
criterion = nn.BCEWithLogitsLoss() # batch_size로 나눠줘야 하는가?

def plot_classes_preds(generator, images, labels, idx):
    rand_idx = torch.randperm(config.batch_size)
    trg_labels = labels[rand_idx]

    fake_images = generator(images, trg_labels)
    plt.plot(fake_images)
for iteration in range(start_iteration, config.nb_iters):
    images, labels = next(iter(CelebA_DATALOADER.data_loader()))
    
    rand_idx = torch.randperm(config.batch_size)
    trg_labels = labels[rand_idx]

    # Discriminator 학습
    # real image
    D_real_src, D_real_cls = discriminator(images)

    D_real_src_loss = -torch.mean(D_real_src)
    D_real_cls_loss = criterion(D_real_cls, labels)

    # fake image
    fake_images = generator(images, trg_labels)
    D_fake_src, D_fake_cls = discriminator(fake_images)

    D_fake_src_loss = torch.mean(D_fake_src)
    
    D_loss = D_real_src_loss + D_real_cls_loss + D_fake_src_loss

    # Generator 학습
    fake_images = generator(images, trg_labels)
    G_fake_src, G_fake_cls = discriminator(fake_images)

    G_fake_src_loss = -torch.mean(G_fake_src)
    G_fake_cls_loss = criterion(G_fake_cls, trg_labels)

    recon_images = generator(fake_images, labels)
    G_recon_loss = torch.mean(torch.abs(recon_images - images))

    G_loss = G_fake_src_loss + G_fake_cls_loss + G_recon_loss

    # 매 100 iteration마다 loss 출력
    if iteration % 100 == 0:    
        writer.add_scalar('G_loss', G_loss, 100) # tag, scalar_value, global_step, ...
        writer.add_scalar('D_loss', D_loss, 100)
        writer.add_figure('input VS predicted', plot_classes_preds, 100) # tag, figure, global_step, ...
        # input image와 predicted image를 같이 보여준다.




# Test 평가 -> PR curve
