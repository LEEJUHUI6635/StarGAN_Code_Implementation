# Data Loader
# Image 불러오기
# list_attr_celeba.txt 파일에서 ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'] attribute에 대한 one-hot encoding vector를 생성

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset # 샘플과 정답(label)을 저장한다.
from torch.utils.data import DataLoader
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# 해당 폴더 안의 데이터 파일 경로를 리스트에 담아두는 과정이 필요하다.
data_path = 'data/celeba/images/' # 상대 경로
attr_path = 'data/celeba/list_attr_celeba.txt'
target_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
crop_size = 178
image_size = 128
batch_size = 16
mode = 'Train'
num_workers = 1

# Custom Dataset
class CelebA_DATASET(Dataset):
    def __init__(self, data_path, attr_path, target_attrs, transform):
        self.data_path = data_path # data 경로
        self.attr_path = attr_path # attribute 경로
        self.target_attrs = target_attrs # target attributes
        self.transform = transform
        self.labeling()

    def labeling(self): # image + one-hot vector
        lines = [line.rstrip() for line in open(attr_path)]
        attr_name = lines[1].split() # attribute name
        target_idx = []
        # attribute name에 해당하는 index 추출
        for i in range(len(attr_name)):
            if attr_name[i] in self.target_attrs:
                target_idx.append(i)
        lines = lines[2:] # line 하나의 첫 번째 값은 image명

        line_fin = []
        for line in lines:
            line_attr = []
            line = line.split()
            image_file = line[0]
            for i in target_idx:
                if line[i] == '-1':
                    line[i] = 0
                elif line[i] == '1':
                    line[i] = 1
                line_attr.append(line[i]) # attribute -> one-hot encoding
            line_fin.append([image_file, line_attr])

        self.line_fin = line_fin

    def __len__(self): # dataset 크기, shuffle을 위해서는 dataset 크기 선언해야 한다.
        return len(self.line_fin)

    def __getitem__(self, index): # data sample을 이에 해당하는 label과 함께 불러온다. for문을 쓰지 않아도 for문의 역할을 한다. 따라서 getitem은 index 인자와 함께 쓰여야 한다.
        # self.line_fin [image, label]을 가져와서, os.path.join(attr_path, image)의 경로에서 이미지를 가져오고, label return
        image_path, label = self.line_fin[index] # debugging
        image_path = image_path
        image_path = os.path.join(self.data_path, image_path)

        # image_path에서 해당하는 image 불러오기
        image = Image.open(image_path)
        image_label = torch.FloatTensor(label)

        # image에 대한 transform
        image = self.transform(image)

        # image + label -> list
        sample = [image, image_label]

        return sample # class의 최종 output

# DataLoader
class CelebA_DATALOADER(object):
    def __init__(self, data_path, attr_path, target_attrs, crop_size, image_size, batch_size, mode='Train', num_workers=num_workers):
        self.data_path = data_path
        self.attr_path = attr_path
        self.target_attrs = target_attrs
        self.crop_size = crop_size
        self.image_size = image_size
        self.batch_size = batch_size
        self.mode = mode
        self.num_workers = num_workers
        self.transform()
        # self.data_loader()

    def transform(self):
        transform_list = []
        if self.mode == 'Train':
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.CenterCrop(self.crop_size))
        transform_list.append(transforms.Resize(self.image_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = transforms.Compose(transform_list)
        self.sample = CelebA_DATASET(self.data_path, self.attr_path, self.target_attrs, transform)

    def data_loader(self):
        data_loader = DataLoader(dataset=self.sample, batch_size=self.batch_size, shuffle=(self.mode=='Train'), num_workers=self.num_workers, drop_last=(self.mode=='Train'))

        return data_loader

# data_loader = CelebA_DATALOADER(data_path, attr_path, target_attrs, crop_size, image_size, batch_size, mode)

# for idx, [image, label] in enumerate(data_loader.data_loader()):
#     print(image)
#     print(label)
#     print(type(image)) # Tensor
#     print(type(label)) # Tensor
#     break
