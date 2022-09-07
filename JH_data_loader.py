import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image

# Custom Dataset
class CelebA_DATASET(Dataset):
    def __init__(self, data_path, attr_path, target_attrs, transform, mode):
        self.data_path = data_path 
        self.attr_path = attr_path
        self.target_attrs = target_attrs
        self.transform = transform
        self.mode = mode
        self.labeling()

    def labeling(self):
        lines = [line.rstrip() for line in open(self.attr_path)]
        attr_name = lines[1].split()
        target_idx = []
        for i in range(len(attr_name)):
            if attr_name[i] in self.target_attrs: 
                target_idx.append(i)
        lines = lines[2:]
        # Test, Train Split
        if self.mode == 'Test':
            lines = lines[2:2000]
        elif self.mode == 'Train':
            lines = lines[2000:]
        line_fin = []
        for line in lines:
            line_attr = []
            line = line.split()
            image_file = line[0]
            for i in target_idx:
                if line[i + 1] == '-1':
                    line[i + 1] = 0
                elif line[i + 1] == '1':
                    line[i + 1] = 1
                line_attr.append(line[i + 1])
            # Hair attributes
            for j in range(len(line_attr[:3])):
                if line_attr[j] == 1:
                    line_attr[j] = 1
                    for k in range(len(line_attr[:3])):
                        if k!= j:
                            line_attr[k] = 0
            line_fin.append([image_file, line_attr])
        self.line_fin = line_fin

    def __len__(self):
        return len(self.line_fin)
    
    def __getitem__(self, index):
        image_path, label = self.line_fin[index]
        image_path = os.path.join(self.data_path, image_path)
        image = Image.open(image_path)
        image = self.transform(image)
        image_label = torch.FloatTensor(label)
        sample = [image, image_label]
        return sample

# DataLoader
class CelebA_DATALOADER(object):
    def __init__(self, data_path, attr_path, target_attrs, crop_size, image_size, batch_size, mode, num_workers):
        self.data_path = data_path
        self.attr_path = attr_path
        self.target_attrs = target_attrs
        self.crop_size = crop_size
        self.image_size = image_size
        self.batch_size = batch_size
        self.mode = mode
        self.num_workers = num_workers
        self.transform()

    def transform(self):
        transform_list = []
        if self.mode == 'Train':
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.CenterCrop(self.crop_size))
        transform_list.append(transforms.Resize(self.image_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = transforms.Compose(transform_list)
        self.sample = CelebA_DATASET(self.data_path, self.attr_path, self.target_attrs, transform, self.mode)

    def data_loader(self):
        data_loader = DataLoader(dataset=self.sample, batch_size=self.batch_size, shuffle=(self.mode=='Train'), num_workers=self.num_workers, drop_last=(self.mode=='Train'))
        return data_loader
