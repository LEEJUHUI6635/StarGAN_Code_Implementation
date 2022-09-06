from JH_data_loader import CelebA_DATASET, CelebA_DATALOADER

data_path = 'data/celeba/images/' # 상대 경로
attr_path = 'data/celeba/list_attr_celeba.txt'
target_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
crop_size = 178
image_size = 128
batch_size = 16
mode = 'Train'
num_workers = 1

data_loader = CelebA_DATALOADER(data_path=data_path, attr_path=attr_path, target_attrs=target_attrs, crop_size=crop_size, image_size=image_size, batch_size=batch_size, mode=mode, num_workers=num_workers).data_loader()

# image, label = next(iter(data_loader))

# print(image)
# print(label)

for i in range(20):
    image, label = next(iter(data_loader))
    print(image)
    print(label)