## StarGAN Study 

```bash
# Train StarGAN using the CelebA dataset
python main.py --mode train --dataset CelebA --image_size 128 --c_dim 5 \
               --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs \
               --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results \
               --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young

# Test StarGAN using the CelebA dataset
python main.py --mode test --dataset CelebA --image_size 128 --c_dim 5 \
               --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs \
               --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results \
               --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young
```

To train StarGAN on RaFD:

```bash
# Train StarGAN using the RaFD dataset
python main.py --mode train --dataset RaFD --image_size 128 \
               --c_dim 8 --rafd_image_dir data/RaFD/train \
               --sample_dir stargan_rafd/samples --log_dir stargan_rafd/logs \
               --model_save_dir stargan_rafd/models --result_dir stargan_rafd/results

# Test StarGAN using the RaFD dataset
python main.py --mode test --dataset RaFD --image_size 128 \
               --c_dim 8 --rafd_image_dir data/RaFD/test \
               --sample_dir stargan_rafd/samples --log_dir stargan_rafd/logs \
               --model_save_dir stargan_rafd/models --result_dir stargan_rafd/results
```

To train StarGAN on both CelebA and RafD:

```bash
# Train StarGAN using both CelebA and RaFD datasets
python main.py --mode=train --dataset Both --image_size 256 --c_dim 5 --c2_dim 8 \
               --sample_dir stargan_both/samples --log_dir stargan_both/logs \
               --model_save_dir stargan_both/models --result_dir stargan_both/results

# Test StarGAN using both CelebA and RaFD datasets
python main.py --mode test --dataset Both --image_size 256 --c_dim 5 --c2_dim 8 \
               --sample_dir stargan_both/samples --log_dir stargan_both/logs \
               --model_save_dir stargan_both/models --result_dir stargan_both/results
```

To train StarGAN on your own dataset, create a folder structure in the same format as [RaFD](https://github.com/yunjey/StarGAN/blob/master/jpg/RaFD.md) and run the command:

```bash
# Train StarGAN on custom datasets
python main.py --mode train --dataset RaFD --rafd_crop_size CROP_SIZE --image_size IMG_SIZE \
               --c_dim LABEL_DIM --rafd_image_dir TRAIN_IMG_DIR \
               --sample_dir stargan_custom/samples --log_dir stargan_custom/logs \
               --model_save_dir stargan_custom/models --result_dir stargan_custom/results

# Test StarGAN on custom datasets
python main.py --mode test --dataset RaFD --rafd_crop_size CROP_SIZE --image_size IMG_SIZE \
               --c_dim LABEL_DIM --rafd_image_dir TEST_IMG_DIR \
               --sample_dir stargan_custom/samples --log_dir stargan_custom/logs \
               --model_save_dir stargan_custom/models --result_dir stargan_custom/results

# Download Data in my Google Drive
https://drive.google.com/drive/folders/1nSYZeW_oSaZvBcPVLMSEykmYHTX04Juy
