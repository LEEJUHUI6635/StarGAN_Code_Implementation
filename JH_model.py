import torch
import torch.nn as nn

# Residual Block
class Residual_Block(nn.Module):
    def __init__(self):
        super(Residual_Block, self).__init__()
        residual_list = []
        residual_list.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False))
        residual_list.append(nn.InstanceNorm2d(num_features=256, affine=True, track_running_stats=True))
        residual_list.append(nn.ReLU(inplace=True))
        residual_list.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False))
        residual_list.append(nn.InstanceNorm2d(num_features=256, affine=True, track_running_stats=True))
        self.residual_block = nn.Sequential(*residual_list)

    def forward(self, x):
        out = self.residual_block(x) + x
        return out

# Generator
# input -> [height, width, 3 + domain]
# output -> [height, width, 3]
class StarGAN_Generator(nn.Module):
    def __init__(self, domain_dim=5, batch_size=16, image_size=128):
        super(StarGAN_Generator, self).__init__()
        self.domain_dim = domain_dim
        self.conv_dim = 64
        self.batch_size = batch_size
        self.image_size = image_size
        self.generator_list = []
        self.down_sampling() 
        self.bottleneck() 
        self.up_sampling()

    def down_sampling(self):
        self.generator_list.append(nn.Conv2d(in_channels=3+self.domain_dim, out_channels=self.conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        self.generator_list.append(nn.InstanceNorm2d(num_features=self.conv_dim, affine=True, track_running_stats=True))
        self.generator_list.append(nn.ReLU(inplace=True))
        for i in range(2):
            self.generator_list.append(nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            self.generator_list.append(nn.InstanceNorm2d(num_features=self.conv_dim*2, affine=True, track_running_stats=True))
            self.generator_list.append(nn.ReLU(inplace=True))
            self.conv_dim *= 2
    
    def bottleneck(self):
        for i in range(6):
            self.generator_list.append(Residual_Block())
    
    def up_sampling(self):
        for i in range(2):
            self.generator_list.append(nn.ConvTranspose2d(in_channels=self.conv_dim, out_channels=self.conv_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            self.generator_list.append(nn.InstanceNorm2d(num_features=self.conv_dim//2, affine=True, track_running_stats=True))
            self.generator_list.append(nn.ReLU(inplace=True))
            self.conv_dim = self.conv_dim // 2
        self.generator_list.append(nn.Conv2d(in_channels=self.conv_dim, out_channels=3, kernel_size=7, stride=1, padding=3, bias=False))
        self.generator_list.append(nn.Tanh())
        self.generator_block = nn.Sequential(*self.generator_list) # nn : parameter 선언 -> self로 받아 와서 우회해야 한다.

    def forward(self, image, label):
        label = label.reshape(self.batch_size, self.domain_dim, 1, 1)
        label = label.repeat(1, 1, self.image_size, self.image_size)
        out = torch.cat([image, label], dim = 1)
        out = self.generator_block(out)
        return out

# Discriminator
# input -> [height, width, 3]
# output -> [height//64, width//64, 1], [1, 1, domain]
class StarGAN_Discriminator(nn.Module):
    def __init__(self, domain_dim, image_size):
        super(StarGAN_Discriminator, self).__init__()
        self.discriminator_layer = []
        self.domain_dim = domain_dim
        self.conv_dim = 64
        self.image_size = image_size
        self.input_layer()
        self.hidden_layer()
        self.output_layer()

    def input_layer(self):
        self.discriminator_layer.append(nn.Conv2d(in_channels=3, out_channels=self.conv_dim, kernel_size=4, stride=2, padding=1))
        self.discriminator_layer.append(nn.LeakyReLU(negative_slope=0.01))

    def hidden_layer(self):
        for i in range(5):
            self.discriminator_layer.append(nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim*2, kernel_size=4, stride=2, padding=1))
            self.discriminator_layer.append(nn.LeakyReLU(negative_slope=0.01))
            self.conv_dim *= 2

    def output_layer(self):
        self.discriminator_block = nn.Sequential(*self.discriminator_layer)
        self.src_out = nn.Conv2d(in_channels=self.conv_dim, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.cls_out = nn.Conv2d(in_channels=self.conv_dim, out_channels=self.domain_dim, kernel_size=self.image_size//64, stride=1, padding=0, bias=False)
        
    def forward(self, image):
        hidden = self.discriminator_block(image)
        out_src = self.src_out(hidden)
        out_cls = self.cls_out(hidden)
        return out_src, out_cls
