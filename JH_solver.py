# train, test
from JH_data_loader import CelebA_DATASET, CelebA_DATALOADER
from JH_model import StarGAN_Discriminator, StarGAN_Generator

class Solver(object):
    def __init__(self, config):
        # hyper parameter -> main에서 config 인자를 받아온다.
        
        self.train()
        self.test()

    def train(self):
        print()
    def test(self):
        print()
