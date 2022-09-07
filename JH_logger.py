from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

class Logger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
    def scalar_writer(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
    def generate_image(self, image): # 이 때의 image는 Tensor type
        fig = plt.figure(figsize=(12, 48))
    def image_writer(self, tag, image, step):
        self.writer.add_image(tag, image, step)