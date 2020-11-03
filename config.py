import torch
import sys


class Config:
    base_dir = './Kannada-MNIST/' if sys.platform.startswith('win') else '/mnt/data/fyy/kannada/'
    batch_size = 1024
    learning_rate = 1e-3
    num_epoch = 10 if sys.platform.startswith('win') else 40
    # gpu num
    device = 0 if torch.cuda.is_available() else 'cpu'
    checkpoint = './checkpoint/'
    use_pretrain = True
    enable_plot = True
