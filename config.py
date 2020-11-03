import torch
import sys


class Config:
    # basic configurations
    base_dir = './Kannada-MNIST/' if sys.platform.startswith('win') else '/mnt/data/fyy/kannada/'
    # gpu num
    device = 0 if torch.cuda.is_available() else 'cpu'
    checkpoint = './checkpoint/'
    use_pretrain = False
    enable_plot = True

    # hyper parameters
    batch_size = 512
    learning_rate = 1e-3
    num_epoch = 40 if sys.platform.startswith('win') else 40
    lambda_msda = 1e-1
