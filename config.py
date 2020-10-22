import torch


class Config:
    base_dir = './Kannada-MNIST/'
    batch_size = 128
    learning_rate = 1e-3
    num_epoch = 10
    # gpu num
    device = 0 if torch.cuda.is_available() else 'cpu'
    checkpoint = './checkpoint/'
