import pandas as pd
from config import Config
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


def load_data(filename):
    """
    :param filename: name of csv file
    :return: numpy array of shape (N, 785)
    """
    data_df = pd.read_csv(Config.base_dir + filename + '.csv')
    # print(type(data_df))   # DataFrame
    # print(type(data_df.values))    # numpy array
    # print(data_df.values.shape)    # (N, 785)
    # data = torch.from_numpy(data_df.values)
    return data_df.values


class KannadaDataset(Dataset):
    def __init__(self, source_csv_filename, target_csv_filename, transform=None):
        self.source_data = load_data(source_csv_filename)
        self.target_data = load_data(target_csv_filename)
        self.transform = transform

    def __len__(self):
        # fixme: probably with bugs
        min_len = min(len(self.source_data), len(self.target_data))
        return min_len

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        source_image = self.source_data[item, 1:].reshape(28, 28, 1).astype(np.float32) / 255
        target_image = self.target_data[item, 1:].reshape(28, 28, 1).astype(np.float32) / 255
        # image size: [28, 28, 1]
        source_label = torch.tensor(self.source_data[item, 0]).long()
        target_label = torch.tensor(self.target_data[item, 0]).long()
        # label size: []

        # do not augment target sample
        if self.transform:
            source_image = self.transform(source_image)
        else:
            source_image = torch.tensor(source_image.reshape(-1, 28, 28)).float()
        target_image = torch.tensor(target_image.reshape(-1, 28, 28)).float()

        sample = {
            'source_image': source_image,
            'source_label': source_label,
            'target_image': target_image,
            'target_label': target_label
        }

        return sample


def data_loader(source, target):
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(30),
        transforms.ToTensor()
        # handle with caution
        # transforms.RandomErasing()
    ])
    dataset = KannadaDataset(source_csv_filename=source, target_csv_filename=target, transform=trans)
    loader = DataLoader(dataset=dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0)
    return loader


if __name__ == '__main__':
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.RandomErasing()
    ])
    dataset = KannadaDataset(source_csv_filename='train', target_csv_filename='Dig-MNIST', transform=trans)
    loader = DataLoader(dataset=dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0)
    for idx, batch in enumerate(loader):
        s_img = batch['source_image']
        t_img = batch['target_image']
        s_label = batch['source_label']
        t_label = batch['target_label']
        print(s_img.size())
        # [batch_size, 1, 28, 28]
        print(s_label.size())
        # [batch_size,]

        # plot sample
        plt.figure('Source')
        for i in range(10):
            s_img_arr = s_img[i].numpy().reshape(28, 28)
            plt.subplot(2, 5, i+1)
            plt.imshow(s_img_arr, cmap='gray')
        plt.show()

        plt.figure('Target')
        for i in range(10):
            t_img_arr = t_img[i].numpy().reshape(28, 28)
            plt.subplot(2, 5, i+1)
            plt.imshow(t_img_arr, cmap='gray')
        plt.show()
