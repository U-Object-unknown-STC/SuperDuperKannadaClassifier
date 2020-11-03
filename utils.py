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
    def __init__(self, csv_filename, transform=None):
        self.data = load_data(csv_filename)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        image = self.data[item, 1:].reshape(28, 28, 1).astype(np.float32) / 255
        # image size: [28, 28, 1]
        label = torch.tensor(self.data[item, 0]).long()
        # label size: []

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image).float()

        sample = {'image': image, 'label': label}

        return sample


def data_loader(filename):
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        # handle with caution
        transforms.RandomErasing()
    ])
    dataset = KannadaDataset(csv_filename=filename, transform=trans)
    loader = DataLoader(dataset=dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0)
    return loader


if __name__ == '__main__':
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(30),
        transforms.ToTensor()
    ])
    dataset = KannadaDataset(csv_filename='Dig-MNIST', transform=trans)
    loader = DataLoader(dataset=dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0)
    for idx, batch in enumerate(loader):
        img = batch['image']
        label = batch['label']
        print(img.size())
        # [batch_size, 1, 28, 28]
        print(label.size())
        # [batch_size,]

        # plot sample
        for i in range(10):
            img_arr = img[i].numpy().reshape(28, 28)
            plt.subplot(2, 5, i+1)
            plt.imshow(img_arr, cmap='gray')
        plt.show()
