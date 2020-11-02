import torch
from torch import nn
from torch import optim
from config import Config
import utils
from model import *


def evaluate():
    # test_loader = utils.data_loader('Dig-MNIST')
    data = utils.load_data('Dig-MNIST')
    data = torch.from_numpy(data).to(Config.device)
    num_sample = 1000
    idx = torch.randint(data.size(0), size=(num_sample,))
    data = data[idx, :]
    img = data[:, 1:].float() / 255
    img = img.view(-1, 1, 28, 28)
    label = data[:, 0].long()

    F = SuperDuperFeatureExtractor().to(Config.device)
    C = SuperDuperClassifier().to(Config.device)

    F.load_state_dict(torch.load(Config.checkpoint + 'F.pth'))
    C.load_state_dict(torch.load(Config.checkpoint + 'C.pth'))
    F.eval()
    C.eval()

    # for idx, batch in enumerate(test_loader):
    #     img = batch['image']
    #     # img: [batch_size, 1, 28, 28]
    #     label = batch['label']
    #     # label: [batch_size,]
    #
    #     img = img.view(-1, 28 * 28)
    #
    #     feat = F(img)
    #     rec = R(feat)
    #     pred = C(feat)
    #
    #     # accuracy
    #     pred_label = pred.argmax(1)
    #     accuracy = (pred_label == label).sum().item() / label.size(0)
    #     print('Batch: {}, Test Accuracy: {}'.format(idx, accuracy))

    feat = F(img)
    pred = C(feat)

    # accuracy
    pred_label = pred.argmax(1)
    accuracy = (pred_label == label).sum().item() / label.size(0)
    print('Test Accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    evaluate()
