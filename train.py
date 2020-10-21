import torch
from torch import nn
from torch import optim
from config import Config
import utils
from model import *


def train():
    print('Load training data...')
    train_loader = utils.data_loader('train')
    print('Done!')

    # model
    # F = SuperDuperFeatureExtractor()
    # R = SuperDuperReconstructor()
    F = nn.Sequential(
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 128)
    )
    R = nn.Sequential(
        nn.Linear(128, 256),
        nn.Linear(256, 28 * 28)
    )
    C = SuperDuperClassifier()

    F.to(Config.device)
    R.to(Config.device)
    C.to(Config.device)

    op_F = optim.Adam(F.parameters(), lr=Config.learning_rate)
    op_R = optim.Adam(R.parameters(), lr=Config.learning_rate)
    op_C = optim.Adam(C.parameters(), lr=Config.learning_rate)

    criterion = nn.CrossEntropyLoss()

    accuracy = 0

    print('Training...')
    for epoch in range(Config.num_epoch):
        for idx, batch in enumerate(train_loader):
            img = batch['image'].to(Config.device)
            # img: [batch_size, 1, 28, 28]
            label = batch['label'].to(Config.device)
            # label: [batch_size,]

            img = img.view(-1, 28 * 28)

            feat = F(img)
            rec = R(feat)
            pred = C(feat)

            loss_s = criterion(pred, label)
            loss_r = 1e-4 * ((img - rec) ** 2).sum()

            loss = loss_s + loss_r

            op_F.zero_grad()
            op_R.zero_grad()
            op_C.zero_grad()

            loss.backward()

            op_F.step()
            op_R.step()
            op_C.step()

            # accuracy
            pred_label = pred.argmax(1)
            accuracy = (pred_label == label).sum().item() / label.size(0)

        print('Epoch: {}, Loss_s: {}, Loss_r: {}, Accuracy: {}'.format(epoch, loss_s, loss_r, accuracy))

    print('Done!')

    torch.save(F.state_dict(), './checkpoint/F.pth')
    torch.save(R.state_dict(), './checkpoint/R.pth')
    torch.save(C.state_dict(), './checkpoint/C.pth')


if __name__ == '__main__':
    train()
