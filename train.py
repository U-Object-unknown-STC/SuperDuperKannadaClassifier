import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from config import Config
import utils
from model import *
import matplotlib.pyplot as plt
from metric import *


def train():
    print('Load training data...')
    train_loader = utils.data_loader('train', 'test')
    print('Done!')

    # model
    F = SuperDuperFeatureExtractor()
    C = SuperDuperClassifier()

    # load pretrained model
    if Config.use_pretrain:
        F.load_state_dict(torch.load(Config.checkpoint + 'F.pth'))
        C.load_state_dict(torch.load(Config.checkpoint + 'C.pth'))

    F.to(Config.device)
    C.to(Config.device)

    op_F = optim.Adam(F.parameters(), lr=Config.learning_rate)
    op_C = optim.Adam(C.parameters(), lr=Config.learning_rate)

    criterion = nn.CrossEntropyLoss()

    accuracy_s = accuracy_v = accuracy_t = 0

    # plot training route
    plot_loss = []
    plot_s_acc = []
    plot_v_acc = []
    plot_t_acc = []

    print('Training...')
    for epoch in range(Config.num_epoch):
        for idx, batch in enumerate(train_loader):
            s_img = batch['source_image'].to(Config.device)
            t_img = batch['target_image'].to(Config.device)
            v_img = batch['val_image'].to(Config.device)
            # img: [batch_size, 1, 28, 28]
            s_label = batch['source_label'].to(Config.device)
            v_label = batch['val_label'].to(Config.device)
            t_label = batch['target_label'].to(Config.device)
            # label: [batch_size,]

            feat_s = F(s_img)
            feat_v = F(v_img)
            feat_t = F(t_img)
            pred_s = C(feat_s)
            pred_v = C(feat_v)
            pred_t = C(feat_t)

            loss_s = criterion(pred_s, s_label)
            loss_msda = k_moment([feat_t, feat_s], k=4)

            loss = loss_s + Config.lambda_msda * loss_msda

            op_F.zero_grad()
            op_C.zero_grad()

            loss.backward()

            op_F.step()
            op_C.step()

            # accuracy
            pred_label_s = pred_s.argmax(1)
            accuracy_s = (pred_label_s == s_label).sum().item() / s_label.size(0)
            pred_label_v = pred_v.argmax(1)
            accuracy_v = (pred_label_v == v_label).sum().item() / v_label.size(0)
            pred_label_t = pred_t.argmax(1)
            accuracy_t = (pred_label_t == t_label).sum().item() / t_label.size(0)

            # plot history
            plot_loss.append(loss.item())
            plot_s_acc.append(accuracy_s)
            plot_v_acc.append(accuracy_v)
            plot_t_acc.append(accuracy_t)

        print('Epoch: {}, Loss_s: {}, Loss_msda: {}, Accuracy_s: {}, Accuracy_v: {}, Accuracy_t: {}'.format(
            epoch, loss_s, loss_msda, accuracy_s, accuracy_v, accuracy_t
        ))

    if Config.enable_plot:
        plt.figure('Accuracy & Loss')
        plt.ylim(0.0, 3.2)
        plt.plot(plot_loss, label='Loss')
        plt.plot(plot_s_acc, label='Source Accuracy')
        plt.plot(plot_v_acc, label='Validate Accuracy')
        plt.plot(plot_t_acc, label='Target Accuracy')
        plt.xlabel('Batch')
        plt.title('Train Accuracy & Loss')
        plt.legend()
        plt.show()

    print('Done!')

    # save model
    torch.save(F.state_dict(), Config.checkpoint + 'F.pth')
    torch.save(C.state_dict(), Config.checkpoint + 'C.pth')


if __name__ == '__main__':
    train()
