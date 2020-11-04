import pandas as pd
import utils
import torch
from config import *
from model import *
import matplotlib.pyplot as plt


def predict():
    # test_loader = utils.data_loader('Dig-MNIST')
    data = utils.load_data('test')
    data = torch.from_numpy(data).to(Config.device)
    num_sample = data.size(0)
    idx = torch.randint(data.size(0), size=(num_sample,))
    data = data[idx, :]
    img = data[:, 1:].float() / 255
    img = img.view(-1, 1, 28, 28)
    idx = data[:, 0].long()

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

    curr_label = 0
    pred_label = []
    num_sample = img.size(0)
    for i in range(num_sample):
        tmp_img = img[i]
        tmp_img = tmp_img.view(-1, 1, 28, 28)
        feat = F(tmp_img)
        pred = C(feat)
        tmp_pred_label = pred.argmax(1)
        pred_label.append(tmp_pred_label.item())

        # plot
        plt.figure('Test Sample Predictions')
        if tmp_pred_label.item() == curr_label:
            curr_label += 1
            tmp_img_arr = tmp_img.cpu().numpy()
            tmp_img_arr = tmp_img_arr.reshape(28, 28)
            plt.subplot(2, 5, curr_label)
            plt.imshow(tmp_img_arr, cmap='gray')
            plt.title(tmp_pred_label.item())

        if curr_label == 10:
            break

    plt.show()

    # accuracy
    # submission = pd.DataFrame({'id': idx.tolist()})
    # submission = submission.join(pd.DataFrame({'label': pred_label}))
    # submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    predict()
