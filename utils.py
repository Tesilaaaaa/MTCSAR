import glob
import numpy as np
import h5py
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset
import torch
import torch.nn as nn
# import matplotlib.pyplot as plt
import os

patchSize = 40
stride = patchSize
center = 4

# Ref400Path = 'G:\\SARData\\JingChengSLC-temporalMultilook\\train-val-test\\train\\Ref400\\'
# Noisy400Path = 'G:\\SARData\\JingChengSLC-temporalMultilook\\train-val-test\\train\\Noisy400\\'
# CC400Path = 'G:\\SARData\\JingChengSLC-temporalMultilook\\train-val-test\\train\\CC400\\'


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        # nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def dataAug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img, axes=(2, 3))
    elif mode == 3:
        return np.flipud(np.rot90(img, axes=(2, 3)))
    elif mode == 4:
        return np.rot90(img, k=2, axes=(2, 3))
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2, axes=(2, 3)))
    elif mode == 6:
        return np.rot90(img, k=3, axes=(2, 3))
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3, axes=(2, 3)))


def psnr(denoised_img, truth_img):
    mse = torch.mean((denoised_img - truth_img) ** 2)
    if mse == 0:
        return float('inf')
    else:
        return 20 * torch.log10(1 / torch.sqrt(mse))


def dataGenerator(truthPath, noisyPath, CCPath):
    target_batch = []
    train_batch = []
    CC_batch = []
    trueImgList = sorted(glob.glob(truthPath + '/*.mat'))
    noisyImgList = sorted(glob.glob(noisyPath + '/*.mat'))
    CCImgList = sorted(glob.glob(CCPath + '/*.mat'))

    for i in range(len(noisyImgList)):

        truthImg = h5py.File(trueImgList[i], 'r')
        truthImg = np.array(truthImg['Ref400multi']).astype(np.float32)
        truthImg = np.transpose(truthImg)
        noisyImg = h5py.File(noisyImgList[i], 'r')
        noisyImg = np.array(noisyImg['Noisy400Multi']).astype(np.float32)
        noisyImg = np.transpose(noisyImg)
        CCImg = h5py.File(CCImgList[i], 'r')
        CCImg = np.array(CCImg['CC400Multi']).astype(np.float32)
        CCImg = np.transpose(CCImg)
        n,c,H,W = truthImg.shape

        # truthImg=np.log(truthImg+1)
        # noisyImg=np.log(noisyImg+1)
        truthImg = truthImg / np.max(truthImg)
        noisyImg = noisyImg / np.max(noisyImg)

        noisyPatch = []
        truthPatch = []
        CCPatch = []

        for i in range(0, H - patchSize + 1, patchSize//2):
            for j in range(0, W - patchSize + 1, patchSize//2):
                true = truthImg[:, :, i:i + patchSize, j:j + patchSize]
                noisy = noisyImg[:, :, i:i + patchSize, j:j + patchSize]
                CC = CCImg[:, :, i:i + patchSize, j:j + patchSize]
                mode = np.random.randint(0, 8)
                trueAug = dataAug(true, mode)
                noisyAug = dataAug(noisy, mode)
                CCAug = dataAug(CC, mode)
                noisyPatch.append(noisyAug)
                truthPatch.append(trueAug)
                CCPatch.append(CCAug)

        for truth in truthPatch:
            target_batch.append(truth)
        for noisy in noisyPatch:
            train_batch.append(noisy)
        for CC in CCPatch:
            CC_batch.append(CC)

    print('共有patch个数：', len(target_batch), len(train_batch), len(CC_batch))
    train_batch = np.array(train_batch, dtype='float32')
    target_batch = np.array(target_batch, dtype='float32')
    CC_batch = np.array(CC_batch, dtype='float32')
    target_batch = torch.from_numpy(target_batch).view(-1, 5, 1, patchSize, patchSize)
    train_batch = torch.from_numpy(train_batch).view(-1, 5, 1, patchSize, patchSize)
    CC_batch = torch.from_numpy(CC_batch).view(-1, 5 - 1, 1, patchSize, patchSize)

    return target_batch, train_batch, CC_batch


def dataGenerator_single(truthPath, noisyPath):
    target_batch = []
    train_batch = []
    CC_batch = []
    trueImgList = sorted(glob.glob(truthPath + '/*.mat'))
    noisyImgList = sorted(glob.glob(noisyPath + '/*.mat'))

    for i in range(len(noisyImgList)):

        truthImg = h5py.File(trueImgList[i], 'r')
        truthImg = np.array(truthImg['Ref400multi']).astype(np.float32)
        truthImg = np.transpose(truthImg)
        noisyImg = h5py.File(noisyImgList[i], 'r')
        noisyImg = np.array(noisyImg['Noisy400Multi']).astype(np.float32)
        noisyImg = np.transpose(noisyImg)
        CCImg = h5py.File(noisyImgList[i], 'r')
        CCImg = np.array(CCImg['Noisy400Multi']).astype(np.float32)
        CCImg = np.transpose(CCImg)
        N, C, H, W = truthImg.shape

        # truthImg=np.log(truthImg+1)
        # noisyImg=np.log(noisyImg+1)
        truthImg = (truthImg[4,:,:,:] / np.max(truthImg[4,:,:,:]))[np.newaxis,:]
        noisyImg = (noisyImg[4,:,:,:] / np.max(noisyImg[4,:,:,:]))[np.newaxis,:]
        CCImg=CCImg[4,:,:,:][np.newaxis,:]

        noisyPatch = []
        truthPatch = []
        CCPatch = []

        for i in range(0, H - patchSize + 1,  patchSize//2):
            for j in range(0, W - patchSize + 1,  patchSize//2):
                true = truthImg[:, :, i:i + patchSize, j:j + patchSize]
                noisy = noisyImg[:, :, i:i + patchSize, j:j + patchSize]
                CC = CCImg[:, :, i:i + patchSize, j:j + patchSize]
                mode = np.random.randint(0, 8)
                trueAug = dataAug(true, mode)
                noisyAug = dataAug(noisy, mode)
                CCAug = dataAug(CC, mode)
                noisyPatch.append(noisyAug)
                truthPatch.append(trueAug)
                CCPatch.append(CCAug)

        for truth in truthPatch:
            target_batch.append(truth)
        for noisy in noisyPatch:
            train_batch.append(noisy)
        for CC in CCPatch:
            CC_batch.append(CC)

    print('共有patch个数：', len(target_batch), len(train_batch), len(CC_batch))
    train_batch = np.array(train_batch, dtype='float32')
    target_batch = np.array(target_batch, dtype='float32')
    CC_batch = np.array(CC_batch, dtype='float32')
    target_batch = torch.from_numpy(target_batch).view(-1, 1, C, patchSize, patchSize)
    train_batch = torch.from_numpy(train_batch).view(-1, 1, C, patchSize, patchSize)
    CC_batch = torch.from_numpy(CC_batch).view(-1, 1, C, patchSize, patchSize)

    return target_batch, train_batch, CC_batch

class get_dataset(Dataset):

    def __init__(self, noisyData, truthData, CCData):
        super(get_dataset, self).__init__()
        self.noisyData = noisyData
        self.truthData = truthData
        self.CCData = CCData

    def __getitem__(self, index):
        speck_data_b = self.noisyData[index]
        truth_data_b = self.truthData[index]
        cc_data_b = self.CCData[index]

        return speck_data_b.float(), truth_data_b.float(), cc_data_b.float()

    def __len__(self):
        return self.noisyData.size(0)


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, x, y,):
        loss1 = torch.mean(torch.sqrt(torch.pow(y - x, 2)+1e-6))
        return loss1


# def draw_loss(Loss_list, val_Loss_List, path):
#     # 我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
#     plt.cla()
#     x1 = range(1, len(Loss_list) + 1)
#     y1 = Loss_list
#     y2 = val_Loss_List
#     plt.title('loss vs. epoches', fontsize=5)
#     plt.plot(x1, y1, '.-')
#     plt.plot(x1, y2, '.-')
#     plt.xlabel('epoches', fontsize=5)
#     plt.ylabel('loss', fontsize=5)
#     plt.grid()
#     plt.savefig(os.path.join(path, "loss.png"))
#     plt.show()
#
#
# def draw_PSNR(PSNR_List, path):
#     # 我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
#     plt.cla()
#     x1 = range(1, len(PSNR_List) + 1)
#     y1 = PSNR_List
#     plt.title('Val PSNR vs. epoches', fontsize=5)
#     plt.plot(x1, y1, '.-')
#     plt.xlabel('epoches', fontsize=5)
#     plt.ylabel('PSNR', fontsize=5)
#     plt.grid()
#     plt.savefig(os.path.join(path, "psnr.png"))
#     plt.show()
