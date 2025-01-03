# -*-coding:utf-8-*-
# 训练不同版本的网络，tinybaselarge
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import argparse
import numpy as np
import torch
import glob
# import h5py
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

import utils
import model_grl2_trick1_tiny as model
from Regularization import Regularization


def train(data_loader, net, Loss, RegLoss, optimizer, cuda):
    net.train()
    epoch_loss = []

    for count, train_batch in enumerate(data_loader):
        if cuda:
            train_input = train_batch[0].cuda()
            target = train_batch[1].cuda()
            CC = train_batch[2].cuda()
        else:
            train_input = train_batch[0]
            target = train_batch[1]
            CC = train_batch[2]
        optimizer.zero_grad()
        train_denoised = net(train_input, CC)
        loss = Loss(train_denoised, target[:, args.center, :, :, :])
        reg_loss = RegLoss(net)
        loss = loss + reg_loss
        total_loss = loss.cpu().detach().numpy().item()
        epoch_loss.append(total_loss)
        loss.backward()
        optimizer.step()
        # if count % 20 == 0:
        #     print(np.mean(epoch_loss))

    return np.mean(epoch_loss)


def val(val_loader, model, Loss, RegLoss, cuda):
    model.eval()
    with torch.no_grad():
        epoch_psnr = []
        epoch_loss = []

        for count, train_batch in enumerate(val_loader):
            if cuda:
                val_input = train_batch[0].cuda()
                target = train_batch[1].cuda()
                CC = train_batch[2].cuda()
            else:
                val_input = train_batch[0]
                target = train_batch[1]
                CC = train_batch[2]

            val_denoised = model(val_input, CC)
            loss = Loss(val_denoised, target[:, args.center, :, :, :])
            reg_loss = RegLoss(net)
            total_loss = loss + reg_loss
            total_loss = np.array(total_loss.item())

            # val_denoised = torch.exp(val_denoised) - 1
            # target = torch.exp(target) - 1

            psnr = utils.psnr(val_denoised, target[:, args.center, :, :, :])
            epoch_psnr.append(psnr.item())
            epoch_loss.append(total_loss)

    return np.mean(epoch_loss), np.mean(epoch_psnr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="it's usage tip.", description="help info.")
    # parser.add_argument("--gpu", choices=['0', '1', '2', '3'], default='0', help="gpu_id")
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--epoch", default=100)
    parser.add_argument("--epsilon", default=1e-4)
    parser.add_argument("--weight_decay", default=1e-8)
    parser.add_argument("--center", default=4)
    parser.add_argument("--image_size", default=60)
    # parser.add_argument("--NormPara", default=0.623210802613665)
    args = parser.parse_args()

    model_path = './model_tiny_trick1'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    logs_path = './logs_tiny_trick1'
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)

    train_noisyimg_path = 'G:/SARData/JingChengSLC-temporalMultilook/train-val-test/train/Noisy400/'
    train_truthimg_path = 'G:/SARData/JingChengSLC-temporalMultilook/train-val-test/train/Ref400/'
    train_CC_path = 'G:/SARData/JingChengSLC-temporalMultilook/train-val-test/train/CC400/'
    val_noisyimg_path = 'G:/SARData/JingChengSLC-temporalMultilook/train-val-test/val/Noisy400/'
    val_truthimg_path = 'G:/SARData/JingChengSLC-temporalMultilook/train-val-test/val/Ref400/'
    val_CC_path = 'G:/SARData/JingChengSLC-temporalMultilook/train-val-test/val/CC400/'

    [train_target_batch, train_noisy_batch, train_CC_batch] = utils.dataGenerator(train_truthimg_path,
                                                                                  train_noisyimg_path, train_CC_path)
    print('train data OK')
    [val_target_batch, val_noisy_batch, val_CC_batch] = utils.dataGenerator(val_truthimg_path, val_noisyimg_path,
                                                                            val_CC_path)
    print('val data OK')

    cuda = torch.cuda.is_available()

    net = model.MSARCC(center=args.center)
    net.apply(utils.weight_init)
    if cuda:
        net = net.cuda()
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net, device_ids=[0, 1])
    Loss = utils.Loss()
    reg_Loss = Regularization(net, args.weight_decay)
    train_dataset = utils.get_dataset(train_noisy_batch, train_target_batch, train_CC_batch)
    val_dataset = utils.get_dataset(val_noisy_batch, val_target_batch, val_CC_batch)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.Adam(net.parameters(), lr=args.epsilon)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50,90], gamma=0.1, last_epoch=-1)

    if cuda:
        Loss = Loss.cuda()
        reg_Loss = reg_Loss.cuda()

    train_loss = []
    val_loss = []
    val_psnr = []

    for i in range(0, args.epoch):
        train_loss0 = train(train_loader, net, Loss, reg_Loss, optimizer, cuda)
        train_loss.append(train_loss0)

        [val_loss0, val_psnr0] = val(val_loader, net, Loss, reg_Loss, cuda)
        val_loss.append(val_loss0)
        val_psnr.append(val_psnr0)

        scheduler.step()
        if (i + 1) % 10 == 0:
            torch.save(net.state_dict(), os.path.join(model_path, 'CC_CNN_itr%d.ckpt' % (i + 1)))
        print('epoch ' + str(i + 1) + ' / 100 mean train loss:' + str(train_loss0))
        print('mean val psnr:' + str(val_psnr0))

        train_loss_save = np.array(train_loss)
        val_loss_save = np.array(val_loss)
        val_psnr_save = np.array(val_psnr)
        np.save(os.path.join(logs_path, 'train_loss.npy'), train_loss_save)
        np.save(os.path.join(logs_path, 'val_loss.npy'), val_loss_save)
        np.save(os.path.join(logs_path, 'val_psnr.npy'), val_psnr_save)

    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    val_psnr = np.array(val_psnr)
