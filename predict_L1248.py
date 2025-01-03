import argparse
import os, time, datetime
import numpy as np
import torch.nn as nn
import torch
import os
import h5py
import model1 as model

import utils
from scipy.io import savemat
from tqdm import tqdm
import time

cuda = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='G:/SARData/JingChengSLC-temporalMultilook/train-val-test/test/', type=str,
                        help='directory of test dataset')
    parser.add_argument('--set_dir_simu', default='Noisy400L1248', type=str, help='directory of test dataset-noisy')
    parser.add_argument('--set_dir_ref', default='Ref400', type=str, help='directory of test dataset-true')
    parser.add_argument('--set_dir_cc', default='CC400', type=str, help='directory of test dataset-CC')
    parser.add_argument('--model_dir', default='./model_tiny_trick1/', help='directory of the model')
    parser.add_argument('--model_name', default='CC_CNN_itr100.ckpt', type=str, help='the model name')
    # parser.add_argument('--result_dir', default='./res_L1248_single/', type=str, help='directory of test dataset')
    parser.add_argument('--result_dir', default='./res_L1248_base_trick1_1/', type=str, help='directory of test dataset')
    parser.add_argument('--save_result', default=0, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def save_result(result, path):
    path = path if path.find('.') != -1 else path + '.tif'
    ext = os.path.splitext(path)[-1]
    np.savetxt(path, result, fmt='%2.4f')


if __name__ == '__main__':

    args = parse_args()
    #
    # net = model.MSARCC(center=0, T=1)
    net = model.MSARCC(center=4, T=5)
    load_stict = torch.load(os.path.join(args.model_dir, args.model_name))
    net = nn.DataParallel(net).cuda()
    net.load_state_dict(load_stict)
    net = net.cuda()
    net.eval()

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    img_list=os.listdir(os.path.join(args.set_dir, args.set_dir_simu))
    img_list= [item for item in img_list if '.mat' in item]

    start_time = time.perf_counter()
    for im in tqdm(img_list):
        imName = im.split('-')
        imName = imName[1] + '-' + imName[2]

        y = h5py.File(os.path.join(args.set_dir, args.set_dir_simu, im), 'r')
        y = np.array(y['Noisy400Multi']).astype(np.float32)

        y = np.transpose(y)
        # y = y[0, :, :, :][np.newaxis, :, :, :]
        mx = np.max(y)
        y = y / mx

        imRef = 'Ref400List-' + imName + '.mat'
        x = h5py.File(os.path.join(args.set_dir, args.set_dir_ref, imRef), 'r')
        x = np.array(x['Ref400multi']).astype(np.float32)
        mx2=np.max(x)

        imCC = 'CC400-' + imName + '.mat'
        cc = h5py.File(os.path.join(args.set_dir, args.set_dir_cc, imCC), 'r')
        cc = np.array(cc['CC400Multi']).astype(np.float32)
        # cc=cc[:,:,0:200,0:200]
        cc = np.transpose(cc)

        # cc = y

        N, C, H, W = y.shape
        # max_=np.max(x[4,0,:,:])

        y_ = torch.from_numpy(y).contiguous().view(-1, N, 1, H, W).float()
        cc_ = torch.from_numpy(cc).contiguous().view(-1, N - 1, 1, H, W)
        # cc_ = torch.from_numpy(cc).contiguous().view(-1, 1, 1, H, W)
        torch.cuda.synchronize()
        y_ = y_.cuda()
        cc_ = cc_.cuda()
        with torch.no_grad():
            x_ = net(y_, cc_)
        x_ = x_.cpu()
        x_ = x_.view(H, W)
        # psnr = utils.psnr(x_, x[4, 0])

        x_ = x_.detach().numpy().astype(np.float32)
        x_ = x_ * mx2
        # x_ = np.exp(x_) - 1

        torch.cuda.synchronize()

        # print(im + ' ' + str(psnr.item()))

        savemat(os.path.join(args.result_dir, 'depseckling' + im[13:len(im) - 4] + '-msar_mscc.mat'),
                {'despeckling': x_})
    end_time = time.perf_counter()
    time=(end_time-start_time)/280
    print(time)
