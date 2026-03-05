import numpy as np
import os, sys, math
import argparse
from tqdm import tqdm
from einops import rearrange, repeat
import cv2

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, '../dataset/'))
sys.path.append(os.path.join(dir_name, '..'))

import scipy.io as sio
from dataset.dataset_dehaze_denseHaze import *
import utils

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss


from model import MFAT

parser = argparse.ArgumentParser(description='MPI Resolution Enhancement Evaluation on DenseHaze')
parser.add_argument('--input_dir', default='/MFAT/test', type=str, help='Directory of test images')
parser.add_argument('--result_dir', default='./results/reconstruction/MFAT/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./logs/reconstruction/DenseHaze/MFAT_final/models/model_best.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='2', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='MFAT', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--dd_in', type=int, default=3, help='dd_in')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)

test_dataset = get_validation_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)


model_restoration = MFAT(
    img_size=256, 
    in_chans=3, 
    dd_in=args.dd_in, 
    embed_dim=28,                  
    depths=[2, 2, 2, 2, 2, 2, 2],         
    num_heads=[1, 2, 4, 8, 4, 2, 1],      
    win_size=8,                   
    mlp_ratio=2.0,                 
    drop_path_rate=0.1
)

# 加载权重
utils.load_checkpoint(model_restoration, args.weights)
print("===> Testing using weights: ", args.weights)

model_restoration.cuda()
model_restoration.eval()

from utils.image_utils import splitimage, mergeimage

with torch.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_gt = data_test[0].numpy().squeeze().transpose((1,2,0))
        filenames = data_test[2]

        input_ = data_test[1].cuda()
        B, C, H, W = input_.shape
        
       
        if H <= 256 and W <= 256:
            restored = model_restoration(input_).cpu()
        else:
            
            corp_size_arg = 1152
            overlap_size_arg = 384
            split_data, starts = splitimage(input_, crop_size=corp_size_arg, overlap_size=overlap_size_arg)
            for i, data in enumerate(split_data):
                split_data[i] = model_restoration(data).cpu()
            restored = mergeimage(split_data, starts, crop_size=corp_size_arg, resolution=(B, C, H, W))

        rgb_restored = torch.clamp(restored, 0, 1).permute(0, 2, 3, 1).numpy()

        psnr = psnr_loss(rgb_restored[0], rgb_gt)
        ssim = ssim_loss(rgb_restored[0], rgb_gt, channel_axis=2, data_range=1)

        psnr_val_rgb.append(psnr)
        ssim_val_rgb.append(ssim)
        
        
        rgb_img = img_as_ubyte(rgb_restored[0])
        output_path = os.path.join(args.result_dir, filenames[0]+'.PNG')
        utils.save_grayscale_img(output_path, rgb_img)
        
        with open(os.path.join(args.result_dir,'psnr_ssim.txt'),'a') as f:
            f.write(filenames[0]+'.PNG ---->'+"PSNR: %.4f, SSIM: %.4f] "% (psnr, ssim)+'\n')

psnr_avg = sum(psnr_val_rgb)/len(test_dataset)
ssim_avg = sum(ssim_val_rgb)/len(test_dataset)
print("Final Average PSNR: %f, SSIM: %f " %(psnr_avg, ssim_avg))