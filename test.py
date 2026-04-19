import numpy as np
import os, sys, math
import argparse
from tqdm import tqdm
from einops import rearrange, repeat
import cv2

import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
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
from utils.image_utils import splitimage, mergeimage

parser = argparse.ArgumentParser(description='5-Fold Evaluation on DenseHaze')
# input_dir should be the same as the training data directory
parser.add_argument('--input_dir', default='/MFAT/train_dense', type=str, help='Directory of images')
# weights_dir should be the root folder containing fold_0, fold_1, etc.
parser.add_argument('--weights_dir', default='./logs/reconstruction/DenseHaze/MFAT_5Fold/', type=str, help='Root directory for 5-fold weights')
parser.add_argument('--gpus', default='2', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='MFAT', type=str, help='arch')
parser.add_argument('--dd_in', type=int, default=3, help='dd_in')
parser.add_argument('--save_images', action='store_true', help='Save reconstructed images')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

# 1. Load the full dataset (to recreate the splits)
full_dataset = get_validation_data(args.input_dir)
indices = np.arange(len(full_dataset))

# 2. Replicate the KFold logic used in training
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf_splits = list(kf.split(indices))

fold_results_psnr = []
fold_results_ssim = []

# 3. Iterate through each fold
for fold in range(5):
    print(f"\n{'='*20} Evaluating Fold {fold} {'='*20}")
    
    # Setup directories for this fold
    result_dir = os.path.join(args.weights_dir, f'fold_{fold}', 'test_results')
    utils.mkdir(result_dir)
    
    # Prepare the validation loader for this specific fold
    _, val_idx = kf_splits[fold]
    val_dataset = Subset(full_dataset, val_idx)
    test_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Initialize Model
    model_restoration = MFAT(
        img_size=256, in_chans=3, dd_in=args.dd_in, embed_dim=28,
        depths=[2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 4, 2, 1],
        win_size=8, mlp_ratio=2.0, drop_path_rate=0.1
    )
    
    # Load fold-specific weights
    weight_path = os.path.join(args.weights_dir, f'fold_{fold}', 'models', 'model_best.pth')
    if not os.path.exists(weight_path):
        print(f"Skipping: {weight_path} not found.")
        continue
        
    utils.load_checkpoint(model_restoration, weight_path)
    model_restoration.cuda()
    model_restoration.eval()

    psnr_val_rgb = []
    ssim_val_rgb = []

    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader, desc=f'Fold {fold} Inference')):
            rgb_gt = data_test[0].numpy().squeeze().transpose((1,2,0))
            filenames = data_test[2]
            input_ = data_test[1].cuda()
            B, C, H, W = input_.shape

            # Tiling strategy for high-resolution images
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

            # Calculate Metrics
            psnr = psnr_loss(rgb_restored[0], rgb_gt)
            ssim = ssim_loss(rgb_restored[0], rgb_gt, channel_axis=2, data_range=1)

            psnr_val_rgb.append(psnr)
            ssim_val_rgb.append(ssim)

            # Save Images if requested
            if args.save_images:
                rgb_img = img_as_ubyte(rgb_restored[0])
                output_path = os.path.join(result_dir, filenames[0]+'.PNG')
                utils.save_grayscale_img(output_path, rgb_img)

            # Log individual scores
            with open(os.path.join(result_dir, 'psnr_ssim.txt'), 'a') as f:
                f.write(f"{filenames[0]}.PNG ----> PSNR: {psnr:.4f}, SSIM: {ssim:.4f}\n")

    fold_avg_psnr = np.mean(psnr_val_rgb)
    fold_avg_ssim = np.mean(ssim_val_rgb)
    fold_results_psnr.append(fold_avg_psnr)
    fold_results_ssim.append(fold_avg_ssim)
    
    print(f"Fold {fold} Finished. Avg PSNR: {fold_avg_psnr:.4f}, SSIM: {fold_avg_ssim:.4f}")
    
    # Memory cleanup
    del model_restoration
    torch.cuda.empty_cache()

# 4. Final Cross-Validation Statistics
final_psnr = np.mean(fold_results_psnr)
final_ssim = np.mean(fold_results_ssim)
std_psnr = np.std(fold_results_psnr)

print("\n" + "#"*30)
print("FINAL CROSS-VALIDATION SUMMARY")
print(f"Mean PSNR: {final_psnr:.4f} ± {std_psnr:.4f}")
print(f"Mean SSIM: {final_ssim:.4f}")
print("#"*30)
