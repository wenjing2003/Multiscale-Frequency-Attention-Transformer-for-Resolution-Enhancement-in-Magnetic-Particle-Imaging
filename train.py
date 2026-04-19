import os
import sys
import matplotlib.pyplot as plt
import argparse
import datetime
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler

# Import custom modules
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, '../dataset/'))
sys.path.append(os.path.join(dir_name, '..'))

import utils
import options
from dataset.dataset_dehaze_denseHaze import get_training_data
from losses import CharbonnierLoss
from pytorch_msssim import SSIM
from model import MFAT

######### 1. Parameter Parsing ###########
opt = options.Options().init(argparse.ArgumentParser(description='MFAT 5-Fold Training')).parse_args()

######### 2. Environment & GPU Setup ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
torch.backends.cudnn.benchmark = True

######### 3. Log & Model Save Path Setup ###########
base_log_dir = os.path.join(opt.save_dir, 'reconstruction', opt.dataset, opt.arch + opt.env + '_5Fold')
utils.mkdir(base_log_dir)

######### 4. Set Random Seed ###########
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

######### 5. Data Preparation ###########
img_options_train = {'patch_size': 256}
full_dataset = get_training_data(opt.train_dir, img_options_train)
print(f"===> Full dataset loaded, total samples: {len(full_dataset)}")

######### 6. Define 5-Fold Cross Validation ###########
kf = KFold(n_splits=5, shuffle=True, random_state=42)
indices = np.arange(len(full_dataset))
all_folds_best_psnr = []

######### 7. 5-Fold Loop ###########
for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
    print(f"\n{'-'*20} Starting Fold {fold} {''-'*20}")
    
    # Create separate directories for each fold
    fold_dir = os.path.join(base_log_dir, f'fold_{fold}')
    model_dir = os.path.join(fold_dir, 'models')
    result_dir = os.path.join(fold_dir, 'results')
    utils.mkdir(model_dir)
    utils.mkdir(result_dir)

    # Construct Fold-specific DataLoaders
    train_loader = DataLoader(dataset=Subset(full_dataset, train_idx), batch_size=opt.batch_size, shuffle=True, num_workers=opt.train_workers)
    val_loader = DataLoader(dataset=Subset(full_dataset, val_idx), batch_size=opt.batch_size, shuffle=False, num_workers=opt.eval_workers)

    # --- C. Model Initialization with Updated Parameters ---
    # depths: [encoder_depths, bottleneck, decoder_depths]
    # heads: symmetric distribution across levels
    model_restoration = MFAT(
        img_size=256, 
        in_chans=3, 
        dd_in=3, 
        embed_dim=28,                # As specified
        depths=[2, 2, 2, 2, 2, 2, 2], # Encoder (2,2,2), Bottleneck (2), Decoder (2,2,2)
        num_heads=[1, 2, 4, 8, 4, 2, 1], 
        win_size=8,                  # As specified
        mlp_ratio=2.0,               # As specified
        drop_path_rate=0.1
    )
    model_restoration = torch.nn.DataParallel(model_restoration).cuda()

    # --- D. Optimizer, Scheduler, and Loss Functions ---
    optimizer = optim.Adam(model_restoration.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8)
    
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch - warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()

    criterion = CharbonnierLoss().cuda()
    ssim_loss = SSIM(data_range=1.0, size_average=True, channel=3).cuda()
    ssim_weight = 0.2
    loss_scaler = NativeScaler()

    # --- E. Training Loop ---
    best_psnr = 0
    train_losses = []
    eval_now = len(train_loader) // 4

    for epoch in range(1, opt.nepoch + 1):
        model_restoration.train()
        epoch_loss = 0
        
        for i, data in enumerate(tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch}"), 0):
            optimizer.zero_grad()
            target, input_ = data[0].cuda(), data[1].cuda()

            if epoch > 5:
                target, input_ = utils.MixUp_AUG().aug(target, input_)

            with torch.cuda.amp.autocast():
                restored = model_restoration(input_)
                loss = criterion(restored, target) + ssim_weight * (1 - ssim_loss(restored, target))

            loss_scaler(loss, optimizer, parameters=model_restoration.parameters())
            epoch_loss += loss.item()

            if (i + 1) % eval_now == 0:
                model_restoration.eval()
                psnr_val = []
                with torch.no_grad():
                    for data_val in val_loader:
                        target_val, input_val = data_val[0].cuda(), data_val[1].cuda()
                        restored_val = torch.clamp(model_restoration(input_val), 0, 1)
                        psnr_val.append(utils.batch_PSNR(restored_val, target_val, False).item())
                
                psnr_avg = sum(psnr_val) / len(psnr_val)
                if psnr_avg > best_psnr:
                    best_psnr = psnr_avg
                    torch.save({'state_dict': model_restoration.state_dict()}, os.path.join(model_dir, "model_best.pth"))
                model_restoration.train()

        scheduler.step()
        
        # Save loss curve
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        plt.figure()
        plt.plot(train_losses)
        plt.savefig(os.path.join(result_dir, 'loss_curve.png'))
        plt.close()

    print(f"===> Fold {fold} Finished. Best PSNR: {best_psnr:.4f}")
    all_folds_best_psnr.append(best_psnr)

    # Clean up memory
    del model_restoration, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()

######### 8. Final Statistics ###########
print("\n" + "="*30)
print(f"5-Fold Cross Validation Complete!")
print(f"Best PSNR per Fold: {all_folds_best_psnr}")
print(f"Mean PSNR: {np.mean(all_folds_best_psnr):.4f}")
print(f"Standard Deviation: {np.std(all_folds_best_psnr):.4f}")
print("="*30)
