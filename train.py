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
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler

# 自定义模块导入
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, '../dataset/'))
sys.path.append(os.path.join(dir_name, '..'))

import utils
import options
from dataset.dataset_dehaze_denseHaze import get_training_data
from losses import CharbonnierLoss
from pytorch_msssim import SSIM


from model import MFAT

######### 1. 解析参数 ###########
opt = options.Options().init(argparse.ArgumentParser(description='MPI Resolution Enhancement Training')).parse_args()

######### 2. 设置环境与 GPU ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
torch.backends.cudnn.benchmark = True

######### 3. 设置日志与模型保存路径 ###########
log_dir = os.path.join(opt.save_dir, 'reconstruction', opt.dataset, opt.arch + opt.env)
utils.mkdir(log_dir)
result_dir = os.path.join(log_dir, 'results')
model_dir = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(model_dir)

logname = os.path.join(log_dir, datetime.datetime.now().isoformat().replace(':', '-') + '.txt')

######### 4. 设置随机种子 (固定为 42 以保证可复现性 ) ###########
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

######### 5. 模型初始化 ###########
model_restoration = MFAT(
    img_size=256,              
    in_chans=3, 
    dd_in=3, 
    embed_dim=28,               
    depths=[2, 2, 2, 2, 2, 2, 2], 
    num_heads=[1, 2, 4, 8, 4, 2, 1], 
    win_size=8,                
    mlp_ratio=2.0,              
    drop_path_rate=0.1         
)

# 打印模型信息并写入日志
with open(logname, 'a') as f:
    f.write(str(opt) + '\n')
    f.write(str(model_restoration) + '\n')

######### 6. 优化器与调度器 ###########

optimizer = optim.Adam(model_restoration.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8)


warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch - warmup_epochs, eta_min=1e-6)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

######### 7. 数据并行与 GPU 加载 ###########
model_restoration = torch.nn.DataParallel(model_restoration)
model_restoration.cuda()

######### 8. 损失函数 (Charbonnier Loss ) ###########
criterion = CharbonnierLoss().cuda()
ssim_loss = SSIM(data_range=1.0, size_average=True, channel=3).cuda()
ssim_weight = 0.2 # 混合 SSIM 损失增强结构保真度

######### 9. 数据加载与划分 ###########
img_options_train = {'patch_size': 256}
full_train_dataset = get_training_data(opt.train_dir, img_options_train)

# 随机划分 10% 作为验证集 [cite: 4]
val_len = int(len(full_train_dataset) * 0.1)
train_len = len(full_train_dataset) - val_len
train_dataset, val_dataset = random_split(full_train_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.train_workers)
val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.eval_workers)

######### 10. 训练循环 ###########
best_psnr = 0
train_losses = []
eval_now = len(train_loader) // 4
loss_scaler = NativeScaler()

print(f"===> 开始训练: 总 Epochs {opt.nepoch}")

for epoch in range(1, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    model_restoration.train()

    for i, data in enumerate(tqdm(train_loader), 0):
        optimizer.zero_grad()
        target = data[0].cuda()
        input_ = data[1].cuda()

        
        if epoch > 5:
            target, input_ = utils.MixUp_AUG().aug(target, input_)

        with torch.cuda.amp.autocast():
            restored = model_restoration(input_)
            loss = criterion(restored, target) + ssim_weight * (1 - ssim_loss(restored, target))

        loss_scaler(loss, optimizer, parameters=model_restoration.parameters())
        epoch_loss += loss.item()

        # 阶段性验证
        if (i + 1) % eval_now == 0:
            model_restoration.eval()
            psnr_val = []
            for ii, data_val in enumerate(val_loader, 0):
                target_val = data_val[0].cuda()
                input_val = data_val[1].cuda()
                with torch.no_grad():
                    restored_val = torch.clamp(model_restoration(input_val), 0, 1)
                psnr_val.append(utils.batch_PSNR(restored_val, target_val, False).item())
            
            psnr_avg = sum(psnr_val) / len(psnr_val)
            if psnr_avg > best_psnr:
                best_psnr = psnr_avg
                torch.save({'state_dict': model_restoration.state_dict()}, os.path.join(model_dir, "model_best.pth"))
            
            print(f"[Epoch {epoch} Iter {i}] PSNR: {psnr_avg:.4f} | Best: {best_psnr:.4f}")
            model_restoration.train()

    scheduler.step()
    
    # 记录并保存损失曲线
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    plt.figure()
    plt.plot(train_losses)
    plt.savefig(os.path.join(result_dir, 'loss_curve.png'))
    plt.close()

    # 每天 Epoch 保存最新模型
    torch.save({'epoch': epoch, 'state_dict': model_restoration.state_dict()}, os.path.join(model_dir, "model_latest.pth"))

print(f"训练完成！最佳验证 PSNR: {best_psnr:.4f}")