import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import numpy as np

# --- 基础卷积模块 ---
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def forward(self, x):
        return self.block(x) + self.conv11(x)

# --- 注意力机制相关 ---
class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        return q[0], kv[0], kv[1]

class WindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.win_size = win_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))
        
        coords_h = torch.arange(self.win_size[0])
        coords_w = torch.arange(self.win_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.win_size[0] - 1
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)
            
        self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        rel_pos_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()
        attn = attn + rel_pos_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))

# --- 辅助模块 ---
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

class GSAU(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = n_feats * 2
        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats)
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()
        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.Conv2(x)
        return x * self.scale + shortcut

class MLKA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = 2 * n_feats
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        p = n_feats // 3
        self.splits = [p, p, n_feats - 2*p]

        self.LKA3 = nn.Sequential(nn.Conv2d(self.splits[0], self.splits[0], 3, 1, 1, groups=self.splits[0]),
                                  nn.Conv2d(self.splits[0], self.splits[0], 5, 1, 5, groups=self.splits[0], dilation=2),
                                  nn.Conv2d(self.splits[0], self.splits[0], 1, 1, 0))
        self.LKA5 = nn.Sequential(nn.Conv2d(self.splits[1], self.splits[1], 5, 1, 2, groups=self.splits[1]),
                                  nn.Conv2d(self.splits[1], self.splits[1], 7, 1, 9, groups=self.splits[1], dilation=3),
                                  nn.Conv2d(self.splits[1], self.splits[1], 1, 1, 0))
        self.LKA7 = nn.Sequential(nn.Conv2d(self.splits[2], self.splits[2], 7, 1, 3, groups=self.splits[2]),
                                  nn.Conv2d(self.splits[2], self.splits[2], 9, 1, 16, groups=self.splits[2], dilation=4),
                                  nn.Conv2d(self.splits[2], self.splits[2], 1, 1, 0))

        self.X3 = nn.Conv2d(self.splits[0], self.splits[0], 3, 1, 1, groups=self.splits[0])
        self.X5 = nn.Conv2d(self.splits[1], self.splits[1], 5, 1, 2, groups=self.splits[1])
        self.X7 = nn.Conv2d(self.splits[2], self.splits[2], 7, 1, 3, groups=self.splits[2])
        self.proj_first = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.proj_last = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_first(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        a1, a2, a3 = torch.split(a, self.splits, dim=1)
        a = torch.cat([self.LKA3(a1)*self.X3(a1), self.LKA5(a2)*self.X5(a2), self.LKA7(a3)*self.X7(a3)], dim=1)
        return self.proj_last(x * a) * self.scale + shortcut

class MAB(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.LKA = MLKA(n_feats)
        self.LFE = GSAU(n_feats)
    def forward(self, x):
        return self.LFE(self.LKA(x))

class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(DFFN, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.patch_size = 8
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0: x = F.pad(x, (0, pad_w, 0, pad_h))
        
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=self.patch_size, p2=self.patch_size)
        x_fft = torch.fft.rfft2(x_patch.float()) * self.fft
        x_patch = torch.fft.irfft2(x_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x[:, :, :h, :w] if (pad_h > 0 or pad_w > 0) else x

def window_partition(x, win_size):
    B, H, W, C = x.shape
    x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)

def window_reverse(windows, win_size, H, W):
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

# --- Transformer 模块 ---
class MFATBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, use_mab=True):
        super().__init__()
        self.dim, self.win_size, self.shift_size, self.use_mab = dim, win_size, shift_size, use_mab
        if min(input_resolution) <= self.win_size: self.shift_size = 0

        self.norm1 = norm_layer(dim)
        if use_mab:
            self.attn = MAB(dim)
        else:
            self.attn = WindowAttention(dim, to_2tuple(win_size), num_heads, qkv_bias, qk_scale, attn_drop, drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = DFFN(dim=dim, ffn_expansion_factor=mlp_ratio, bias=False)

    def forward(self, x, mask=None):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.use_mab:
            x_mab = x.permute(0, 3, 1, 2).contiguous()
            x = self.attn(x_mab).flatten(2).transpose(1, 2)
        else:
            shifted_x = x 
            x_win = window_partition(shifted_x, self.win_size).view(-1, self.win_size**2, C)
            attn_win = self.attn(x_win, mask=mask)
            x = window_reverse(attn_win.view(-1, self.win_size, self.win_size, C), self.win_size, H, W).flatten(1, 2)

        x = shortcut + self.drop_path(x)
        x_mlp = self.norm2(x).transpose(1, 2).reshape(B, C, H, W)
        return x + self.drop_path(self.mlp(x_mlp).flatten(2).transpose(1, 2))

class BasicMFATLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, win_size, mlp_ratio=2., drop_path=0., use_mab=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            MFATBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, win_size=win_size,
                      shift_size=0, mlp_ratio=mlp_ratio,
                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, use_mab=use_mab)
            for i in range(depth)])

    def forward(self, x, mask=None):
        for blk in self.blocks: x = blk(x, mask)
        return x

# --- 主模型 MFAT  ---
class MFAT(nn.Module):
    def __init__(self, img_size=256, in_chans=3, dd_in=3, 
                 embed_dim=28,                  
                 depths=[2, 2, 2, 2, 2, 2, 2],         
                 num_heads=[1, 2, 4, 8, 4, 2, 1],      
                 win_size=8,                    
                 mlp_ratio=2.,                  
                 drop_path_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.dd_in = dd_in
        
        
        dim0, dim1, dim2, dim_b = 28, 48, 96, 96 
        
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:4]))]
        
        self.input_proj = nn.Sequential(nn.Conv2d(dd_in, dim0, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.output_proj = nn.Conv2d(2*dim0, in_chans, 3, 1, 1)

        # Encoder 
        self.enc0 = BasicMFATLayer(dim0, (img_size, img_size), depths[0], num_heads[0], win_size, mlp_ratio, enc_dpr[:depths[0]], False)
        self.down0 = nn.Sequential(nn.Conv2d(dim0, dim1, 4, 2, 1))
        self.enc1 = BasicMFATLayer(dim1, (img_size//2, img_size//2), depths[1], num_heads[1], win_size, mlp_ratio, enc_dpr[depths[0]:sum(depths[:2])], False)
        self.down1 = nn.Sequential(nn.Conv2d(dim1, dim2, 4, 2, 1))
        self.enc2 = BasicMFATLayer(dim2, (img_size//4, img_size//4), depths[2], num_heads[2], win_size, mlp_ratio, enc_dpr[sum(depths[:2]):sum(depths[:3])], False)
        self.down2 = nn.Sequential(nn.Conv2d(dim2, dim_b, 4, 2, 1))
        
        # Bottleneck 
        self.bottleneck = BasicMFATLayer(dim_b, (img_size//8, img_size//8), depths[3], num_heads[3], win_size, mlp_ratio, drop_path_rate, True)

        # Decoder 
        self.up0 = nn.ConvTranspose2d(dim_b, dim2, 2, 2)
        self.dec0 = BasicMFATLayer(dim2*2, (img_size//4, img_size//4), depths[4], num_heads[4], win_size, mlp_ratio, 0., True)
        self.up1 = nn.ConvTranspose2d(dim2, dim1, 2, 2)
        self.dec1 = BasicMFATLayer(dim1*2, (img_size//2, img_size//2), depths[5], num_heads[5], win_size, mlp_ratio, 0., True)
        self.up2 = nn.ConvTranspose2d(dim1, dim0, 2, 2)
        self.dec2 = BasicMFATLayer(dim0*2, (img_size, img_size), depths[6], num_heads[6], win_size, mlp_ratio, 0., True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        y = self.input_proj(x).flatten(2).transpose(1, 2)
        
        # Encoder
        c0 = self.enc0(y)
        p0 = self.down0(c0.transpose(1, 2).reshape(x.shape[0], -1, 256, 256)).flatten(2).transpose(1, 2)
        c1 = self.enc1(p0)
        p1 = self.down1(c1.transpose(1, 2).reshape(x.shape[0], -1, 128, 128)).flatten(2).transpose(1, 2)
        c2 = self.enc2(p1)
        p2 = self.down2(c2.transpose(1, 2).reshape(x.shape[0], -1, 64, 64)).flatten(2).transpose(1, 2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder
        u0 = self.up0(b.transpose(1, 2).reshape(x.shape[0], -1, 32, 32)).flatten(2).transpose(1, 2)
        d0 = self.dec0(torch.cat([u0, c2], -1))
        u1 = self.up1(d0.transpose(1, 2).reshape(x.shape[0], -1, 64, 64)).flatten(2).transpose(1, 2)
        d1 = self.dec1(torch.cat([u1, c1], -1))
        u2 = self.up2(d1.transpose(1, 2).reshape(x.shape[0], -1, 128, 128)).flatten(2).transpose(1, 2)
        d2 = self.dec2(torch.cat([u2, c0], -1))

        # Output
        out = self.output_proj(d2.transpose(1, 2).reshape(x.shape[0], -1, 256, 256))
        return x + out if self.dd_in == 3 else out