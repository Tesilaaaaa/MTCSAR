## 将所有的swin模块都改成anchor模块，调整head和深度

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.init as init
import torch.nn.functional as F
import math
from torch.autograd import Variable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.utils.checkpoint as checkpoint
import numpy as np

import thop

from ops import *


class AnchorProjection2(nn.Module):
    def __init__(self, dim, anchor_window_down_factor):
        super(AnchorProjection2, self).__init__()
        self.pooling = nn.AvgPool2d(anchor_window_down_factor, anchor_window_down_factor)
        self.linear = nn.Linear(dim, dim)
        self.df = anchor_window_down_factor
        self.dim = dim

    def forward(self, x, x_size):
        # x: B,T*H*W,C
        B_, N, C = x.shape
        T = N // (x_size[0] * x_size[1])
        x = blc_to_bchw(x, x_size)
        x = self.pooling(x)
        x = bchw_to_blc(x, T)
        x = self.linear(x)
        x = blc_to_bchw(x, [s // self.df for s in x_size]).view(-1, T, self.dim, x_size[0] // self.df,
                                                                x_size[1] // self.df)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        return x


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        self._initialize_weights()

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class ResidualBlock(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU+-
     |__________|
    '''

    def __init__(self, nf):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.norm = nn.BatchNorm2d(num_features=nf)

    def forward(self, x):
        out = F.relu(self.norm(self.conv1(x)), inplace=True)
        return x + out


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, T=5):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.T = T

    def forward(self, x, x_size):
        B, NHW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, self.T, x_size[0], x_size[1])  # B Ph*Pw C
        x = x.permute(0, 2, 1, 3, 4).view(B * self.T, self.embed_dim, x_size[0], x_size[1])
        return x

    def flops(self):
        flops = 0
        return flops


class Fea_extract(nn.Module):

    def __init__(self, inchannel=1, outchannel=64):
        super(Fea_extract, self).__init__()
        self.fea = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, 1, 1),
            nn.LeakyReLU()
        )
        layers1 = []
        for i in range(5):
            layers1.append(ResidualBlock_noBN(outchannel))

        self.fea_extraction = nn.Sequential(*layers1)
        # self.drop = nn.Dropout()

    def forward(self, x):
        x1 = self.fea(x)
        out = self.fea_extraction(x1)
        # out = self.drop(out)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, T=5):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.T = T
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        bt, C, H, W = x.size()
        b = bt // self.T
        x = x.view(b, self.T, C, H, W)
        x = x.permute(0, 1, 3, 4, 2).contiguous().view(b, self.T * H * W, C)
        # x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C  这个操作将张量x的第2个维度展平，然后将第1个维度和第2个维度交换
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        reduction (int): Channel reduction factor. Default: 16.
    """

    def __init__(self, num_feat, reduction=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // reduction, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // reduction, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=4, reduction=18):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, reduction),
        )

    def forward(self, x, x_size):
        B, L, C = x.shape
        H, W = x_size
        x_bchw = x.view(-1, H, W, C).permute(0, 3, 1, 2)
        x = self.cab(x_bchw.contiguous())
        x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, window_shift=True, qk_scale=None, attn_drop=0., proj_drop=0., T=5):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.T = T
        shift_size_h = window_size[0] // 2 if window_shift else 0
        shift_size_w = window_size[1] // 2 if window_shift else 0
        self.shift_size = to_2tuple((shift_size_h, shift_size_w))

        # define a parameter table of relative position bias 是表格，不是索引表，索引表是根据相对位置算出来的
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((T * 2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_t = torch.arange(T)
        coords = torch.stack(torch.meshgrid([coords_t, coords_h, # 生成网格坐标
                                             coords_w]))  # 3, T, Wh, Ww  关于meshgrid函数，查看 https://blog.csdn.net/wsj_jerry521/article/details/126678226
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww  横纵坐标相加 生成一维索引
        self.register_buffer("relative_position_index", relative_position_index) # 将索引注册为不需要学习的张量

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, x_size, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        H, W = x_size
        B, L, C = qkv.shape
        window_size = self.window_size[0]
        qkv = qkv.view(B, self.T, H, W, C)

        # cyclic shift
        if self.shift_size[0] > 0:
            shifted_qkv = torch.roll(
                qkv, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3)
            )
        else:
            shifted_qkv = qkv

        shifted_qkv = window_partition(shifted_qkv, self.window_size)
        shifted_qkv = shifted_qkv.view(-1, self.T * window_size * window_size, C)

        B_, N, _ = shifted_qkv.shape

        shifted_qkv = shifted_qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = shifted_qkv[0], shifted_qkv[1], shifted_qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.T * self.window_size[0] * self.window_size[1], self.T * self.window_size[0] * self.window_size[1],
            -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C // 3)
        x = self.proj(x)
        x = self.proj_drop(x)

        # merge windows
        x = x.view(-1, self.T, window_size, window_size, C // 3)
        shifted_x = window_reverse(x, self.window_size, H, W, self.T)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size[0] > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(2, 3))
        else:
            x = shifted_x
        x = x.view(B, self.T * H * W, C // 3)

        return x


class AnchorStripeAttention(nn.Module):
    def __init__(self, input_resolution, stripe_size, stripe_shift, num_heads, attn_drop, anchor_window_down_factor=1,
                 T=5, stripe_type="H"):
        super(AnchorStripeAttention, self).__init__()
        self.input_resolution = input_resolution
        self.stripe_size = stripe_size  # Wh, Ww
        self.stripe_shift = stripe_shift
        self.num_heads = num_heads
        self.anchor_window_down_factor = anchor_window_down_factor
        self.T = T
        self.stripe_type = stripe_type

        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.cbp_mlp_table = nn.Sequential(
            nn.Linear(3, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4, bias=False)
        )
        # self.table_index = self.set_table_index(self.input_resolution)

    def set_table_index(self, device=None):
        ss = self.stripe_size
        sss = []
        for i in ss:
            sss.append(i // 2 if self.stripe_shift else 0)
        df = self.anchor_window_down_factor

        if self.stripe_type == "H":
            table_s = get_relative_coords_table_all(ss, df)
            index_s_a2w = get_relative_position_index_simple(ss, df, False)
            index_s_w2a = get_relative_position_index_simple(ss, df, True)
        else:
            table_s = get_relative_coords_table_all(  # 垂直方向上的条带
                ss[::-1], df
            )
            index_s_a2w = get_relative_position_index_simple(ss[::-1], df, False)
            index_s_w2a = get_relative_position_index_simple(ss[::-1], df, True)
        return {
            "table_s": table_s.to(device),
            "index_s_a2w": index_s_a2w.to(device),
            "index_s_w2a": index_s_w2a.to(device),
        }

    def anchorAttn(self, q, k, v, table, index, mask, reshape=True):
        # q, k, v: # nW*B, H, wh*ww, dim
        # cosine attention map
        B_, _, H, head_dim = q.shape
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        B_attn, H_attn, N1, N2 = attn.shape

        table = self.cbp_mlp_table(table)
        bias_table = table.view(-1, H_attn)

        bias = bias_table[index.view(-1)]
        bias = bias.view(N1, N2, -1).permute(2, 0, 1).contiguous()

        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(B_ // nW, nW, H_attn, N1, N2) + mask
            attn = attn.view(-1, H_attn, N1, N2)

        # attention
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v  # B_, H, N1, head_dim
        if reshape:
            x = x.transpose(1, 2).reshape(B_, -1, H * head_dim)
        # B_, N, C
        return x

    def forward(self, qkv, anchor, x_size, mask):
        H, W = x_size
        B, L_, C = qkv.shape
        C_anchor = anchor.shape[4]
        stripe_size = self.stripe_size
        device = qkv.device
        shift_size = []
        if mask["mask_s_a2w"] != None:
            mask_a2w = mask["mask_s_a2w"].to(device)
            mask_w2a = mask["mask_s_w2a"].to(device)
        else:
            mask_a2w = mask["mask_s_a2w"]
            mask_w2a = mask["mask_s_w2a"]

        table_index = self.set_table_index(device)
        for s in stripe_size:
            shift_size.append(s // 2 if self.stripe_shift else 0)
        anchor_stripe_size = [s // self.anchor_window_down_factor for s in stripe_size]
        anchor_shift_size = [s // self.anchor_window_down_factor for s in shift_size]
        qkv = qkv.view(B, self.T, H, W, C)

        if self.stripe_shift:
            qkv = torch.roll(qkv, shifts=(-shift_size[0], -shift_size[1]), dims=(2, 3))
            anchor = torch.roll(
                anchor,
                shifts=(-anchor_shift_size[0], -anchor_shift_size[1]),
                dims=(2, 3),
            )

        qkv = window_partition(qkv, stripe_size)
        qkv = qkv.view(-1, self.T * self.stripe_size[0] * self.stripe_size[1], C)
        anchor = window_partition(anchor, anchor_stripe_size)
        anchor = anchor.view(-1, self.T * anchor_stripe_size[0] * anchor_stripe_size[1], C_anchor)

        B_, N1, _ = qkv.shape
        N2 = anchor.shape[1]
        qkv = qkv.reshape(B_, N1, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        anchor = anchor.reshape(B_, N2, self.num_heads, -1).permute(0, 2, 1, 3)

        # attention
        x = self.anchorAttn(
            anchor, k, v, table_index["table_s"], table_index["index_s_a2w"],
            mask_a2w, reshape=False
        )
        x = self.anchorAttn(q, anchor, x, table_index["table_s"],
                            table_index["index_s_w2a"], mask_w2a)

        # merge windows
        x = x.view(B_, -1, C // 3)
        x = window_reverse(x, stripe_size, H, W, self.T)  # B H' W' C

        # reverse the shift
        if self.stripe_shift:
            x = torch.roll(x, shifts=shift_size, dims=(2, 3))

        x = x.view(B, self.T * H * W, C // 3)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (tuple): Window size.
        shift_size (tuple): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        stripe_size(tuple)
    """

    def __init__(self, dim, input_resolution, num_heads, T, window_size=(8, 8), window_shift=True,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, anchor_window_down_factor=2,
                 stripe_size=(8, 8), stripe_shift=False, stripe_type="H"):
        super().__init__()
        self.T = T
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.window_shift = window_shift
        self.mlp_ratio = mlp_ratio
        self.stripe_size = stripe_size
        self.stripe_shift = stripe_shift
        self.stripe_type = stripe_type
        self.anchor_window_down_factor = anchor_window_down_factor
        # if min(self.input_resolution) <= min(self.window_size):
        #     # if window size is larger than input resolution, we don't partition windows
        #     self.shift_size = (0, 0)
        #     self.window_size = self.input_resolution
        # assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size_h must in 0-window_size"
        # assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size_w must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.winAttn = WindowAttention(
            dim // 2, window_size=to_2tuple(self.window_size), num_heads=num_heads, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, T=self.T,
            window_shift=self.window_shift)
        self.ancAttn = AnchorStripeAttention(self.input_resolution, self.stripe_size, self.stripe_shift, self.num_heads,
                                             attn_drop=attn_drop,
                                             anchor_window_down_factor=anchor_window_down_factor, T=T,
                                             stripe_type=stripe_type)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()  # 随机深度率,DropPath是一种在深度神经网络中使用的正则化方法，它可以随机地将一些神经元的输出设置为0。这种方法可以有效地防止过拟合，提高模型的泛化能力
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)  # 多层感知机

        self.cab = CAB(num_feat=dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.anchor = AnchorProjection2(self.dim, anchor_window_down_factor)

    def calculate_win_mask(self, x_size, N):
        # calculate attention mask for SW-MSA
        if self.window_shift == False:
            attn_mask = None
        else:
            H, W = x_size
            shift_size = []
            for i in self.window_size:
                shift_size.append(i // 2 if self.window_shift else 0)
            img_mask = torch.zeros((1, N, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -shift_size[0]),  # 因为shift_size不同导致的mask不同
                        slice(-shift_size[0], None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -shift_size[1]),
                        slice(-shift_size[1], None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, :, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, N * self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            attn_mask = attn_mask.cuda()

        return attn_mask

    def calculate_anchor_mask(self, x_size, N):
        if self.stripe_shift == False:
            mask_s_a2w = None
            mask_s_w2a = None
        else:
            ss = self.stripe_size
            sss = []
            for i in ss:
                sss.append(i // 2 if self.stripe_shift else 0)
            df = self.anchor_window_down_factor

            if self.stripe_type == "H":
                mask_s_a2w = calculate_mask_all(x_size, ss, sss, df, False, N)
                mask_s_w2a = calculate_mask_all(x_size, ss, sss, df, True, N)
            else:
                mask_s_a2w = calculate_mask_all(x_size, ss[::-1], sss[::-1], df, False, N)
                mask_s_w2a = calculate_mask_all(x_size, ss[::-1], sss[::-1], df, True, N)
            mask_s_a2w = mask_s_a2w.cuda()
            mask_s_w2a = mask_s_w2a.cuda()

        return {
            "mask_s_a2w": mask_s_a2w,
            "mask_s_w2a": mask_s_w2a,
        }

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        # Channel Attention
        x_cab = self.cab(x, x_size)

        shortcut = x
        x = self.norm1(x)

        qkv = self.qkv(x)
        # qkv_window, qkv_stripe = torch.split(qkv, C * 3 // 2, dim=-1)
        qkv_stripe = qkv
        anchor = self.anchor(x, x_size)
        # x = x.view(B, self.T, H, W, C)

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        # x_windows = self.winAttn(qkv_window, x_size=x_size, mask=self.calculate_win_mask(x_size, self.T))
        x_anchor = self.ancAttn(qkv_stripe, anchor, x_size,
                                mask=self.calculate_anchor_mask(x_size, self.T))

        # x = torch.cat([x_windows, x_anchor], dim=-1)
        x = x_anchor

        # FFN
        x = shortcut + self.drop_path(x) + x_cab
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (tuple): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, window_shift, T,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 stripe_size=(8, 8), stripe_shift=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.T = T

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 window_shift=i % 2 == 0 if window_shift else False,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer, T=self.T,
                                 stripe_size=stripe_size,
                                 stripe_type="H" if i % 2 == 0 else "W",
                                 stripe_shift=i % 4 in [2, 3] if stripe_shift else False,
                                 )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (tuple): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, window_shift, T=5,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv', stripe_size=(8, 8), stripe_shift=False):
        super(RSTB, self).__init__()

        self.dim = dim
        self.T = T
        self.input_resolution = input_resolution
        self.window_shift = window_shift

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint,
                                         T=self.T,
                                         stripe_size=stripe_size,
                                         stripe_shift=stripe_shift,
                                         window_shift=window_shift)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x


class Swin(nn.Module):
    def __init__(self, img_size=80, patch_size=1, in_chans=1, T=5,
                 embed_dim=128, depths=[2, 2], num_heads=[2, 2],
                 window_size=(8, 8), window_shift=True, mlp_ratio=2, qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
                 stripe_size=(8, 8), stripe_shift=True,
                 **kwargs):
        super(Swin, self).__init__()
        self.window_size = window_size
        self.T = T

        self.img_range = img_range
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.window_shift = window_shift

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.stripe_size = stripe_size
        self.stripe_shift = stripe_shift

        self.conv_first = nn.Conv2d(self.embed_dim // 2, self.embed_dim, 3, 1, 1, bias=True)

        self.conv_last_swin = nn.Conv2d(self.embed_dim, self.embed_dim // 2, 3, 1, 1, bias=True)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection,
                         T=self.T,
                         window_shift=window_shift,
                         stripe_shift=stripe_shift
                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        self.conv_after_body = nn.Conv2d(embed_dim, int(embed_dim), 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size[0] - h % self.window_size[0]) % self.window_size[0]
        mod_pad_w = (self.window_size[1] - w % self.window_size[1]) % self.window_size[1]
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        BN = x.shape[0]
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]
        BN = x.shape[0]
        x = self.check_image_size(x)

        # for image denoising and JPEG compression artifact reduction
        x_first = self.conv_first(x)
        res = self.conv_after_body(self.forward_features(x_first)) + x_first
        res = self.conv_last_swin(res)
        res = res.view(BN // self.T, self.T, -1, H, W)

        return res


class TFusion(nn.Module):
    def __init__(self, center=4, in_chan=90, out_chan=180):
        super(TFusion, self).__init__()

        self.center = center
        self.embchan = in_chan
        self.conv1 = nn.Conv2d(in_chan, in_chan, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(in_chan, in_chan, 3, 1, 1, bias=True)

        self.conv3 = nn.Conv2d(in_chan, in_chan, 3, 1, 1, bias=True)

    def forward(self, x):
        B, N, C, H, W = x.size()
        ref = self.conv1(x[:, self.center, :, :, :].clone())

        cor = []
        for i in range(N):
            nbr = self.conv2(x[:, i, :, :, :])
            tmp = torch.mean(ref * nbr, 1).unsqueeze(1)
            cor.append(tmp)
        prob = torch.sigmoid(torch.cat(cor, dim=1))
        prob = prob.unsqueeze(2).repeat(1, 1, C, 1, 1)
        x = torch.mean(x * prob, dim=1)

        x = self.lrelu(self.conv3(x))

        return x


class cc_fusion(nn.Module):
    def __init__(self, dim=90, T=5):
        super(cc_fusion, self).__init__()
        self.dim = dim
        self.T = T
        self.feature_extraction = Fea_extract(inchannel=1, outchannel=dim)
        # self.multi_to_one_conv = nn.Conv2d(self.T * dim, dim, 3, 1, 1)
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.lrelu = nn.LeakyReLU()

    def forward(self, cc, x):
        B, N, C, H, W = cc.shape
        T = x.shape[1]

        x_cc = []
        for i in range(T - 1):
            cc_tmp = self.feature_extraction(cc[:, i, :, :, :])
            cc_tmp = self.conv1(cc_tmp)
            tmp = x[:, i, :, :, :].clone() * torch.sigmoid(cc_tmp)
            x_cc.append(tmp.unsqueeze(1))
        x_cc.append(x[:, T - 1, :, :, :].unsqueeze(1))
        x_cc = torch.cat(x_cc, dim=1)
        x_cc = torch.mean(x_cc, dim=1)

        x_cc = self.lrelu(self.conv(x_cc))
        # x = x * cc
        # x = x.view(B, T, -1, H, W).contiguous().view(B, T * self.dim, H, W)
        # x = self.lrelu(self.multi_to_one_conv(x))

        return x_cc


class MSARCC(nn.Module):
    def __init__(self, center=4, embedding_chan=64, T=5):
        super(MSARCC, self).__init__()
        self.center = center
        self.T = T

        self.conv_first = nn.Conv2d(1, embedding_chan, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.fea = ResidualBlock_noBN(nf=embedding_chan)

        self.swin = Swin(center=self.center, in_chans=embedding_chan, embed_dim=embedding_chan * 2, T=self.T)
        self.tattn = TFusion(center=center, in_chan=embedding_chan)
        self.cc_Fusion = cc_fusion(dim=embedding_chan)

        self.conv_ta = nn.Conv2d(embedding_chan, embedding_chan, 1, 1, bias=True)
        self.conv_cc = nn.Conv2d(embedding_chan, embedding_chan, 1, 1, bias=True)

        self.conv_after_fusion1 = nn.Conv2d(embedding_chan, embedding_chan, 3, 1, 1, bias=True)
        self.conv_after_fusion2 = nn.Conv2d(embedding_chan, embedding_chan, 3, 1, 1, bias=True)
        self.conv_after_fusion3 = nn.Conv2d(embedding_chan, embedding_chan, 3, 1, 1, bias=True)
        layers1 = []
        for i in range(3):
            layers1.append(ResidualBlock_noBN(nf=embedding_chan))
        self.recon = nn.Sequential(*layers1)

        self.conv_last = nn.Conv2d(embedding_chan, 1, 3, 1, 1, bias=True)

    def forward(self, x, CC):
        B, N, C, H, W = x.size()
        x_first = []
        for i in range(N):
            x_nbr = x[:, i, :, :, :].contiguous()
            x_nbr = self.lrelu(self.conv_first(x_nbr))
            fea_tmp = self.fea(x_nbr).unsqueeze(1)
            x_first.append(fea_tmp)
        x_first = torch.cat(x_first, dim=1).view(B * N, -1, H, W)

        x_swin = self.swin(x_first)  # 1,5,90,80,80

        x_fusion = self.relu(self.conv_ta(self.tattn(x_swin.view(B, N, -1, H, W)))) + self.relu(
            self.conv_cc(self.cc_Fusion(CC, x_swin)))
        x_fusion = self.relu(self.conv_after_fusion1(x_fusion))
        x_fusion = self.relu(self.conv_after_fusion2(x_fusion + x_first.view(B, N, -1, H, W)[:, self.center, :, :, :]))
        x_fusion = self.lrelu(self.conv_after_fusion3(self.recon(x_fusion) + x_fusion))
        res = self.conv_last(x_fusion) + x[:, self.center, :, :, :]

        return res


if __name__ == "__main__":
    window_size = 8

    model = MSARCC(center=4)

    # print(model)
    # print(height, width, model.flops() / 1e9)

    x = torch.randn((1, 5, 1, 40, 40))
    cc = torch.randn((1, 4, 1, 40, 40))
    y=model(x,cc)
    flops, params = thop.profile(model, inputs=(x,cc))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.4f G, params: %.4f M' % (flops / 1000000000.0, params / 1000000.0))
    # print(x.shape)
    num_params = 0
    for p in model.parameters():
        if p.requires_grad:
            num_params += p.numel()
    print(f"Number of parameters {num_params / 10 ** 6: 0.2f}")
# -*-coding:utf-8-*-
