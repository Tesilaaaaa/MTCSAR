# -*-coding:utf-8-*-
from math import prod
from typing import Tuple
import torch.nn as nn
import numpy as np
import torch
from timm.models.layers import to_2tuple


class AnchorProjection(nn.Module):
    def __init__(self, dim, anchor_window_down_factor):
        super(AnchorProjection, self).__init__()
        self.pooling = nn.AvgPool2d(anchor_window_down_factor, anchor_window_down_factor)
        self.linear = nn.Linear(dim, dim // 2)
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
        x = blc_to_bchw(x, [s // self.df for s in x_size]).view(-1, T, self.dim // 2, x_size[0] // self.df,
                                                                x_size[1] // self.df)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        return x


def window_partition(x, window_size):  # 1,48,48,1 -> 36,8,8,1
    """
    Args:
        x: (B, H, W, C)
        window_size (tuple): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, N, H, W, C = x.shape
    x = x.view(B, N, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(-1, N, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W, N):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (tuple): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], N, window_size[0], window_size[1], -1)
    x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(B, N, H, W, -1)
    return x


def blc_to_bchw(x: torch.Tensor, x_size: Tuple) -> torch.Tensor:
    B, L, C = x.shape
    [H, W] = x_size
    return x.view(-1, H, W, C).permute(0, 3, 1, 2)


def bchw_to_blc(x: torch.Tensor, T) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, C, H, W) to (B, L, C)."""
    B_, C, H, W = x.shape
    B = B_ // T
    blc = x.view(B, T, C, H, W).permute(0, 1, 3, 4, 2).contiguous().view(B, T * H * W, C)
    return blc


def get_relative_coords_table_all(
        window_size, anchor_window_down_factor=1, T=5
):
    """
    Use case: 3)

    Support all window shapes.
    Args:
        window_size:
        pretrained_window_size:
        anchor_window_down_factor:

    Returns:

    """
    # get relative_coords_table
    ws = window_size
    aws = [w // anchor_window_down_factor for w in window_size]

    # positive table size: (Ww - 1) - (Ww - AWw) // 2
    ts_p = [w1 - 1 - (w1 - w2) // 2 for w1, w2 in zip(ws, aws)]
    # negative table size: -(AWw - 1) - (Ww - AWw) // 2
    ts_n = [-(w2 - 1) - (w1 - w2) // 2 for w1, w2 in zip(ws, aws)]

    # TODO: pretrained window size and pretrained anchor window size is only used here.
    # TODO: Investigate whether it is really important to use this setting when finetuning large window size
    # TODO: based on pretrained weights with small window size.

    coord_h = torch.arange(ts_n[0], ts_p[0] + 1, dtype=torch.float32)
    coord_w = torch.arange(ts_n[1], ts_p[1] + 1, dtype=torch.float32)
    coord_t = torch.arange(0, T, dtype=torch.float32)
    table = torch.stack(torch.meshgrid([coord_t, coord_h, coord_w], indexing="ij")).permute(1, 2, 3,
                                                                                            0).contiguous().unsqueeze(0)
    table[:, :, :, 0] /= ts_p[0]
    table[:, :, :, 1] /= ts_p[1]
    table *= 8  # normalize to -8, 8
    table = torch.sign(table) * torch.log2(torch.abs(table) + 1.0) / np.log2(8)
    # 1, Wh+AWh-1, Ww+AWw-1, 2
    return table


def _get_meshgrid_coords(start_coords, end_coords, T):
    coord_h = torch.arange(start_coords[0], end_coords[0])
    coord_w = torch.arange(start_coords[1], end_coords[1])
    coord_t = torch.arange(0, T)
    coords = torch.stack(torch.meshgrid([coord_t, coord_h, coord_w], indexing="ij"))  # 3,T, Wh, Ww
    coords = torch.flatten(coords, 1)  # 3,T*Wh*Ww
    return coords


def coords_diff_odd(coords1, coords2, start_coord, max_diff):
    # The coordinates starts from (-start_coord[0], -start_coord[1])
    coords = coords1[:, :, None] - coords2[:, None, :]  # 2, Wh*Ww, AWh*AWw
    coords = coords.permute(1, 2, 0).contiguous()  # Wh*Ww, AWh*AWw, 2
    coords[:, :, 0] += start_coord[0]  # shift to start from 0  relative_coords[:, :, 0] += self.window_size[0] - 1
    coords[:, :, 1] += start_coord[1]  # relative_coords[:, :, 1] += self.window_size[1] - 1
    coords[:, :, 0] *= max_diff  # relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
    idx = coords.sum(-1)  # Wh*Ww, AWh*AWw  relative_position_index = relative_coords.sum(-1)
    return idx


def get_relative_position_index_simple(
        window_size, anchor_window_down_factor=1, window_to_anchor=True, T=5
):
    """
    Use case: 3)
    This is a simplified version of get_relative_position_index_all
    The start coordinate of anchor window is also (0, 0)
    get pair-wise relative position index for each token inside the window
    """
    ws = window_size
    aws = [w // anchor_window_down_factor for w in window_size]

    coords = _get_meshgrid_coords((0, 0), window_size, T)  # 2, Wh*Ww
    coords_anchor = _get_meshgrid_coords((0, 0), aws, T)
    # 2, AWh*AWw

    max_horizontal_diff = aws[1] + ws[1] - 1  # 2 * self.window_size[1] - 1
    if window_to_anchor:  # 这块两种条件的区别在于offset，一个是从锚到窗口，一个是从窗口到锚
        offset = [w2 - 1 for w2 in aws]  # self.window_size[0] - 1
        idx = coords_diff_odd(coords, coords_anchor, offset, max_horizontal_diff)
    else:
        offset = [w1 - 1 for w1 in ws]
        idx = coords_diff_odd(coords_anchor, coords, offset, max_horizontal_diff)
    return idx  # Wh*Ww, AWh*AWw or AWh*AWw, Wh*Ww


def calculate_mask_all(
        input_resolution,
        window_size,
        shift_size,
        anchor_window_down_factor=1,
        window_to_anchor=True,
        T=5
):
    """
    Use case: 3)
    """
    # calculate attention mask for SW-MSA
    anchor_resolution = [s // anchor_window_down_factor for s in input_resolution]
    aws = [s // anchor_window_down_factor for s in window_size]
    anchor_shift = [s // anchor_window_down_factor for s in shift_size]

    # mask of window1: nW, Wh**Ww
    mask_windows = _fill_window(input_resolution, window_size, shift_size, T)
    # mask of window2: nW, AWh*AWw
    mask_anchor = _fill_window(anchor_resolution, aws, anchor_shift, T)

    if window_to_anchor:
        attn_mask = mask_windows.unsqueeze(2) - mask_anchor.unsqueeze(1)
    else:
        attn_mask = mask_anchor.unsqueeze(2) - mask_windows.unsqueeze(1)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )  # nW, Wh**Ww, AWh*AWw

    return attn_mask


def _fill_window(input_resolution, window_size, shift_size=None, T=5):
    if shift_size is None:
        shift_size = [s // 2 for s in window_size]

    img_mask = torch.zeros((1, T, *input_resolution, 1))  # 1 N,H W 1
    h_slices = (
        slice(0, -window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    )
    w_slices = (
        slice(0, -window_size[1]),
        slice(-window_size[1], -shift_size[1]),
        slice(-shift_size[1], None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, :, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, T * prod(window_size))
    return mask_windows
