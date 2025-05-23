import os
import numpy as np
import torch
import pickle
import h5py
import random
import math
from datetime import datetime
from einops import rearrange, repeat
from extensions.pointops.functions import pointops
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt


def index_points(pts, idx):
    """
    Input:
        pts: input points data, [B, C, N]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, C, S, [K]]
    """
    batch_size = idx.shape[0]
    sample_num = idx.shape[1]
    fdim = pts.shape[1]
    reshape = False
    if len(idx.shape) == 3:
        reshape = True
        idx = idx.reshape(batch_size, -1)
    # (b, c, (s k))
    res = torch.gather(pts, 2, idx[:, None].repeat(1, fdim, 1))
    if reshape:
        res = rearrange(res, "b c (s k) -> b c s k", s=sample_num)

    return res


def FPS(pts, fps_pts_num):
    # input: (b, 3, n)

    # (b, n, 3)
    pts_trans = rearrange(pts, "b c n -> b n c").contiguous()
    # (b, fps_pts_num)
    sample_idx = pointops.furthestsampling(pts_trans, fps_pts_num).long()
    # (b, 3, fps_pts_num)
    sample_pts = index_points(pts, sample_idx)

    return sample_pts


def get_knn_pts(k, pts, center_pts, return_idx=False):
    # input: (b, 3, n)
    # 1，GT，sample
    # (b, n, 3)
    pts_trans = rearrange(pts, "b c n -> b n c").contiguous()
    # (b, m, 3)
    center_pts_trans = rearrange(center_pts, "b c m -> b m c").contiguous()
    # (b, m, k)
    knn_idx = pointops.knnquery_heap(k, pts_trans, center_pts_trans).long()
    # (b, 3, m, k)
    knn_pts = index_points(pts, knn_idx)

    if return_idx is False:
        return knn_pts
    else:
        return knn_pts, knn_idx


def midpoint_interpolate(
    sparse_pts,
    up_rate=4,
):
    # sparse_pts: (b, 3, 256)

    pts_num = sparse_pts.shape[-1]
    up_pts_num = int(pts_num * up_rate)
    k = int(2 * up_rate)
    # (b, 3, n, k)
    knn_pts = get_knn_pts(k, sparse_pts, sparse_pts)
    # (b, 3, n, k)
    repeat_pts = repeat(sparse_pts, "b c n -> b c n k", k=k)
    # (b, 3, n, k)
    mid_pts = (knn_pts + repeat_pts) / 2.0
    # (b, 3, (n k))
    mid_pts = rearrange(mid_pts, "b c n k -> b c (n k)")
    # note that interpolated_pts already contain sparse_pts
    interpolated_pts = mid_pts
    # fps: (b, 3, up_pts_num)
    interpolated_pts = FPS(interpolated_pts, up_pts_num)

    return interpolated_pts
