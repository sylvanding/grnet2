# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 18:34:19
# @Email:  cshzxie@gmail.com

import matplotlib.pyplot as plt
import numpy as np
import torch

from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO

def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    return x


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.ConvTranspose2d or \
       type(m) == torch.nn.Conv3d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def count_parameters(network):
    return sum(p.numel() for p in network.parameters())


def get_ptcloud_img(ptcloud):
    fig = plt.figure(figsize=(12, 12))

    x, y, z = ptcloud.transpose(1,0) # 3, 2048
    ax = fig.add_subplot(111, projection=Axes3D.name, adjustable='box')
    # ax = fig.gca(projection=Axes3D.name)
    ax.axis('off')
    ax.set_box_aspect([1,1,1]) # 替代 ax.axis('scaled') or ax.axis('auto')
    ax.view_init(elev=90, azim=-90) # 水平倾斜, 旋转：xy平面

    # max, min = np.max(ptcloud), np.min(ptcloud)
    max, min = -1, 1
    ax.set_xbound(min, max)
    ax.set_ybound(min, max)
    ax.set_zbound(min, max)
    sc = ax.scatter(x, y, z, zdir='z', c=z, cmap='jet', s=0.7)
    
    # # add colorbar
    # cbar = fig.colorbar(sc, ax=ax)
    # cbar.set_label('Z')

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)
    return img[200:1000, 200:1000, :]
    # return img


