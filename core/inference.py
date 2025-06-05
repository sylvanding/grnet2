# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-12-23 11:46:33
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 19:12:44
# @Email:  cshzxie@gmail.com

import logging
import os
import torch
import pandas as pd

import utils.data_loaders
import utils.helpers
import utils.io

from models.grnet import GRNet


def denormalize_pc(points, params):
    """
    将归一化的点云恢复原始尺度

    Args:
        points: 归一化后的点云 (b, n, 3)
        params: normalize_pc_pair返回的参数字典

    Returns:
        denormalized_points: 恢复原始尺度的点云
    """
    scale = params['scale'].detach().cpu().numpy()
    centroid = params['centroid'].detach().cpu().numpy()
    return points * scale[0,:,[0]] + centroid[0]


def descale_z(points, params):
    """
    z from [-1, 1] to [-0...,0...]
    """

    scale = params['scale'].detach().cpu().numpy()
    scale = scale[..., 2]
    tensor_points = isinstance(points, torch.Tensor)
    if tensor_points:
        points = points.detach().cpu().numpy()
    if scale.ndim == 2:
        scale = scale[0]
    points[..., 2] *= scale
    if tensor_points:
        points = torch.from_numpy(points).cuda()
    return points


def inference_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data loader
    dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg, 'test')
    test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader,
                                                   batch_size=1,
                                                   num_workers=cfg.CONST.NUM_WORKERS,
                                                   # collate_fn=utils.data_loaders.collate_fn,
                                                   # pin_memory=True,
                                                   shuffle=False)

    # Setup networks and initialize networks
    grnet = GRNet(cfg)

    if torch.cuda.is_available():
        grnet = torch.nn.DataParallel(grnet).cuda()

    # Load the pretrained model from a checkpoint
    logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
    checkpoint = torch.load(cfg.CONST.WEIGHTS)
    grnet.load_state_dict(checkpoint['grnet'])

    # Switch models to evaluation mode
    grnet.eval()

    # The inference loop
    n_samples = len(test_data_loader)
    for model_idx, data in enumerate(test_data_loader):
        # taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        # model_id = model_id[0]

        with torch.no_grad():
            for k, v in data.items():
                if k == 'normalize_params':
                    continue
                data[k] = utils.helpers.var_or_cuda(v)

            sparse_ptcloud, dense_ptcloud = grnet(data)
            # output_folder = os.path.join(cfg.DIR.OUT_PATH, 'benchmark', taxonomy_id)
            output_folder = os.path.join(cfg.DIR.OUT_PATH, 'benchmark')
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            if cfg.DATASETS.SMLM.is_scale_z:
                dense_ptcloud = descale_z(dense_ptcloud, data['normalize_params'])
                data['gtcloud'] = descale_z(data['gtcloud'], data['normalize_params'])

            # output_file_path = os.path.join(output_folder, '%s.h5' % model_id)
            output_file_path = os.path.join(output_folder, '%s_' % model_idx + '%s.xyz')
            # utils.io.IO.put(output_file_path, dense_ptcloud.squeeze().cpu().numpy())

            dense_ptcloud = dense_ptcloud.squeeze().cpu().numpy() # 16384, 3
            df = pd.DataFrame(dense_ptcloud, columns=['x', 'y', 'z'])
            df[['x', 'y', 'z']].to_csv(output_file_path%'dense_ptcloud', sep=' ', index=False, header=False)

            gtcloud = data['gtcloud']
            gtcloud = gtcloud.squeeze().cpu().numpy()
            df = pd.DataFrame(gtcloud, columns=['x', 'y', 'z'])
            df[['x', 'y', 'z']].to_csv(output_file_path%'gtcloud', sep=' ', index=False, header=False)

            # thunderstorm format
            dense_ptcloud_norm = denormalize_pc(dense_ptcloud, data['normalize_params'])
            output_file_path_csv = os.path.join(output_folder, '%s_' % model_idx + '%s.csv')
            df = pd.DataFrame(dense_ptcloud_norm, columns=['x', 'y', 'z'])
            df[['x', 'y', 'z']].to_csv(output_file_path_csv % 'dense_ptcloud', sep=',', index=False, header=True)

            gtcloud_norm = denormalize_pc(gtcloud, data['normalize_params'])
            df = pd.DataFrame(gtcloud_norm, columns=['x', 'y', 'z'])
            df[['x', 'y', 'z']].to_csv(output_file_path_csv % 'gtcloud', sep=',', index=False, header=True)

            # logging.info('Test[%d/%d] Taxonomy = %s Sample = %s File = %s' %
            #              (model_idx + 1, n_samples, taxonomy_id, model_id, output_file_path))
