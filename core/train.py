# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-07-04 11:01:37
# @Email:  cshzxie@gmail.com

import logging
import os
import torch
import torch.nn as nn

import utils.data_loaders
import utils.helpers


from datetime import datetime
from time import time
from tensorboardX import SummaryWriter

from core.test import test_net
from extensions.gridding import Gridding
from extensions.chamfer_dist import ChamferDistance
from extensions.gridding_loss import GriddingLoss
from models.grnet_2 import GRNet_2
from models.grnet_2_2D import GRNet_2_2D
from utils.average_meter import AverageMeter
from utils.metrics import Metrics
from tqdm import tqdm
import shutil


def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data loader
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg, 'train')
    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg, 'valid')
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader,
                                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                                    num_workers=cfg.CONST.NUM_WORKERS,
                                                    # collate_fn=utils.data_loaders.collate_fn,
                                                    # pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader,
                                                  batch_size=1,
                                                  num_workers=cfg.CONST.NUM_WORKERS,
                                                  # collate_fn=utils.data_loaders.collate_fn,
                                                  # pin_memory=True,
                                                  shuffle=False)

    # Set up folders for logs and checkpoints
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s')
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    cfg.DIR.LOGS = output_dir % 'logs'
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)

    if os.path.exists(cfg.DIR.LOGS):
        if True:
            logging.warning('Logs directory already exists. It will be removed.')
            # os.system('rm -rf %s' % cfg.DIR.LOGS)
            shutil.rmtree(cfg.DIR.LOGS)
        else:
            logging.warning('Logs directory already exists. It will be renamed.')
            # if not os.path.exists(os.path.join(cfg.DIR.OUT_PATH, 'logs_before')):
            #     os.makedirs(os.path.join(cfg.DIR.OUT_PATH, 'logs_before'))
            os.rename(cfg.DIR.LOGS,
                      os.path.join(cfg.DIR.OUT_PATH, 'logs_before_%s') % datetime.now().strftime(
                          '%Y-%m-%d_%H-%M-%S'))

    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    # Create the networks
    if cfg.NETWORK.USE_2D_GRNET2:
        grnet = GRNet_2_2D(cfg)
    else:
        grnet = GRNet_2(cfg)
    grnet.apply(utils.helpers.init_weights)
    logging.debug('Parameters in GRNet: %d.' % utils.helpers.count_parameters(grnet))

    # Move the network to GPU if possible
    if torch.cuda.is_available():
        grnet = torch.nn.DataParallel(grnet).cuda()

    # Create the optimizers
    grnet_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, grnet.parameters()),
                                       lr=cfg.TRAIN.LEARNING_RATE,
                                       weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                       betas=cfg.TRAIN.BETAS)
    grnet_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(grnet_optimizer,
                                                              milestones=cfg.TRAIN.LR_MILESTONES,
                                                              gamma=cfg.TRAIN.GAMMA)

    # Set up loss functions
    chamfer_dist = ChamferDistance()
    # gridding_loss_sparse = GriddingLoss(
    #     scales=cfg.NETWORK.GRIDDING_LOSS_SCALES_SPARSE,
    #     alphas=cfg.NETWORK.GRIDDING_LOSS_ALPHAS_SPARSE)
    gridding_loss_dense = GriddingLoss(
        scales=cfg.NETWORK.GRIDDING_LOSS_SCALES_DENSE,
        alphas=cfg.NETWORK.GRIDDING_LOSS_ALPHAS_DENSE)
    if cfg.NETWORK.USE_3D_UNET_RECON_GRID_L1_LOSS:
        gridding_scales = (128, 128, 32)
        gridding = Gridding(scales=gridding_scales)
        l1_loss = nn.L1Loss()
    else:
        gridding_scales = None
        gridding = None
        l1_loss = None

    # Load pretrained model if exists
    init_epoch = 0
    best_metrics = None
    if 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        best_metrics = Metrics(cfg.TEST.METRIC_NAME, checkpoint['best_metrics'])
        grnet.load_state_dict(checkpoint['grnet'])
        logging.info('Recover complete. Current epoch = #%d; best metrics = %s.' % (init_epoch, best_metrics))

    # Training/Testing the network
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
        if epoch_idx < 50: # train sparse
            alpha = 0.1
            gridding_loss_sparse_ratio = 0.1
            gridding_loss_dense_ratio = 0.1
        elif epoch_idx < 100: # train sparse and dense
            alpha = 0.5
            gridding_loss_sparse_ratio = 0.15
            gridding_loss_dense_ratio = 0.15
        else: # train dense
            alpha = 1.0
            gridding_loss_sparse_ratio = 0.15
            gridding_loss_dense_ratio = 0.2
        if cfg.TRAIN.is_fine_tune:
            alpha = 1.0

        epoch_start_time = time()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['ChamferDist', 'GriddingLoss', 'L1_3d_unet_recon_grid'])

        grnet.train()

        batch_end_time = time()
        n_batches = len(train_data_loader)
        for batch_idx, data in enumerate(train_data_loader):
            data_time.update(time() - batch_end_time)
            for k, v in data.items():
                data[k] = utils.helpers.var_or_cuda(v)

            dense_cloud_interp, dense_cloud_pred, pt_features_xyz_r = grnet(data)

            if cfg.NETWORK.USE_3D_UNET_RECON_GRID_L1_LOSS:
                gridding_gt = gridding(data['gtcloud']).view(-1, 1, *gridding_scales)
                _loss_l1_3d_unet_recon_grid = l1_loss(pt_features_xyz_r, gridding_gt)
            else:
                _loss_l1_3d_unet_recon_grid = torch.tensor(0.0, device=pt_features_xyz_r.device)

            _loss_chamfer_dist = chamfer_dist(dense_cloud_pred, data['gtcloud'])
            _loss_gridding_loss = gridding_loss_dense(dense_cloud_pred, data['gtcloud'])
            _loss_chamfer_dist = cfg.TRAIN.cdloss_weight * _loss_chamfer_dist
            if cfg.NETWORK.USE_3D_UNET_RECON_GRID_L1_LOSS:
                _loss = 0.4 * _loss_chamfer_dist + 0.4 * _loss_gridding_loss + 0.2 * _loss_l1_3d_unet_recon_grid
            else:
                _loss = 0.5 * _loss_chamfer_dist + 0.5 * _loss_gridding_loss
            losses.update([_loss_chamfer_dist.item() * 1000, _loss_gridding_loss.item() * 1000, _loss_l1_3d_unet_recon_grid.item() * 1000])

            grnet.zero_grad()
            _loss.backward()
            grnet_optimizer.step()

            n_itr = (epoch_idx - 1) * n_batches + batch_idx
            train_writer.add_scalar('Loss/Batch/ChamferDist', _loss_chamfer_dist.item() * 1000, n_itr)
            train_writer.add_scalar('Loss/Batch/GriddingLoss', _loss_gridding_loss.item() * 1000, n_itr)
            train_writer.add_scalar('Loss/Batch/L1_3d_unet_recon_grid', _loss_l1_3d_unet_recon_grid.item() * 1000, n_itr)

            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            logging.info('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s' %
                         (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches, batch_time.val(), data_time.val(),
                          ['%.4f' % l for l in losses.val()]))

        grnet_lr_scheduler.step()
        epoch_end_time = time()
        train_writer.add_scalar('Loss/Epoch/ChamferDist', losses.avg(0), epoch_idx)
        train_writer.add_scalar('Loss/Epoch/GriddingLoss', losses.avg(1), epoch_idx)
        train_writer.add_scalar('Loss/Epoch/L1_3d_unet_recon_grid', losses.avg(2), epoch_idx)
        train_writer.add_scalar('Loss/Epoch/LR', grnet_optimizer.param_groups[0]['lr'], epoch_idx)
        logging.info(
            '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
            (epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]))

        # Validate the current model
        metrics = test_net(cfg, epoch_idx, val_data_loader, val_writer, grnet,
                           gridding_loss_sparse_ratio=gridding_loss_sparse_ratio,
                           gridding_loss_dense_ratio=gridding_loss_dense_ratio,
                           alpha=alpha)

        # Save ckeckpoints
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or cfg.TRAIN.LOCAL:
            file_name = 'ckpt-best.pth' if metrics.better_than(best_metrics) else 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            torch.save({
                'epoch_index': epoch_idx,
                'best_metrics': metrics.state_dict(),
                'grnet': grnet.state_dict()
            }, output_path)  # yapf: disable

            logging.info('Saved checkpoint to %s ...' % output_path)
            if metrics.better_than(best_metrics):
                best_metrics = metrics

    train_writer.close()
    val_writer.close()
