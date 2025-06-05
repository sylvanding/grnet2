# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 19:29:37
# @Email:  cshzxie@gmail.com

import logging
import os
import torch
import torch.nn as nn
from extensions.gridding import Gridding
import utils.data_loaders
import utils.helpers
import utils.data_saver

from extensions.chamfer_dist import ChamferDistance
from extensions.gridding_loss import GriddingLoss
from models.grnet_2 import GRNet_2
from utils.average_meter import AverageMeter
from utils.metrics import Metrics
from tqdm import tqdm


def _sample_points(pc, n_points):
    """
    Randomly sample n_points from the point cloud.
    """
    num_points_in_pool = pc.size(1)
        
    if num_points_in_pool == 0:
        logging.warning("Sampling pool is empty during loop upsampling. Stopping loop.")
        return None
    if num_points_in_pool >= n_points:
        indices = torch.randperm(num_points_in_pool, device=pc.device)[:n_points]
        sampled_input_pc_loop = pc[:, indices, :]
    else:
        logging.warning(
            f"Sampling pool size ({num_points_in_pool}) is less than N_INPUT_POINTS ({n_points}). "
            f"Sampling with replacement."
        )
        indices = torch.randint(0, num_points_in_pool, (n_points,), device=pc.device)
        sampled_input_pc_loop = pc[:, indices, :]

    return sampled_input_pc_loop


def _perform_loop_upsampling(grnet, original_input_pc_for_loop, initial_dense_cloud_pred, cfg):
    """
    Performs loop upsampling to refine the predicted dense point cloud.
    """
    if cfg.TEST.LOOP_UPSAMPLE <= 0:
        return initial_dense_cloud_pred

    current_dense_cloud_pred = initial_dense_cloud_pred.cpu()
    
    # Ensure original_input_pc_for_loop is on the correct device, same as predictions

    N_INPUT_POINTS_loop = cfg.CONST.N_INPUT_POINTS
    LOOP_UPSAMPLE_count_val = cfg.TEST.LOOP_UPSAMPLE

    sampling_pool = original_input_pc_for_loop.cpu()

    # logging.info(f"Starting loop upsampling for {LOOP_UPSAMPLE_count_val} iterations.")
    for i in range(LOOP_UPSAMPLE_count_val):
        # logging.info(f"Loop upsampling iteration {i + 1}/{LOOP_UPSAMPLE_count_val}")
        
        # 1. Merge current dense prediction and original input_pc for sampling
        # Ensure devices are consistent for concatenation
        sampling_pool = torch.cat((current_dense_cloud_pred, sampling_pool), dim=1).cpu()

        # 2. Sample N_INPUT_POINTS from the merged cloud
        sampled_input_pc_loop = _sample_points(sampling_pool, N_INPUT_POINTS_loop)

        # 3. Get new prediction and other outputs from grnet
        _, new_pred, _ = grnet(sampled_input_pc_loop.cuda())
        
        # 4. Update current outputs
        current_dense_cloud_pred = new_pred.cpu()

    # logging.info("Loop upsampling finished.")
    return torch.cat((sampling_pool, current_dense_cloud_pred), dim=1)


def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, grnet=None, **kwargs):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    if test_data_loader is None:
        # Set up data loader
        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg, 'test-exp') # test
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader,
                                                       batch_size=1,
                                                       num_workers=cfg.CONST.NUM_WORKERS,
                                                       # collate_fn=utils.data_loaders.collate_fn,
                                                       # pin_memory=True,
                                                       shuffle=False)

    # Setup networks and initialize networks
    if grnet is None:
        grnet = GRNet_2(cfg)

        if torch.cuda.is_available():
            grnet = torch.nn.DataParallel(grnet).cuda()

        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        grnet.load_state_dict(checkpoint['grnet'])

    # Switch models to evaluation mode
    grnet.eval()

    # Set up loss functions
    chamfer_dist = ChamferDistance()
    # gridding_loss_sparse = GriddingLoss(
    #     scales=cfg.NETWORK.GRIDDING_LOSS_SCALES_SPARSE,
    #     alphas=cfg.NETWORK.GRIDDING_LOSS_ALPHAS_SPARSE)
    gridding_loss_dense = GriddingLoss(
        scales=cfg.NETWORK.GRIDDING_LOSS_SCALES_DENSE,
        alphas=cfg.NETWORK.GRIDDING_LOSS_ALPHAS_DENSE)
    
    if cfg.NETWORK.USE_3D_UNET_RECON_GRID_L1_LOSS:
        l1_loss = nn.L1Loss()
        gridding_scales = (128, 128, 32)
        gridding = Gridding(scales=gridding_scales)
    else:
        l1_loss = None
        gridding_scales = None
        gridding = None

    # Testing loop
    n_samples = len(test_data_loader)
    test_losses = AverageMeter(['ChamferDist', 'GriddingLoss', 'L1_3d_unet_recon_grid'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()

    # Testing loop
    for model_idx, data in enumerate(test_data_loader):
        # taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        # model_id = model_id[0]

        with torch.no_grad():
            for k, v in data.items():
                if k == 'normalize_params':
                    continue
                data[k] = utils.helpers.var_or_cuda(v)

            dense_cloud_interp, dense_cloud_pred, pt_features_xyz_r = grnet(data)
            _loss_chamfer_dist = chamfer_dist(dense_cloud_pred, data['gtcloud'])
            _loss_gridding_loss = gridding_loss_dense(dense_cloud_pred, data['gtcloud'])
            if cfg.NETWORK.USE_3D_UNET_RECON_GRID_L1_LOSS:
                gridding_gt = gridding(data['gtcloud']).view(-1, 1, *gridding_scales)
                _loss_l1_3d_unet_recon_grid = l1_loss(pt_features_xyz_r, gridding_gt)
            else:
                _loss_l1_3d_unet_recon_grid = torch.tensor(0.0, device=pt_features_xyz_r.device)
            _loss_chamfer_dist = cfg.TRAIN.cdloss_weight * _loss_chamfer_dist
            if cfg.NETWORK.USE_3D_UNET_RECON_GRID_L1_LOSS:
                _loss = 0.4 * _loss_chamfer_dist + 0.4 * _loss_gridding_loss + 0.2 * _loss_l1_3d_unet_recon_grid
            else:
                _loss = 0.5 * _loss_chamfer_dist + 0.5 * _loss_gridding_loss
            test_losses.update([_loss_chamfer_dist.item() * 1000, _loss_gridding_loss.item() * 1000, _loss_l1_3d_unet_recon_grid.item() * 1000])
            _metrics = Metrics.get(dense_cloud_pred, data['gtcloud'])
            test_metrics.update(_metrics)

            # if taxonomy_id not in category_metrics:
            #     category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            # category_metrics[taxonomy_id].update(_metrics)

            if test_writer is not None and model_idx < 3:
                print('start to write images with test_writer')
                scale = cfg.DATASETS.SMLM.scale
                
                # Loop upsampling
                dense_cloud_pred_loop = _perform_loop_upsampling(grnet, data['partial_cloud'], dense_cloud_pred, cfg)
                dense_cloud_pred_loop = dense_cloud_pred_loop.squeeze().cpu().numpy()
                dense_cloud_pred_loop_img = utils.helpers.get_ptcloud_img(dense_cloud_pred_loop/scale)
                test_writer.add_image('Model%02d/DensePredReconstruction-Loop' % model_idx, dense_cloud_pred_loop_img, epoch_idx, dataformats='HWC')
                logging.info('Shape of dense_cloud_pred_loop: %s', dense_cloud_pred_loop.shape)
                
                input_pc = data['partial_cloud'].squeeze().cpu().numpy()
                input_pc_img = utils.helpers.get_ptcloud_img(input_pc/scale)
                test_writer.add_image('Model%02d/Input' % model_idx, input_pc_img, epoch_idx, dataformats='HWC')
                dense_cloud_interp = dense_cloud_interp.squeeze().cpu().numpy()  # duplicated points
                dense_cloud_interp_img = utils.helpers.get_ptcloud_img(dense_cloud_interp/scale)
                test_writer.add_image('Model%02d/InterpReconstruction' % model_idx, dense_cloud_interp_img, epoch_idx, dataformats='HWC')
                dense_cloud_pred = dense_cloud_pred.squeeze().cpu().numpy()
                dense_cloud_pred_img = utils.helpers.get_ptcloud_img(dense_cloud_pred/scale)
                test_writer.add_image('Model%02d/DensePredReconstruction' % model_idx, dense_cloud_pred_img, epoch_idx, dataformats='HWC')
                gt_ptcloud = data['gtcloud'].squeeze().cpu().numpy()
                gt_ptcloud_img = utils.helpers.get_ptcloud_img(gt_ptcloud/scale)
                test_writer.add_image('Model%02d/GroundTruth-Sampled' % model_idx, gt_ptcloud_img, epoch_idx, dataformats='HWC')
                
                original_pc = data['original_cloud'].squeeze().cpu().numpy()
                original_pc_img = utils.helpers.get_ptcloud_img(original_pc/scale)
                test_writer.add_image('Model%02d/Original' % model_idx, original_pc_img, epoch_idx, dataformats='HWC')
                
                if (epoch_idx % cfg.TRAIN.output_pc_csv_freq == 0 and epoch_idx > 0) or cfg.TRAIN.LOCAL:
                    # save the pc to csv
                    save_path = os.path.join(cfg.DIR.OUT_PATH, 'output_pc_csv',f'epoch_{epoch_idx}')
                    utils.data_saver.save_pc_to_csv(cfg, dense_cloud_pred_loop, save_path, f'dense_cloud_pred_loop', data['normalize_params'])
                    utils.data_saver.save_pc_to_csv(cfg, input_pc, save_path, f'input_pc', data['normalize_params'])
                    utils.data_saver.save_pc_to_csv(cfg, gt_ptcloud, save_path, f'gt_ptcloud', data['normalize_params'])
                    utils.data_saver.save_pc_to_csv(cfg, original_pc, save_path, f'original_pc', data['normalize_params'])
                
            # logging.info('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
            #              (model_idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()
            #                                                                 ], ['%.4f' % m for m in _metrics]))
            logging.info('Test[%d/%d] Losses = %s Metrics = %s' %
                         (model_idx + 1, n_samples, ['%.4f' % l for l in test_losses.val()
                                                     ], ['%.4f' % m for m in _metrics]))

    # Print testing results
    print('============================ TEST RESULTS ============================')
    
    # print('Taxonomy', end='\t')
    # print('#Sample', end='\t')
    # for metric in test_metrics.items:
    #     print(metric, end='\t')
    # print()
    #
    # for taxonomy_id in category_metrics:
    #     print(taxonomy_id, end='\t')
    #     print(category_metrics[taxonomy_id].count(0), end='\t')
    #     for value in category_metrics[taxonomy_id].avg():
    #         print('%.4f' % value, end='\t')
    #     print()

    print('Overall', end='\t\t\t')
    for value in test_metrics.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('Loss/Epoch/ChamferDist', test_losses.avg(0), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/GriddingLoss', test_losses.avg(1), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/L1_3d_unet_recon_grid', test_losses.avg(2), epoch_idx)
        for i, metric in enumerate(test_metrics.items):
            test_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch_idx)

    return Metrics(cfg.TEST.METRIC_NAME, test_metrics.avg())
