# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 17:18:04
# @Email:  cshzxie@gmail.com

from easydict import EasyDict as edict
import numpy as np

__C                                              = edict()
cfg                                              = __C

#
# Dataset Config
#
__C.DATASETS                                     = edict()
__C.DATASETS.COMPLETION3D                        = edict()
__C.DATASETS.COMPLETION3D.CATEGORY_FILE_PATH     = './datasets/Completion3D.json'
__C.DATASETS.COMPLETION3D.PARTIAL_POINTS_PATH    = '/home/SENSETIME/xiehaozhe/Datasets/Completion3D/%s/partial/%s/%s.h5'
__C.DATASETS.COMPLETION3D.COMPLETE_POINTS_PATH   = '/home/SENSETIME/xiehaozhe/Datasets/Completion3D/%s/gt/%s/%s.h5'
__C.DATASETS.SHAPENET                            = edict()
__C.DATASETS.SHAPENET.CATEGORY_FILE_PATH         = './datasets/ShapeNet.json'
__C.DATASETS.SHAPENET.N_RENDERINGS               = 8
__C.DATASETS.SHAPENET.N_POINTS                   = 16384
__C.DATASETS.SHAPENET.PARTIAL_POINTS_PATH        = '/home/SENSETIME/xiehaozhe/Datasets/ShapeNet/ShapeNetCompletion/%s/partial/%s/%s/%02d.pcd'
__C.DATASETS.SHAPENET.COMPLETE_POINTS_PATH       = '/home/SENSETIME/xiehaozhe/Datasets/ShapeNet/ShapeNetCompletion/%s/complete/%s/%s.pcd'
__C.DATASETS.KITTI                               = edict()
__C.DATASETS.KITTI.CATEGORY_FILE_PATH            = './datasets/KITTI.json'
__C.DATASETS.KITTI.PARTIAL_POINTS_PATH           = '/home/SENSETIME/xiehaozhe/Datasets/KITTI/cars/%s.pcd'
__C.DATASETS.KITTI.BOUNDING_BOX_FILE_PATH        = '/home/SENSETIME/xiehaozhe/Datasets/KITTI/bboxes/%s.txt'
__C.DATASETS.SMLM                                = edict()
__C.DATASETS.SMLM.ROOT_DIR                       = '/repos/datasets/smlm_pc'
__C.DATASETS.SMLM.DATASET_NAME                   = 'mt_pc_131072_2048_30_40_5.5.h5'
__C.DATASETS.SMLM.is_scale_z                     = True
__C.DATASETS.SMLM.is_scale_half                  = False
__C.DATASETS.SMLM.scale                          = 0.9
__C.DATASETS.SMLM.N_POINTS                       = 16384*8 # 16384 for no loop upsampling

#
# Dataset
#
__C.DATASET                                      = edict()
# Dataset Options: Completion3D, ShapeNet, ShapeNetCars, KITTI
__C.DATASET.TRAIN_DATASET                        = 'SMLM'
__C.DATASET.TEST_DATASET                         = 'SMLM'

#
# Constants
#
__C.CONST                                        = edict()
__C.CONST.DEVICE                                 = '0'
__C.CONST.NUM_WORKERS                            = 2
__C.CONST.N_INPUT_POINTS                         = 2048
__C.CONST.UPSAMPLE_RATIO                         = 8 # 8
__C.CONST.N_OUTPUT_DENSE_POINTS                  = __C.CONST.N_INPUT_POINTS * __C.CONST.UPSAMPLE_RATIO
__C.CONST.IMG_SIZE = 128

#
# Directories
#
__C.DIR                                          = edict()
__C.DIR.OUT_PATH                                 = './output'

#
# Memcached
#
__C.MEMCACHED                                    = edict()
__C.MEMCACHED.ENABLED                            = False
__C.MEMCACHED.LIBRARY_PATH                       = '/mnt/lustre/share/pymc/py3'
__C.MEMCACHED.SERVER_CONFIG                      = '/mnt/lustre/share/memcached_client/server_list.conf'
__C.MEMCACHED.CLIENT_CONFIG                      = '/mnt/lustre/share/memcached_client/client.conf'

#
# Network
#
__C.NETWORK                                      = edict()
__C.NETWORK.N_SAMPLING_POINTS                    = 2048
__C.NETWORK.GRIDDING_LOSS_SCALES_SPARSE          = [(128, 128, 16)]
__C.NETWORK.GRIDDING_LOSS_ALPHAS_SPARSE          = [0.1]
__C.NETWORK.GRIDDING_LOSS_SCALES_DENSE           = [(128, 128, 32), (64, 64, 16), (32, 32, 8)]
__C.NETWORK.GRIDDING_LOSS_ALPHAS_DENSE           = [0.3, 0.3, 0.3]
__C.NETWORK.GRIDDING_SCALES                      = (128, 128, 32)
__C.NETWORK.USE_ATTENTION                        = False
__C.NETWORK.USE_3D_UNET_RECON_GRID_L1_LOSS       = False
__C.NETWORK.USE_2D_GRNET2                        = False  # if is True, use 2D GRNet2
__C.NETWORK.USE_IMG_GUIDE                        = True
__C.NETWORK.GRIDDING_LOSS_SCALES_2D              = [(256, 256, 1), (128, 128, 1)]
__C.NETWORK.GRIDDING_LOSS_ALPHAS_2D              = [0.8, 0.2]

#
# Train
#
__C.TRAIN                                        = edict()
__C.TRAIN.BATCH_SIZE                             = 2 # 1
__C.TRAIN.N_EPOCHS                               = 400
__C.TRAIN.SAVE_FREQ                              = 100
__C.TRAIN.LEARNING_RATE                          = 2e-5 # 1e-5
__C.TRAIN.LR_MILESTONES                          = [500]
__C.TRAIN.GAMMA                                  = .5
__C.TRAIN.BETAS                                  = (.9, .999)
__C.TRAIN.WEIGHT_DECAY                           = 1e-6
__C.TRAIN.LOCAL                                  = True
__C.TRAIN.is_random_sample                       = True
__C.TRAIN.is_fine_tune                           = False
__C.TRAIN.transforms                             = True
__C.TRAIN.transforms_params                      = [
    {
        "callback": "RandomMirrorPoints",
        "objects": ["partial_cloud", "gtcloud"],
    },
    {
        "callback": "RandomScalePoints",
        "parameters": {  # based on scale of 0.9 in config
            "scale_low": 0.9,
            "scale_high": 1.05,
        },
        "objects": ["partial_cloud", "gtcloud"],
    },
    {
        'callback': 'RandomRotatePoints',
        'parameters': {
            'angle': np.pi
        },
        'objects': ['partial_cloud', 'gtcloud']
    },
    {"callback": "ToTensor", "objects": ["partial_cloud", "gtcloud", "original_cloud"]},
]
__C.TRAIN.using_original_data_for_dense_gridding = True
__C.TRAIN.using_original_data_for_dense_chamfer  = True
__C.TRAIN.noise_points_ratio                     = 0.0
__C.TRAIN.cdloss_weight                          = 50 # cfg.TRAIN.cdloss_weight * _loss_chamfer_dist
__C.TRAIN.output_pc_csv_freq                     = 20
#
# Test
#
__C.TEST                                         = edict()
__C.TEST.METRIC_NAME                             = 'ChamferDistance'
__C.TEST.LOOP_UPSAMPLE                           = __C.DATASETS.SMLM.N_POINTS // __C.CONST.N_OUTPUT_DENSE_POINTS - 1
