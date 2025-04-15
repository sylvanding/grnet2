# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 17:18:04
# @Email:  cshzxie@gmail.com

from easydict import EasyDict as edict

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
__C.DATASETS.SMLM.ROOT_DIR                       = '/repos/PUDM/datasets/smlm_pc'
__C.DATASETS.SMLM.is_scale_z                     = True
__C.DATASETS.SMLM.is_scale_half                  = False
__C.DATASETS.SMLM.scale                          = 0.9

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
__C.CONST.NUM_WORKERS                            = 4
__C.CONST.N_INPUT_POINTS                         = 2048

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
__C.NETWORK.GRIDDING_LOSS_SCALES_DENSE           = [(256, 256, 32), (128, 128, 16)]
__C.NETWORK.GRIDDING_LOSS_ALPHAS_DENSE           = [0.2, 0.1]

#
# Train
#
__C.TRAIN                                        = edict()
__C.TRAIN.BATCH_SIZE                             = 1
__C.TRAIN.N_EPOCHS                               = 400
__C.TRAIN.SAVE_FREQ                              = 50
__C.TRAIN.LEARNING_RATE                          = 1e-5
__C.TRAIN.LR_MILESTONES                          = [100, 150, 200, 250, 300]
__C.TRAIN.GAMMA                                  = .4
__C.TRAIN.BETAS                                  = (.9, .999)
__C.TRAIN.WEIGHT_DECAY                           = 1e-4
__C.TRAIN.LOCAL                                  = True
__C.TRAIN.is_random_sample                       = False
__C.TRAIN.is_fine_tune                           = False
__C.TRAIN.transforms                             = True
__C.TRAIN.transforms_params                      = [
    {
        "callback": "RandomMirrorPoints",
        "objects": ["partial_cloud", "gtcloud", "original_cloud"],
    },
    {
        "callback": "RandomScalePoints",
        "parameters": {  # based on scale of 0.9 in config
            "scale_low": 0.9,
            "scale_high": 1.05,
        },
        "objects": ["partial_cloud", "gtcloud", "original_cloud"],
    },
    # {
    #     'callback': 'RandomRotatePoints',
    #     'parameters': {
    #         'angle': np.pi
    #     },
    #     'objects': ['partial_cloud', 'gtcloud', 'original_cloud']
    # },
    {"callback": "ToTensor", "objects": ["partial_cloud", "gtcloud", "original_cloud"]},
]
__C.TRAIN.using_original_data_for_dense_gridding = True
__C.TRAIN.using_original_data_for_dense_chamfer  = True
__C.TRAIN.noise_points_ratio                     = 0.1
#
# Test
#
__C.TEST                                         = edict()
__C.TEST.METRIC_NAME                             = 'ChamferDistance'
