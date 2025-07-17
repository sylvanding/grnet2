# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-07-31 16:57:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 19:21:32
# @Email:  cshzxie@gmail.com

import sys

sys.path.append("/repos/grnet2")

import json
import logging
import numpy as np
import random
import torch.utils.data.dataset

import utils.data_transforms

from enum import Enum, unique
from tqdm import tqdm

from utils.io import IO

import copy
from typing import Union, Tuple
import h5py
import os
from utils.render import render_to_2d_image


@unique
class DatasetSubset(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


def collate_fn(batch):
    taxonomy_ids = []
    model_ids = []
    data = {}

    for sample in batch:
        taxonomy_ids.append(sample[0])
        model_ids.append(sample[1])
        _data = sample[2]
        for k, v in _data.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

    for k, v in data.items():
        data[k] = torch.stack(v, 0)

    return taxonomy_ids, model_ids, data


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, options, file_list, transforms=None):
        self.options = options
        self.file_list = file_list
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = -1
        if 'n_renderings' in self.options:
            rand_idx = random.randint(0, self.options['n_renderings'] - 1) if self.options['shuffle'] else 0

        for ri in self.options['required_items']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]

            data[ri] = IO.get(file_path).astype(np.float32)

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], data


class SMLMDataLoader(torch.utils.data.dataset.Dataset):

    def __init__(
            self, cfg, split
    ):
        assert split in ["train", "valid", "test", "test-exp"], "split error value!"

        self.dataroot = cfg.DATASETS.SMLM.ROOT_DIR
        self.dataset_name = cfg.DATASETS.SMLM.DATASET_NAME
        self.split = split
        self.local = cfg.TRAIN.LOCAL
        self.is_scale_z = cfg.DATASETS.SMLM.is_scale_z
        self.is_scale_half = cfg.DATASETS.SMLM.is_scale_half
        self.scale = cfg.DATASETS.SMLM.scale
        self.n_output_dense_points = cfg.CONST.N_OUTPUT_DENSE_POINTS
        self.use_2d_grnet2 = cfg.NETWORK.USE_2D_GRNET2
        self.use_img_guide = cfg.NETWORK.USE_IMG_GUIDE
        self.img_size = cfg.CONST.IMG_SIZE
        if self.split == "train":
            self.is_random_sample = cfg.TRAIN.is_random_sample
        else:
            self.is_random_sample = False

        data_split = {
            "local": {"train": [0, 5], "valid": [1000, 1024], "test": [1001, 1002], "test-exp": [0, 1]},
            "remote": {"train": [0, 1000], "valid": [1000, 1024], "test": [1001, 1002], "test-exp": [0, 1]},
        }
        if self.split != "test-exp":
            h5_file_path = os.path.join(self.dataroot, self.dataset_name)
        else:
            h5_file_path = os.path.join("datasets/region_x0_y0_z1_2048_16384_norm.h5")

        if self.local:
            start, end = data_split["local"][self.split]
        else:
            start, end = data_split["remote"][self.split]

        with h5py.File(h5_file_path, "r") as f:
            self.input_data = f["input_data"][start:end].astype(np.float32)
            self.gt_data = f["gt_data"][start:end].astype(np.float32)
            # self.original_data = f["original_data"][start:end].astype(np.float32)
            normalize_params = f["norm_params"]
            self.normalize_params = {
                "centroid": normalize_params["centroid"][start:end].astype(np.float32),
                "scale": normalize_params["scale"][start:end].astype(np.float32),
            }
        assert self.input_data.shape[0] == self.gt_data.shape[0]

        if self.is_scale_z:
            self.input_data, self.gt_data, self.normalize_params = SMLMDataLoader.scale_z(  # type: ignore
                self.input_data, self.gt_data, self.normalize_params
            )

        if self.is_scale_half:
            # from -1~1 to -0.5~0.5
            self.input_data /= 2
            self.gt_data /= 2
            # self.original_data /= 2

        self.input_data = self.input_data * self.scale
        self.gt_data = self.gt_data * self.scale
        # self.original_data = self.original_data * self.scale
        
        print("input_data.shape:", self.input_data.shape)
        print("gt_data.shape:", self.gt_data.shape)
        # print("original_data.shape:", self.original_data.shape)
        print("local:", self.local)
        print("is_scale_z:", self.is_scale_z)
        print("is_scale_half:", self.is_scale_half)
        print("scale:", self.scale)
        
        self.transforms = None
        if cfg.TRAIN.transforms:
            self.transforms = self._get_transforms(cfg, self.split)
        
        self.noise_points_ratio = cfg.TRAIN.noise_points_ratio
        if self.split == "train" and self.noise_points_ratio > 0:
            self.input_data = self._get_noise(self.input_data, self.noise_points_ratio)
        
        print("noise added. input_data.shape:", self.input_data.shape)
        print("max of input_data:", np.max(self.input_data,axis=(0,1)))
        print("min of input_data:", np.min(self.input_data,axis=(0,1)))

    def _get_transforms(self, cfg, split):
        if split != "train":
            return None
        return utils.data_transforms.Compose(cfg.TRAIN.transforms_params)

    def _get_noise(self, pc, noise_points_ratio):
        """
        为点云添加噪声点
        
        Args:
            pc: 点云数据，形状为(点云数，点数，3)
            noise_points_ratio: 噪声点比例
            
        Returns:
            添加了噪声点的点云
        """
        if noise_points_ratio <= 0:
            return pc
        
        # 获取点云形状
        batch_size, num_points, _ = pc.shape
        
        # 计算要添加的噪声点数量
        noise_points_count = int(num_points * noise_points_ratio)
        
        # 如果噪声点数量为0，直接返回原始点云
        if noise_points_count == 0:
            return pc
        
        # 创建结果点云的副本
        result_pc = copy.deepcopy(pc)
        
        for i in range(batch_size):
            # 为每个点云生成随机噪声点
            # 分别按照点云x,y,z轴的最大最小值来设置噪声点的范围
            pc_min = np.min(pc[i], axis=0)  # 获取x,y,z三个维度的最小值
            pc_max = np.max(pc[i], axis=0)  # 获取x,y,z三个维度的最大值
            
            # 在每个维度的范围内分别随机生成噪声点坐标
            noise_points = np.zeros((noise_points_count, 3))
            for dim in range(3):  # 分别处理x,y,z三个维度
                noise_points[:, dim] = np.random.uniform(
                    low=pc_min[dim],
                    high=pc_max[dim],
                    size=noise_points_count
                )
            
            # 随机选择要删除的点的索引
            indices_to_remove = np.random.choice(
                num_points, 
                noise_points_count, 
                replace=False
            )
            
            # 删除选定的点
            remaining_indices = np.setdiff1d(np.arange(num_points), indices_to_remove)
            
            # 将剩余点与噪声点合并
            result_pc[i] = np.vstack([
                result_pc[i][remaining_indices],
                noise_points
            ])
                
        return result_pc

    def __getitem__(self, index):
        result = {}
        partial_pc = copy.deepcopy(self.input_data[index])
        complete_pc = copy.deepcopy(self.gt_data[index])
        # original_pc = copy.deepcopy(self.original_data[index])
        
        if self.use_2d_grnet2:
            partial_pc[:, 2] = 0
            complete_pc[:, 2] = 0
        
        # check point number of complete_pc
        complete_pc_sampled = self.random_sample(complete_pc, self.n_output_dense_points)
        
        if self.split == "train" or self.split == "valid":
            if self.is_random_sample:
                partial_pc = self.random_sample(complete_pc, 2048)
            result['partial_cloud'] = partial_pc
            result['gtcloud'] = complete_pc_sampled
            result['original_cloud'] = complete_pc
            if self.split == "train":
                # augment
                if self.transforms is not None:
                    result = self.transforms(result)
                    if self.use_2d_grnet2:
                        result = self.clamp_points(result)
            if self.split == "valid":
                normalize_params = {
                    "centroid": self.normalize_params["centroid"][index],
                    "scale": self.normalize_params["scale"][index],
                }
                result['normalize_params'] = normalize_params
        else:
            normalize_params = {
                "centroid": self.normalize_params["centroid"][index],
                "scale": self.normalize_params["scale"][index],
            }
            result['partial_cloud'] = partial_pc
            result['gtcloud'] = complete_pc_sampled
            result['original_cloud'] = complete_pc
            result['normalize_params'] = normalize_params
        # 如果使用2D GRNet2，则将点云归一化到-1~1之间
        # if self.use_2d_grnet2:
        #     result['partial_cloud'] = torch.clamp(result['partial_cloud'], -1, 1)
        #     result['gtcloud'] = torch.clamp(result['gtcloud'], -1, 1)
        #     result['original_cloud'] = torch.clamp(result['original_cloud'], -1, 1)
        # if self.use_2d_grnet2:
        #     notzero = torch.sum(result['partial_cloud'][:,2] != 0)
        #     if notzero>0:
        #         logging.error("notzero: %d", notzero)
        #     if torch.isnan(result['partial_cloud']).any():
        #         logging.error("partial_cloud has NaN")
        #     if torch.isinf(result['partial_cloud']).any():
        #         logging.error("partial_cloud has Inf")
        if self.use_img_guide:
            # The point cloud is normalized to [-1, 1].
            # We need to scale it to pixel coordinates [0, img_size].
            points_for_render = result["original_cloud"]
            if isinstance(points_for_render, torch.Tensor):
                points_for_render = points_for_render.numpy()
            
            points_pixel_coords = (points_for_render + 1) * self.img_size / 2

            guide_image = render_to_2d_image(
                points=points_pixel_coords,
                image_size=(self.img_size, self.img_size),
                psf_sigma_px=1.0, 
                output_dtype=np.float32
            )
            # Normalize to [0, 1] and add channel dimension
            if guide_image.max() > 0:
                guide_image = guide_image / guide_image.max()
            
            result["guide_img"] = torch.from_numpy(guide_image).unsqueeze(0).float()
            
        return result

    def clamp_points(self, result):
        result['partial_cloud'] = torch.clamp(result['partial_cloud'], -0.95, 0.95)
        result['gtcloud'] = torch.clamp(result['gtcloud'], -0.95, 0.95)
        if 'original_cloud' in result:
            result['original_cloud'] = torch.clamp(result['original_cloud'], -0.95, 0.95)
        return result

    def __len__(self):
        return len(self.input_data)

    @staticmethod
    def scale_z(
        input_data, gt_data, params
    ) -> Union[Tuple[np.ndarray, np.ndarray, dict], Tuple[np.ndarray, np.ndarray]]:
        """
        z轴scale

        Args:
            points: 归一化后的点云 (b, n, 3)
            params: normalize_pc_pair返回的参数字典

        Returns:
            scaled_points: z轴scale后的点云
            params: 更新后的参数字典
        """
        # input_data = input_data.detach().cpu().numpy()  # (B, n, 3)
        # gt_data = gt_data.detach().cpu().numpy()  # (B, n, 3)
        # scale = params['scale'].detach().cpu().numpy() # (B, 1, 3)
        # 只对z轴进行scale：z from 0~ to 0~1
        furthest_distances_z = np.amax(
            np.abs(gt_data[..., 2]), axis=1, keepdims=True  # B, n, 1
        )  # (b, 1, 1)
        gt_data[..., 2] = gt_data[..., 2] / furthest_distances_z
        input_data[..., 2] = input_data[..., 2] / furthest_distances_z
        if params is not None:
            params["scale"][..., 2] = furthest_distances_z
            return input_data, gt_data, params
        else:
            return input_data, gt_data

    def random_sample(self, pc, n):
        idx = np.random.permutation(pc.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n - pc.shape[0])])
        return pc[idx[:n]]

class ShapeNetDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.SHAPENET.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        n_renderings = self.cfg.DATASETS.SHAPENET.N_RENDERINGS if subset == DatasetSubset.TRAIN else 1
        file_list = self._get_file_list(self.cfg, self._get_subset(subset), n_renderings)
        transforms = self._get_transforms(self.cfg, subset)
        return Dataset({
            'n_renderings': n_renderings,
            'required_items': ['partial_cloud', 'gtcloud'],
            'shuffle': subset == DatasetSubset.TRAIN
        }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': cfg.CONST.N_INPUT_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            return utils.data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': cfg.CONST.N_INPUT_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'partial_cloud_path': [
                        cfg.DATASETS.SHAPENET.PARTIAL_POINTS_PATH % (subset, dc['taxonomy_id'], s, i)
                        for i in range(n_renderings)
                    ],
                    'gtcloud_path':
                    cfg.DATASETS.SHAPENET.COMPLETE_POINTS_PATH % (subset, dc['taxonomy_id'], s),
                })

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list


class ShapeNetCarsDataLoader(ShapeNetDataLoader):
    def __init__(self, cfg):
        super(ShapeNetCarsDataLoader, self).__init__(cfg)

        # Remove other categories except cars
        self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']


class Completion3DDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.COMPLETION3D.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = self._get_transforms(self.cfg, subset)
        required_items = ['partial_cloud'] if subset == DatasetSubset.TEST else ['partial_cloud', 'gtcloud']

        return Dataset({
            'required_items': required_items,
            'shuffle': subset == DatasetSubset.TRAIN
        }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return utils.data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': cfg.CONST.N_INPUT_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            return utils.data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': cfg.CONST.N_INPUT_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'partial_cloud_path':
                    cfg.DATASETS.COMPLETION3D.PARTIAL_POINTS_PATH % (subset, dc['taxonomy_id'], s),
                    'gtcloud_path':
                    cfg.DATASETS.COMPLETION3D.COMPLETE_POINTS_PATH % (subset, dc['taxonomy_id'], s),
                })

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list


class KittiDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.KITTI.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = self._get_transforms(self.cfg, subset)
        required_items = ['partial_cloud', 'bounding_box']

        return Dataset({'required_items': required_items, 'shuffle': False}, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        return utils.data_transforms.Compose([{
            'callback': 'NormalizeObjectPose',
            'parameters': {
                'input_keys': {
                    'ptcloud': 'partial_cloud',
                    'bbox': 'bounding_box'
                }
            },
            'objects': ['partial_cloud', 'bounding_box']
        }, {
            'callback': 'RandomSamplePoints',
            'parameters': {
                'n_points': cfg.CONST.N_INPUT_POINTS
            },
            'objects': ['partial_cloud']
        }, {
            'callback': 'ToTensor',
            'objects': ['partial_cloud', 'bounding_box']
        }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                file_list.append({
                    'taxonomy_id': dc['taxonomy_id'],
                    'model_id': s,
                    'partial_cloud_path': cfg.DATASETS.KITTI.PARTIAL_POINTS_PATH % s,
                    'bounding_box_path': cfg.DATASETS.KITTI.BOUNDING_BOX_FILE_PATH % s,
                })

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list


# //////////////////////////////////////////// = Dataset Loader Mapping = //////////////////////////////////////////// #

DATASET_LOADER_MAPPING = {
    'Completion3D': Completion3DDataLoader,
    'ShapeNet': ShapeNetDataLoader,
    'ShapeNetCars': ShapeNetCarsDataLoader,
    'KITTI': KittiDataLoader,
    'SMLM': SMLMDataLoader
}  # yapf: disable

if __name__ == "__main__":
    from config import cfg
    from data_loaders import DATASET_LOADER_MAPPING
    dataset_loader = DATASET_LOADER_MAPPING['ShapeNet'](cfg)
    dataset = dataset_loader.get_dataset(DatasetSubset.TRAIN)
    print(len(dataset))

if __name__ == "__main__":
    from config import cfg
    import utils.helpers
    import matplotlib.pyplot as plt
    smlm_loader = SMLMDataLoader(cfg, "train")
    data = iter(smlm_loader).__next__()
    print(data['partial_cloud'].shape)
    print(data['gtcloud'].shape)
    gt_ptcloud = data['gtcloud'].squeeze().cpu().numpy()
    gt_ptcloud_img = utils.helpers.get_ptcloud_img(gt_ptcloud/cfg.DATASETS.SMLM.scale)
    plt.imsave("/repos/GRNet2/output/gt_ptcloud_img.png", gt_ptcloud_img)
