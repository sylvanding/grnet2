# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-09-06 11:35:30
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 19:20:36
# @Email:  cshzxie@gmail.com

import torch
import numpy as np # Import numpy for potential shape calculations
import sys

sys.path.append("/repos/grnet2")

from extensions.gridding import Gridding, GriddingReverse
from extensions.cubic_feature_sampling import CubicFeatureSampling


class RandomPointSampling(torch.nn.Module):
    def __init__(self, n_points):
        super(RandomPointSampling, self).__init__()
        self.n_points = n_points

    def forward(self, pred_cloud, partial_cloud=None):
        # if partial_cloud is not None:
        #     pred_cloud = torch.cat([partial_cloud, pred_cloud], dim=1)

        _ptcloud = torch.split(pred_cloud, 1, dim=0)
        ptclouds = []
        for p in _ptcloud:
            non_zeros = torch.sum(p, dim=2).ne(0)
            p = p[non_zeros].unsqueeze(dim=0)
            n_pts = p.size(1)
            print("n_pts:", n_pts)

            if n_pts < self.n_points: # copy points
                rnd_idx = torch.cat([torch.randint(0, n_pts, (self.n_points, ))])
            else: # remove points
                rnd_idx = torch.randperm(p.size(1))[:self.n_points]
            ptclouds.append(p[:, rnd_idx, :])

        return torch.cat(ptclouds, dim=0).contiguous()


class GRNet(torch.nn.Module):
    def __init__(self, cfg=None):
        super(GRNet, self).__init__()
        self.gridding_scales = (256, 256, 32)
        self.gridding = Gridding(scales=self.gridding_scales)

        # --- Encoder ---
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=3, padding=1), # Use kernel 3, pad 1 to maintain size before pooling
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(0.2),
            # Pool only X, Y: (256, 256, 32) -> (128, 128, 16)
            torch.nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(0.2),
            # Pool only X, Y: (128, 128, 16) -> (64, 64, 8)
            torch.nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(0.2),
            # Pool only X, Y: (64, 64, 8) -> (32, 32, 4)
            torch.nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(0.2),
            # Pool X, Y, Z: (32, 32, 4) -> (16, 16, 2)
            torch.nn.MaxPool3d(kernel_size=2, stride=2)
        )

        # --- Bottleneck ---
        # Flattened size: 256 * 16 * 16 * 2 = 131072
        self.fc5_in_features = 256 * 16 * 16 * 2
        self.fc5 = torch.nn.Sequential(
            torch.nn.Linear(self.fc5_in_features, self.fc5_in_features//16),
            torch.nn.SiLU()
        )
        self.fc6 = torch.nn.Sequential(
            torch.nn.Linear(self.fc5_in_features//16, self.fc5_in_features),
            torch.nn.SiLU()
        )
        # Shape to reshape back to before dconv7
        self.fc6_reshape_shape = (-1, 256, 16, 16, 2) # Adjusted size

        # --- Decoder ---
        self.dconv7 = torch.nn.Sequential(
            # Upsample (16, 16, 2) -> (32, 32, 4)
            torch.nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(0.2)
        )
        self.dconv8 = torch.nn.Sequential(
            # Upsample (32, 32, 4) -> (64, 64, 8)
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=False, padding=1), # Adjust kernel/stride/pad for Z=1
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(0.2)
        )
        self.dconv9 = torch.nn.Sequential(
            # Upsample (64, 64, 8) -> (128, 128, 16)
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(0.2)
        )
        self.dconv10 = torch.nn.Sequential(
            # Upsample (128, 128, 16) -> (256, 256, 32)
            torch.nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(1),
            torch.nn.LeakyReLU(0.2)
        )

        # --- Final Layers ---
        self.gridding_rev = GriddingReverse(scales=self.gridding_scales)
        self.point_sampling = RandomPointSampling(n_points=2048)
        self.feature_sampling = CubicFeatureSampling(neighborhood_size=1) # ns=1 (8 corners)
        # Sampled feature sizes remain the same (C * 8)
        feat_size_16 = 256 * 8 # 2048
        feat_size_32 = 128 * 8 # 1024
        feat_size_64 = 64 * 8 # 512
        feat_size_128 = 32 * 8 # 256
        feat_size_256 = 1 * 8 # 8
        concat_feat_size = feat_size_256 + feat_size_128 + feat_size_64 + feat_size_32 + feat_size_16 # 3848

        self.fc11 = torch.nn.Sequential(
            torch.nn.Linear(concat_feat_size, 1924),
            torch.nn.SiLU()
        )
        self.fc12 = torch.nn.Sequential(
            torch.nn.Linear(1924, 962),
            torch.nn.SiLU()
        )
        self.fc13 = torch.nn.Sequential(
            torch.nn.Linear(962, 481),
            torch.nn.SiLU()
        )
        self.fc14 = torch.nn.Sequential(
            torch.nn.Linear(481, 112),
            torch.nn.SiLU()
        )
        # Output 24 values per point (8 offsets * 3 coords)
        self.fc15 = torch.nn.Linear(112, 24)
        self.n_dense_points = 16384 # 2048 * 8

    def forward(self, data):
        # Allow for dict or tensor input for flexibility
        if isinstance(data, dict):
            partial_cloud = data['partial_cloud'] # Expected shape (B, N_partial, 3) in [-1, 1]
        else:
            partial_cloud = data # Assume input is the partial cloud tensor

        # print(partial_cloud.size())     # torch.Size([batch_size, N_partial, 3])
        # Gridding output: (B, Dx*Dy*Dz), reshape to (B, 1, Dx, Dy, Dz)
        pt_features_xyz_l = self.gridding(partial_cloud).view(-1, 1, *self.gridding_scales)
        # print("Initial Grid:", pt_features_xyz_l.shape) # [B, 1, 256, 256, 32]

        # --- Encoder ---
        pt_features_128 = self.conv1(pt_features_xyz_l)
        # print("After conv1:", pt_features_128.shape)  # [B, 32, 128, 128, 16]
        pt_features_64 = self.conv2(pt_features_128)
        # print("After conv2:", pt_features_64.shape)  # [B, 64, 64, 64, 8]
        pt_features_32 = self.conv3(pt_features_64)
        # print("After conv3:", pt_features_32.shape)  # [B, 128, 32, 32, 4]
        pt_features_16 = self.conv4(pt_features_32)
        # print("After conv4 (bottleneck input):", pt_features_16.shape)   # [B, 256, 16, 16, 2]

        # --- Bottleneck ---
        features = self.fc5(pt_features_16.view(pt_features_16.size(0), -1))
        # print("Bottleneck features:", features.shape) # [B, 2048]
        pt_features_16_r = self.fc6(features).view(*self.fc6_reshape_shape) + pt_features_16
        # print("After fc6 + skip:", pt_features_16_r.shape) # [B, 256, 16, 16, 2]

        # --- Decoder ---
        pt_features_32_r = self.dconv7(pt_features_16_r) + pt_features_32 # Input skip from conv3 output
        # print("After dconv7 + skip:", pt_features_32_r.shape) # [B, 128, 32, 32, 4]
        pt_features_64_r = self.dconv8(pt_features_32_r) + pt_features_64 # Input skip from conv2 output
        # print("After dconv8 + skip:", pt_features_64_r.shape) # [B, 64, 64, 64, 8]
        pt_features_128_r = self.dconv9(pt_features_64_r) + pt_features_128 # Input skip from conv1 output
        # print("After dconv9 + skip:", pt_features_128_r.shape) # [B, 32, 128, 128, 16]
        pt_features_xyz_r = self.dconv10(pt_features_128_r) + pt_features_xyz_l # Input skip from initial grid
        # print("Final grid features:", pt_features_xyz_r.shape) # [B, 1, 256, 256, 32]

        # --- Reverse Gridding and Sampling ---
        flat_grid = pt_features_xyz_r.view(pt_features_xyz_r.size(0), -1)
        sparse_cloud = self.gridding_rev(flat_grid)
        # print(sparse_cloud.size())      # torch.Size([B, 256*256*32, 3]) output range approx [-1, 1]

        # Sample points from the sparse cloud generated by GriddingReverse
        # Combine with partial cloud before sampling if provided
        sampled_sparse_cloud = self.point_sampling(sparse_cloud, partial_cloud)
        # print(sampled_sparse_cloud.size()) # torch.Size([B, 2048, 3]) output range approx [-1, 1]

        # --- Feature Sampling and Final FC Layers ---
        # Use features from decoder path with correct shapes
        point_features_16 = self.feature_sampling(sampled_sparse_cloud, pt_features_16_r).flatten(start_dim=2)
        # print("Sampled features 16:", point_features_16.shape) # [B, 2048, 2048]
        point_features_32 = self.feature_sampling(sampled_sparse_cloud, pt_features_32_r).flatten(start_dim=2)
        # print("Sampled features 32:", point_features_32.shape) # [B, 2048, 1024]
        point_features_64 = self.feature_sampling(sampled_sparse_cloud, pt_features_64_r).flatten(start_dim=2)
        # print("Sampled features 64:", point_features_64.shape)  # [B, 2048, 512]
        point_features_128 = self.feature_sampling(sampled_sparse_cloud, pt_features_128_r).flatten(start_dim=2)
        # print("Sampled features 128:", point_features_128.shape)  # [B, 2048, 256]
        point_features_256 = self.feature_sampling(sampled_sparse_cloud, pt_features_xyz_r).flatten(start_dim=2)
        # print("Sampled features 256:", point_features_256.shape)  # [B, 2048, 8]
        
        point_features = torch.cat([point_features_16, point_features_32, point_features_64, point_features_128, point_features_256], dim=2)
        # print("Concatenated point features:", point_features.shape) # [B, 2048, 3848]

        point_features = self.fc11(point_features)
        # print(point_features.size())    # torch.Size([B, 2048, 1924])
        point_features = self.fc12(point_features)
        # print(point_features.size())    # torch.Size([B, 2048, 962])
        point_features = self.fc13(point_features)
        # print(point_features.size())    # torch.Size([B, 2048, 481])
        point_features = self.fc14(point_features)
        # print(point_features.size())    # torch.Size([B, 2048, 112])

        # Calculate point offsets (8 offsets per sampled point)
        point_offset = self.fc15(point_features).view(-1, self.n_dense_points, 3) # Reshape (B, 2048, 24) -> (B, 16384, 3)
        # print(point_offset.size())    # torch.Size([B, 16384, 3])

        # Create dense cloud by repeating sampled points and adding offsets
        dense_cloud = sampled_sparse_cloud.unsqueeze(dim=2).repeat(1, 1, 8, 1).view(-1, self.n_dense_points, 3) + point_offset
        # print(dense_cloud.size())       # torch.Size([B, 16384, 3])

        # Return sparse cloud (sampled) and dense cloud
        return sampled_sparse_cloud, dense_cloud

if __name__ == "__main__":
    from torch.optim import Adam
    grnet = GRNet()
    grnet.to("cuda")
    grnet.train()
    optimizer = Adam(grnet.parameters(), lr=0.001)
    partial_cloud = ((torch.rand(2, 2048, 3)-0.5)*2*0.9).to("cuda")
    gt_cloud = ((torch.rand(2, 16384, 3)-0.5)*2*0.9).to("cuda")

    sampled_sparse_cloud, dense_cloud = grnet(partial_cloud)
    print(sampled_sparse_cloud.size())
    print(dense_cloud.size())

    from extensions.gridding_loss import GriddingLoss
    from extensions.chamfer_dist import ChamferDistance
    gridding_loss = GriddingLoss(    # lgtm [py/unused-local-variable]
        scales=[(128, 128, 16)],
        alphas=[0.1])
    chamfer_dist = ChamferDistance()
    
    loss = 0.1 * gridding_loss(sampled_sparse_cloud, dense_cloud) + 0.9 * chamfer_dist(sampled_sparse_cloud, gt_cloud)
    print(loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Training step finished.')
    