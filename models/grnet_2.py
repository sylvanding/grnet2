import sys

sys.path.append("/repos/GRNet2")

import torch
import torch.nn as nn
import numpy as np  # Import numpy for potential shape calculations

from utils.midpoint_interpolate import midpoint_interpolate

from extensions.gridding import Gridding, GriddingReverse
from extensions.cubic_feature_sampling import CubicFeatureSampling

from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


class AttentionGate3D(nn.Module):
    def __init__(self, F_g, F_x, F_int):
        super(AttentionGate3D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_x, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.leaky_relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, g, x):
        """
        g: Gating signal from the previous layer (decoder upsampled)
        x: Skip connection from the encoder
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi_input = self.leaky_relu(g1 + x1)
        alpha = self.psi(psi_input)
        return x * alpha


class PointNet2FeatureExtractor(nn.Module):
    def __init__(self):
        super(PointNet2FeatureExtractor, self).__init__()
        # Encoder (Feature Extraction)
        # Use PointNetSetAbstraction from models.pointnet2_utils which uses CUDA ops

        # Input points are l0_points (B, 3, N)
        self.sa1 = PointNetSetAbstraction(npoint=2048 // 2, radius=0.05, nsample=16, in_channel=3, mlp=[32, 32, 64], group_all=False)
        # Input points are l1_points (B, 64, N/4)
        self.sa2 = PointNetSetAbstraction(npoint=2048 // 4, radius=0.1, nsample=16, in_channel=64, mlp=[64, 64, 128], group_all=False)
        # Input points are l2_points (B, 128, N/16)
        self.sa3 = PointNetSetAbstraction(npoint=2048 // 8, radius=0.2, nsample=16, in_channel=128, mlp=[128, 128, 256], group_all=False)
        # Input points are l3_points (B, 256, N/32)
        self.sa4 = PointNetSetAbstraction(npoint=2048 // 16, radius=0.4, nsample=16, in_channel=256, mlp=[256, 256, 512], group_all=False)

        # Decoder (For Reconstruction)
        # Input channels based on concatenation of skip connection and upsampled features
        self.fp4 = PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256, 256]) # 512 from sa4, 256 from sa3
        self.fp3 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 256]) # 256 from fp4, 128 from sa2
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 64, mlp=[256, 128])  # 256 from fp3, 64 from sa1
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + 3, mlp=[128, 128])   # 128 from fp2, 3 from l0_points (initial xyz)

        # self.final_conv = nn.Conv1d(128, 3, 1)

    def forward(self, xyz):
        # xyz shape: [B, 3, N]
        l0_xyz = xyz
        l0_points = xyz # Use original xyz as features for the first layer

        # Feature Extraction
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) # l1_points: [B, 64, N/4]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # l2_points: [B, 128, N/16]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # l3_points: [B, 256, N/32]
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points) # l4_points: [B, 512, N/64]

        # Feature Propagation (Decoding)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points) # Output: [B, 256, N/32]
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) # Output: [B, 256, N/16]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) # Output: [B, 128, N/4]
        # For fp1, points1 (features at l0_xyz) are the original coordinates
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points) # Output: [B, 128, N]

        # l0_points = self.final_conv(l0_points)  # Output: [B, 3, N]

        return l0_points


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

            if n_pts < self.n_points:  # copy points
                rnd_idx = torch.cat([torch.randint(0, n_pts, (self.n_points,))])
            else:  # remove points
                rnd_idx = torch.randperm(p.size(1))[: self.n_points]
            ptclouds.append(p[:, rnd_idx, :])

        return torch.cat(ptclouds, dim=0).contiguous()


class GRNet_2(torch.nn.Module):
    def __init__(self, cfg=None):
        super(GRNet_2, self).__init__()
        self.gridding_scales = (128, 128, 32)
        self.gridding = Gridding(scales=self.gridding_scales)

        self.pointnet2_feature_extractor = PointNet2FeatureExtractor()

        # --- Encoder ---
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(
                1, 32, kernel_size=3, padding=1
            ),  # Use kernel 3, pad 1 to maintain size before pooling
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(0.01),
            # (128, 128, 32) -> (64, 64, 16)
            torch.nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(0.01),
            # (64, 64, 16) -> (32, 32, 8)
            torch.nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(0.01),
            # (32, 32, 8) -> (16, 16, 4)
            torch.nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(0.01),
            # (16, 16, 4) -> (8, 8, 2)
            torch.nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # --- Bottleneck ---
        # Flattened size: 256 * 8 * 8 * 2 = 32768
        self.fc5_in_features = 256 * 8 * 8 * 2
        self.fc5 = torch.nn.Sequential(
            torch.nn.Linear(self.fc5_in_features, self.fc5_in_features // 16),
            torch.nn.SiLU(),
        )
        self.fc6 = torch.nn.Sequential(
            torch.nn.Linear(self.fc5_in_features // 16, self.fc5_in_features),
            torch.nn.SiLU(),
        )
        # Shape to reshape back to before dconv7
        self.fc6_reshape_shape = (-1, 256, 8, 8, 2)  # Adjusted size

        # --- Decoder ---
        self.dconv7 = torch.nn.Sequential(
            # Upsample (8, 8, 2) -> (16, 16, 4)
            torch.nn.ConvTranspose3d(
                256, 128, kernel_size=4, stride=2, bias=False, padding=1
            ),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(0.01),
        )
        self.dconv8 = torch.nn.Sequential(
            # Upsample (16, 16, 4) -> (32, 32, 8)
            torch.nn.ConvTranspose3d(
                128, 64, kernel_size=4, stride=2, bias=False, padding=1
            ),  # Adjust kernel/stride/pad for Z=1
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(0.01),
        )
        self.dconv9 = torch.nn.Sequential(
            # Upsample (32, 32, 8) -> (64, 64, 16)
            torch.nn.ConvTranspose3d(
                64, 32, kernel_size=4, stride=2, bias=False, padding=1
            ),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(0.01),
        )
        self.dconv10 = torch.nn.Sequential(
            # Upsample (64, 64, 16) -> (128, 128, 32)
            torch.nn.ConvTranspose3d(
                32, 1, kernel_size=4, stride=2, bias=False, padding=1
            ),
            torch.nn.BatchNorm3d(1),
            torch.nn.LeakyReLU(0.01),
        )

        # Attention Gates
        # F_g: channels of gating signal (from decoder), F_x: channels of skip connection (from encoder)
        # F_int is typically F_x // 2 or F_g // 2
        self.att7 = AttentionGate3D(F_g=128, F_x=128, F_int=64)  # For dconv7 output and pt_features_16
        self.att8 = AttentionGate3D(F_g=64, F_x=64, F_int=32)    # For dconv8 output and pt_features_32
        self.att9 = AttentionGate3D(F_g=32, F_x=32, F_int=16)    # For dconv9 output and pt_features_64
        self.att10 = AttentionGate3D(F_g=1, F_x=1, F_int=1)     # For dconv10 output and pt_features_xyz_l

        # --- Final Layers ---
        # self.gridding_rev = GriddingReverse(scales=self.gridding_scales)
        # self.point_sampling = RandomPointSampling(n_points=2048)
        self.feature_sampling = CubicFeatureSampling(
            neighborhood_size=1
        )  # ns=1 (8 corners)
        # Sampled feature sizes remain the same (C * 8)
        feat_size_8 = 256 * 8  # 2048
        feat_size_16 = 128 * 8  # 1024
        feat_size_32 = 64 * 8  # 512
        feat_size_64 = 32 * 8  # 256
        feat_size_128 = 1 * 8  # 8
        concat_feat_size = (
            feat_size_128 + feat_size_64 + feat_size_32 + feat_size_16 + feat_size_8
        )  # 3848

        self.fc11 = torch.nn.Sequential(
            torch.nn.Linear(concat_feat_size + 128, 962), torch.nn.SiLU()
        )
        self.fc12 = torch.nn.Sequential(torch.nn.Linear(962, 112), torch.nn.SiLU())
        # Output 24 values per point (8 offsets * 3 coords)
        self.fc13 = torch.nn.Linear(112, 24)
        
        self.cls_fc11 = torch.nn.Sequential(
            torch.nn.Linear(concat_feat_size + 128, 481), torch.nn.SiLU()
        )
        self.cls_fc12 = torch.nn.Sequential(torch.nn.Linear(481, 56), torch.nn.SiLU())
        self.cls_fc13 = torch.nn.Linear(56, 2)
        
        self.n_dense_points = 16384  # 2048 * 8

    def forward(self, data):
        # Allow for dict or tensor input for flexibility
        if isinstance(data, dict):
            partial_cloud = data[
                "partial_cloud"
            ]  # Expected shape (B, N_partial, 3) in [-1, 1]
        else:
            partial_cloud = data  # Assume input is the partial cloud tensor

        # dense_cloud_interp = midpoint_interpolate(
        #     partial_cloud.permute(0, 2, 1), up_rate=8
        # ).permute(0, 2, 1)
        
        # dense_cloud_interp = partial_cloud.unsqueeze(dim=2).repeat(1, 1, 8, 1).view(-1, self.n_dense_points, 3).contiguous()

        partial_cloud_features = (
            self.pointnet2_feature_extractor(partial_cloud.permute(0, 2, 1).contiguous())
            .permute(0, 2, 1)
            .contiguous()
        ) # [B, 2048, 128]

        # print(partial_cloud.size())     # torch.Size([batch_size, N_partial, 3])
        # Gridding output: (B, Dx*Dy*Dz), reshape to (B, 1, Dx, Dy, Dz)
        pt_features_xyz_l = self.gridding(partial_cloud).view(
            -1, 1, *self.gridding_scales
        )
        # print("Initial Grid:", pt_features_xyz_l.shape) # [B, 1, 128, 128, 32]

        # --- Encoder ---
        pt_features_64 = self.conv1(pt_features_xyz_l)
        # print("After conv1:", pt_features_64.shape)  # [B, 32, 64, 64, 16]
        pt_features_32 = self.conv2(pt_features_64)
        # print("After conv2:", pt_features_32.shape)  # [B, 64, 32, 32, 8]
        pt_features_16 = self.conv3(pt_features_32)
        # print("After conv3:", pt_features_16.shape)  # [B, 128, 16, 16, 4]
        pt_features_8 = self.conv4(pt_features_16)
        # print("After conv4 (bottleneck input):", pt_features_8.shape)   # [B, 256, 8, 8, 2]

        # --- Bottleneck ---
        features = self.fc5(pt_features_8.view(pt_features_8.size(0), -1))
        # print("Bottleneck features:", features.shape) # [B, 2048]
        pt_features_8_r = (
            self.fc6(features).view(*self.fc6_reshape_shape) + pt_features_8
        )
        # print("After fc6 + skip:", pt_features_8_r.shape) # [B, 256, 8, 8, 2]

        # --- Decoder ---
        d7_out = self.dconv7(pt_features_8_r)
        pt_features_16_att = self.att7(g=d7_out, x=pt_features_16)
        pt_features_16_r = d7_out + pt_features_16_att
        # print("After dconv7 + skip:", pt_features_16_r.shape) # [B, 128, 16, 16, 4]

        d8_out = self.dconv8(pt_features_16_r)
        pt_features_32_att = self.att8(g=d8_out, x=pt_features_32)
        pt_features_32_r = d8_out + pt_features_32_att
        # print("After dconv8 + skip:", pt_features_32_r.shape) # [B, 64, 32, 32, 8]

        d9_out = self.dconv9(pt_features_32_r)
        pt_features_64_att = self.att9(g=d9_out, x=pt_features_64)
        pt_features_64_r = d9_out + pt_features_64_att
        # print("After dconv9 + skip:", pt_features_64_r.shape) # [B, 32, 64, 64, 16]

        d10_out = self.dconv10(pt_features_64_r)
        pt_features_xyz_l_att = self.att10(g=d10_out, x=pt_features_xyz_l)
        pt_features_xyz_r = d10_out + pt_features_xyz_l_att
        # print("Final grid features:", pt_features_xyz_r.shape) # [B, 1, 128, 128, 32]

        # # --- Reverse Gridding and Sampling ---
        # flat_grid = pt_features_xyz_r.view(pt_features_xyz_r.size(0), -1)
        # sparse_cloud = self.gridding_rev(flat_grid)
        # # print(sparse_cloud.size())      # torch.Size([B, 256*256*32, 3]) output range approx [-1, 1]

        # # Sample points from the sparse cloud generated by GriddingReverse
        # # Combine with partial cloud before sampling if provided
        # sampled_sparse_cloud = self.point_sampling(sparse_cloud, partial_cloud)
        # # print(sampled_sparse_cloud.size()) # torch.Size([B, 2048, 3]) output range approx [-1, 1]

        # --- Feature Sampling and Final FC Layers ---
        # Use features from decoder path with correct shapes
        point_features_8 = self.feature_sampling(
            partial_cloud, pt_features_8_r
        ).flatten(start_dim=2)
        # print("Sampled features 8:", point_features_8.shape) # [B, 2048, 2048]
        point_features_16 = self.feature_sampling(
            partial_cloud, pt_features_16_r
        ).flatten(start_dim=2)
        # print("Sampled features 16:", point_features_16.shape) # [B, 2048, 1024]
        point_features_32 = self.feature_sampling(
            partial_cloud, pt_features_32_r
        ).flatten(start_dim=2)
        # print("Sampled features 32:", point_features_32.shape)  # [B, 2048, 512]
        point_features_64 = self.feature_sampling(
            partial_cloud, pt_features_64_r
        ).flatten(start_dim=2)
        # print("Sampled features 64:", point_features_64.shape)  # [B, 2048, 256]
        point_features_128 = self.feature_sampling(
            partial_cloud, pt_features_xyz_r
        ).flatten(start_dim=2)
        # print("Sampled features 128:", point_features_128.shape)  # [B, 2048, 8]

        point_features = torch.cat(
            [
                point_features_8,
                point_features_16,
                point_features_32,
                point_features_64,
                point_features_128,
                partial_cloud_features, # [B, 2048, 128]
            ],
            dim=2,
        )
        # print("Concatenated point features:", point_features.shape) # [B, 2048, 3848]

        cls_features = point_features
        cls_features = self.cls_fc11(cls_features)
        cls_features = self.cls_fc12(cls_features)
        cls_features = self.cls_fc13(cls_features)
        # cls_features.shape = [B, 2048+n, 2]
        
        # 根据cls_features对point_features进行过滤
        # cls_features.shape = [B, 2048+n, 2]，选取不是噪声的概率最高的2048个点
        probs = torch.softmax(cls_features, dim=2)  # 计算softmax概率
        not_noise_probs = probs[:, :, 0]  # 不是噪声的概率
        batch_size = point_features.size(0)
        num_points_to_select = 2048
        
        # 对每个batch中的点按照不是噪声的概率排序，并选择前2048个点
        # 使用一个明确的变量名来存储来自 topk 的原始索引
        _, top_k_point_indices = torch.topk(not_noise_probs, k=num_points_to_select, dim=1, largest=True, sorted=False)
        # top_k_point_indices.shape: [batch_size, num_points_to_select]
        
        # --- 修正 point_features 的 gather 操作 ---
        # 准备 gather 操作所需的 index 张量
        # 其形状应为 [batch_size, num_points_to_select, point_features.size(2)]
        idx_for_pf_gather = top_k_point_indices.unsqueeze(2).expand(-1, -1, point_features.size(2))
        point_features = torch.gather(point_features, 1, idx_for_pf_gather)
        # point_features 现在形状为 [batch_size, num_points_to_select, point_features.size(2)]
        
        # --- 修正 partial_cloud 的 gather 操作 ---
        # 使用相同的 top_k_point_indices 来过滤 partial_cloud
        # 准备 gather 操作所需的 index 张量
        # 其形状应为 [batch_size, num_points_to_select, partial_cloud.size(2)]
        idx_for_pc_gather = top_k_point_indices.unsqueeze(2).expand(-1, -1, partial_cloud.size(2))
        partial_cloud_filtered = torch.gather(partial_cloud, 1, idx_for_pc_gather)
        # partial_cloud_filtered 现在形状为 [batch_size, num_points_to_select, partial_cloud.size(2)]
        
        # 更新dense_cloud_interp以使用过滤后的partial_cloud
        dense_cloud_interp = partial_cloud_filtered.unsqueeze(dim=2).repeat(1, 1, 8, 1).view(-1, self.n_dense_points, 3).contiguous()
        
        point_features = self.fc11(point_features)
        # print(point_features.size())    # torch.Size([B, 2048, 962])
        point_features = self.fc12(point_features)
        # print(point_features.size())    # torch.Size([B, 2048, 112])

        # Calculate point offsets (8 offsets per sampled point)
        point_offset = self.fc13(point_features).view(
            -1, self.n_dense_points, 3
        )  # Reshape (B, 2048, 24) -> (B, 16384, 3)
        # print(point_offset.size())    # torch.Size([B, 16384, 3])

        # Create dense cloud by repeating sampled points and adding offsets
        dense_cloud_pred = dense_cloud_interp + point_offset
        # print(dense_cloud_pred.size())       # torch.Size([B, 16384, 3])

        # dense_cloud_pred = dense_cloud_pred + dense_cloud_interp

        # Return sparse cloud (sampled) and dense cloud
        return dense_cloud_interp, dense_cloud_pred, pt_features_xyz_r, cls_features


if __name__ == "__main__":
    from torch.optim import Adam

    grnet = GRNet_2()
    grnet.to("cuda")
    grnet.train()
    optimizer = Adam(grnet.parameters(), lr=0.001)
    partial_cloud = ((torch.rand(2, 2048, 3) - 0.5) * 2 * 0.9).to("cuda")
    gt_cloud = ((torch.rand(2, 16384, 3) - 0.5) * 2 * 0.9).to("cuda")

    dense_cloud_interp, dense_cloud_pred, pt_features_xyz_r = grnet(partial_cloud)
    print(dense_cloud_interp.size())
    print(dense_cloud_pred.size())
    print(pt_features_xyz_r.size())

    from extensions.gridding_loss import GriddingLoss
    from extensions.chamfer_dist import ChamferDistance

    gridding_loss = GriddingLoss(  # lgtm [py/unused-local-variable]
        scales=[(128, 128, 32)], alphas=[0.1]
    )
    chamfer_dist = ChamferDistance()
    gridding_scales = (128, 128, 32)
    gridding = Gridding(scales=gridding_scales)
    l1_loss = nn.L1Loss()
    
    gridding_gt = gridding(gt_cloud).view(-1, 1, *gridding_scales)
    print(gridding_gt.size())

    loss = 0.1 * gridding_loss(dense_cloud_pred, gt_cloud) + 0.8 * chamfer_dist(
        dense_cloud_pred, gt_cloud
    ) + 0.1 * l1_loss(pt_features_xyz_r, gridding_gt)
    print(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Training step finished.")
