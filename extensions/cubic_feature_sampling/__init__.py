# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-12-19 16:55:15
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-26 13:15:14
# @Email:  cshzxie@gmail.com

import torch
import numpy as np # Import numpy

import cubic_feature_sampling


class CubicFeatureSamplingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ptcloud, cubic_features, neighborhood_size=1):
        # Infer scales from cubic_features shape (B, C, Dx, Dy, Dz)
        if cubic_features.dim() != 5:
            raise ValueError("cubic_features must be 5D (B, C, Dx, Dy, Dz)")
        scales = tuple(cubic_features.shape[2:]) # (Dx, Dy, Dz)
        scale_x, scale_y, scale_z = scales

        # Ensure ptcloud coordinates are scaled appropriately (e.g., [0, Dx-1], [0, Dy-1], [0, Dz-1])
        # This scaling should happen *before* calling this function, typically in the nn.Module wrapper
        # Assuming ptcloud is already scaled appropriately here.

        point_features, grid_pt_indexes = cubic_feature_sampling.forward(
            scale_x, scale_y, scale_z, neighborhood_size, ptcloud, cubic_features)

        # Save scales and neighborhood_size as tensors for backward
        ctx.save_for_backward(torch.tensor(scales, dtype=torch.int),
                              torch.tensor([neighborhood_size], dtype=torch.int),
                              grid_pt_indexes)
        return point_features

    @staticmethod
    def backward(ctx, grad_point_features):
        scales_tensor, neighborhood_size_tensor, grid_pt_indexes = ctx.saved_tensors
        scales = tuple(scales_tensor.tolist()) # Convert tensor back to tuple (Dx, Dy, Dz)
        scale_x, scale_y, scale_z = scales
        neighborhood_size = neighborhood_size_tensor.item()

        grad_point_features = grad_point_features.contiguous() # Ensure contiguous
        grad_ptcloud, grad_cubic_features = cubic_feature_sampling.backward(
            scale_x, scale_y, scale_z, neighborhood_size,
            grad_point_features, grid_pt_indexes)

        # Gradient for ptcloud is returned (although likely zero), gradient for cubic_features, None for neighborhood_size
        return grad_ptcloud, grad_cubic_features, None


class CubicFeatureSampling(torch.nn.Module):
    def __init__(self, neighborhood_size=1):
        super(CubicFeatureSampling, self).__init__()
        # neighborhood_size=1 corresponds to sampling 8 corners in the current kernel interpretation
        if neighborhood_size != 1:
             print("Warning: Current CubicFeatureSampling CUDA kernel likely only supports neighborhood_size=1 (8 corner sampling).")
        self.neighborhood_size = neighborhood_size


    def forward(self, ptcloud, cubic_features):
        # ptcloud expected in range [-1, 1]^3
        # cubic_features expected shape (B, C, Dx, Dy, Dz)
        if cubic_features.dim() != 5:
            raise ValueError("cubic_features must be 5D (B, C, Dx, Dy, Dz)")
        if ptcloud.dim() != 3:
             raise ValueError("ptcloud must be 3D (B, N, 3)")

        # Get grid dimensions (scales) from features
        scales = np.array(cubic_features.shape[2:]) # [Dx, Dy, Dz]

        # Calculate scaling factors to map ptcloud from [-1, 1] to [0, D-1] range for each dimension
        # Formula: scaled_coord = (pt_coord + 1) / 2 * (D - 1)
        # More robust: scaled_coord = (pt_coord + 1) * 0.5 * (D - 1) avoids potential issues if D=1
        scaling_factors = torch.from_numpy((scales - 1) * 0.5).to(ptcloud.device).float()
        biases = scaling_factors # Bias is also (D-1)*0.5
        
        # Scale ptcloud: B, N, 3 -> B, N, 3
        scaled_ptcloud = (ptcloud + 1.0) * scaling_factors.unsqueeze(0).unsqueeze(0) # Apply scaling and bias implicitly

        # Clamp coordinates slightly to avoid floating point issues at boundaries? Optional.
        # scaled_ptcloud = torch.clamp(scaled_ptcloud, 0, scales - 1 - 1e-6)

        # Apply the autograd function
        return CubicFeatureSamplingFunction.apply(scaled_ptcloud, cubic_features, self.neighborhood_size)
