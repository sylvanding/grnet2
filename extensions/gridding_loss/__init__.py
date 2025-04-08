# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-12-30 09:56:06
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-02-22 19:19:43
# @Email:  cshzxie@gmail.com

import torch
import numpy as np # Import numpy

import gridding_distance


class GriddingDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scales, ptcloud1, ptcloud2):
        # scales: tuple (Dx, Dy, Dz) defining the grid resolution
        # ptcloud1, ptcloud2: assumed to be in range [-1, 1]
        scale_x, scale_y, scale_z = scales

        # Define grid boundaries based on scales, centered around 0
        # E.g., for scale_x=128, range is [-64, 63]
        min_x = -float(scale_x // 2)
        max_x = float(scale_x // 2 - 1) if scale_x > 0 else min_x
        min_y = -float(scale_y // 2)
        max_y = float(scale_y // 2 - 1) if scale_y > 0 else min_y
        min_z = -float(scale_z // 2)
        max_z = float(scale_z // 2 - 1) if scale_z > 0 else min_z

        # Scale point clouds from [-1, 1] to grid coordinates [min, max]
        # Formula: scaled = pt * (scale/2)
        scale_factors = torch.tensor([scale_x / 2.0, scale_y / 2.0, scale_z / 2.0],
                                      device=ptcloud1.device, dtype=ptcloud1.dtype)
        scaled_ptcloud1 = ptcloud1 * scale_factors.unsqueeze(0).unsqueeze(0)
        scaled_ptcloud2 = ptcloud2 * scale_factors.unsqueeze(0).unsqueeze(0)

        # Clamp to grid boundaries? Might be necessary if points can slightly exceed [-1, 1]
        # scaled_ptcloud1 = torch.max(torch.min(scaled_ptcloud1, scale_factors - 1e-6), -scale_factors + 1e-6)
        # scaled_ptcloud2 = torch.max(torch.min(scaled_ptcloud2, scale_factors - 1e-6), -scale_factors + 1e-6)

        # Call CUDA forward for both point clouds
        grid1, grid1_pt_weights, grid1_pt_indexes = gridding_distance.forward(
            min_x, max_x, min_y, max_y, min_z, max_z, scaled_ptcloud1)
        # grid1 shape should be (B, Dx*Dy*Dz)

        grid2, grid2_pt_weights, grid2_pt_indexes = gridding_distance.forward(
            min_x, max_x, min_y, max_y, min_z, max_z, scaled_ptcloud2)
        # grid2 shape should be (B, Dx*Dy*Dz)

        ctx.save_for_backward(grid1_pt_weights, grid1_pt_indexes, grid2_pt_weights, grid2_pt_indexes,
                              scale_factors.unsqueeze(0).unsqueeze(0)) # Save scale_factors (B, N, 3 shape for broadcasting)
        return grid1, grid2 # Return corrected shape

    @staticmethod
    def backward(ctx, grad_grid1, grad_grid2):
        # grad_grid1, grad_grid2 shape: (B, Dx*Dy*Dz)
        grid1_pt_weights, grid1_pt_indexes, grid2_pt_weights, grid2_pt_indexes, scale_factors = ctx.saved_tensors

        grad_grid1 = grad_grid1.contiguous()
        grad_grid2 = grad_grid2.contiguous()

        grad_scaled_ptcloud1 = gridding_distance.backward(grid1_pt_weights, grid1_pt_indexes, grad_grid1)
        grad_scaled_ptcloud2 = gridding_distance.backward(grid2_pt_weights, grid2_pt_indexes, grad_grid2)

        # Apply chain rule: dLoss/dpt = dLoss/dscaled_pt * dscaled_pt/dpt = grad_scaled_pt * scale_factor
        grad_ptcloud1 = grad_scaled_ptcloud1 * scale_factors
        grad_ptcloud2 = grad_scaled_ptcloud2 * scale_factors

        return None, grad_ptcloud1, grad_ptcloud2


class GriddingDistance(torch.nn.Module):
    def __init__(self, scales=(64, 64, 64)): # Accept scales tuple
        super(GriddingDistance, self).__init__()
        if not isinstance(scales, tuple) or len(scales) != 3:
             raise ValueError("scales must be a tuple of 3 integers (Dx, Dy, Dz)")
        self.scales = scales

    def forward(self, pred_cloud, gt_cloud):
        '''
        pred_cloud(b, n_pts1, 3) in [-1, 1]
        gt_cloud(b, n_pts2, 3) in [-1, 1]
        '''
        # The scaling and boundary calculation is now handled inside GriddingDistanceFunction
        # We just need to pass the scales and original point clouds
        # However, the original code had a split/filter loop. Let's keep that structure.

        _pred_clouds = torch.split(pred_cloud, 1, dim=0)
        _gt_clouds = torch.split(gt_cloud, 1, dim=0)
        pred_grids = []
        gt_grids = []
        for pc, gc in zip(_pred_clouds, _gt_clouds):
            # Optional: Filter zero points (might affect gradients if points become zero during training)
            # Consider if filtering is still necessary or desired.
            # non_zeros_pc = torch.sum(pc.abs(), dim=2).ne(0)
            # if not non_zeros_pc.any(): continue # Skip if empty
            # pc = pc[non_zeros_pc].unsqueeze(dim=0)

            # non_zeros_gc = torch.sum(gc.abs(), dim=2).ne(0)
            # if not non_zeros_gc.any(): continue # Skip if empty
            # gc = gc[non_zeros_gc].unsqueeze(dim=0)

            # Check if clouds are empty after potential filtering
            if pc.shape[1] == 0 or gc.shape[1] == 0:
                 # Handle empty case: return zero grids of correct shape?
                 n_verts = np.prod(self.scales)
                 zero_grid = torch.zeros(1, n_verts, device=pc.device, dtype=pc.dtype)
                 pred_grids.append(zero_grid)
                 gt_grids.append(zero_grid)
                 continue

            # Apply the function with scales and original clouds
            pred_grid, gt_grid = GriddingDistanceFunction.apply(self.scales, pc, gc)
            pred_grids.append(pred_grid)
            gt_grids.append(gt_grid)

        if not pred_grids: # Handle case where all inputs were empty
             n_verts = np.prod(self.scales)
             return torch.zeros(0, n_verts, device=pred_cloud.device, dtype=pred_cloud.dtype), \
                    torch.zeros(0, n_verts, device=gt_cloud.device, dtype=gt_cloud.dtype)


        return torch.cat(pred_grids, dim=0).contiguous(), torch.cat(gt_grids, dim=0).contiguous()


class GriddingLoss(torch.nn.Module):
    # scales: list of tuples, e.g., [(128, 128, 16), (64, 64, 8)]
    # alphas: list of floats, weights for each scale
    def __init__(self, scales=[(64, 64, 64)], alphas=[1.0]):
        super(GriddingLoss, self).__init__()
        if len(scales) != len(alphas):
            raise ValueError("Number of scales and alphas must match.")
        self.scales = scales
        self.alphas = alphas
        # Instantiate GriddingDistance with the scale tuples
        self.gridding_dists = [GriddingDistance(scales=s) for s in scales]
        self.l1_loss = torch.nn.L1Loss(reduction='mean') # Use mean reduction

    def forward(self, pred_cloud, gt_cloud):
        # Input clouds assumed in [-1, 1]
        total_gridding_loss = 0.0 # Initialize as float
        n_losses = len(self.scales)

        for i in range(n_losses):
            alpha = self.alphas[i]
            gdist = self.gridding_dists[i]
            pred_grid, gt_grid = gdist(pred_cloud, gt_cloud) # Get grids for this scale

            # Compute L1 loss between the grids for this scale
            loss_at_scale = self.l1_loss(pred_grid, gt_grid)
            total_gridding_loss += alpha * loss_at_scale

        # Return average loss across scales? Or weighted sum? Current code does weighted sum.
        return total_gridding_loss
