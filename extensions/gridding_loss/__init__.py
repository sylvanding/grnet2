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

        # Define grid boundaries. min_coord_dim is the coordinate of the "lower-left" corner
        # of the cell with index 0. max_coord_dim is the coordinate of the "lower-left"
        # corner of the cell with index (scale_dim - 1).
        # This ensures that the grid length calculated in CUDA,
        # roundf(max_coord_dim - min_coord_dim + 1), equals scale_dim.
        min_coord_x = -float(scale_x // 2)
        max_coord_x = min_coord_x + scale_x - 1.0
        min_coord_y = -float(scale_y // 2)
        max_coord_y = min_coord_y + scale_y - 1.0
        min_coord_z = -float(scale_z // 2)
        max_coord_z = min_coord_z + scale_z - 1.0

        # Scale point clouds from [-1, 1] to grid coordinates
        # Formula: scaled_coord = pt_coord * (scale_dim / 2.0)
        # This results in scaled_coord range of [-scale_dim/2, scale_dim/2]
        scale_factors = torch.tensor([scale_x / 2.0, scale_y / 2.0, scale_z / 2.0],
                                      device=ptcloud1.device, dtype=ptcloud1.dtype)
        scaled_ptcloud1 = ptcloud1 * scale_factors.unsqueeze(0).unsqueeze(0)
        scaled_ptcloud2 = ptcloud2 * scale_factors.unsqueeze(0).unsqueeze(0)

        # Clamp scaled coordinates. For floor(clamped_scaled_pt) to be in [min_coord, max_coord],
        # clamped_scaled_pt must be in [min_coord, max_coord + 1.0 - epsilon).
        epsilon = 1e-5  # Small epsilon for float precision

        clamped_coords_list1 = []
        for dim in range(3):
            min_val = [min_coord_x, min_coord_y, min_coord_z][dim]
            max_val = [max_coord_x, max_coord_y, max_coord_z][dim]
            clamped_coords_list1.append(
                torch.clamp(scaled_ptcloud1[..., dim], min=min_val, max=max_val + 1.0 - epsilon)
            )
        clamped_scaled_ptcloud1 = torch.stack(clamped_coords_list1, dim=-1)

        clamped_coords_list2 = []
        for dim in range(3):
            min_val = [min_coord_x, min_coord_y, min_coord_z][dim]
            max_val = [max_coord_x, max_coord_y, max_coord_z][dim]
            clamped_coords_list2.append(
                torch.clamp(scaled_ptcloud2[..., dim], min=min_val, max=max_val + 1.0 - epsilon)
            )
        clamped_scaled_ptcloud2 = torch.stack(clamped_coords_list2, dim=-1)

        # Call CUDA forward for both point clouds
        grid1, grid1_pt_weights, grid1_pt_indexes = gridding_distance.forward(
            min_coord_x, max_coord_x, min_coord_y, max_coord_y, min_coord_z, max_coord_z, clamped_scaled_ptcloud1) # Use new min/max and clamped points

        grid2, grid2_pt_weights, grid2_pt_indexes = gridding_distance.forward(
            min_coord_x, max_coord_x, min_coord_y, max_coord_y, min_coord_z, max_coord_z, clamped_scaled_ptcloud2) # Use new min/max and clamped points

        ctx.save_for_backward(grid1_pt_weights, grid1_pt_indexes, grid2_pt_weights, grid2_pt_indexes,
                              scale_factors.unsqueeze(0).unsqueeze(0))
        return grid1, grid2

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
        if not (isinstance(scales, tuple) and len(scales) == 3 and all(isinstance(s, int) and s > 0 for s in scales)):
             raise ValueError("scales must be a tuple of 3 positive integers (Dx, Dy, Dz)")
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
