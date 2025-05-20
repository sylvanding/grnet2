# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-11-15 20:33:52
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-30 09:55:53
# @Email:  cshzxie@gmail.com

import torch
import numpy as np

import gridding


class GriddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scales_tuple, ptcloud): # scales is now a tuple
        # scales_tuple: (scale_x, scale_y, scale_z)
        # ptcloud is already scaled by the Gridding module, typically to [-scale/2, scale/2]

        # --- MODIFICATION START ---
        scale_x, scale_y, scale_z = scales_tuple # Use the passed tuple

        min_x = -float(scale_x // 2)
        max_x = min_x + scale_x - 1.0
        min_y = -float(scale_y // 2)
        max_y = min_y + scale_y - 1.0
        min_z = -float(scale_z // 2)
        max_z = min_z + scale_z - 1.0

        # Clamp the input ptcloud before passing to CUDA
        # ptcloud coordinates are expected to be in [-scale_dim/2, scale_dim/2]
        # We need to clamp them to [min_coord_dim, max_coord_dim_for_floor + 1 - epsilon)
        # where max_coord_dim_for_floor is max_x, max_y, max_z respectively.
        
        clamped_ptcloud_coords_list = []
        min_coords_for_clamp = [min_x, min_y, min_z]
        # max_val for clamp should be max_coord + 1 - epsilon to ensure floor(clamped_pt) <= max_coord
        max_coords_for_clamp_plus_delta = [max_x + 1.0 - 1e-5, max_y + 1.0 - 1e-5, max_z + 1.0 - 1e-5]

        for dim in range(3):
            clamped_dim_coords = torch.clamp(
                ptcloud[..., dim],
                min=min_coords_for_clamp[dim],
                max=max_coords_for_clamp_plus_delta[dim]
            )
            clamped_ptcloud_coords_list.append(clamped_dim_coords)
        
        clamped_ptcloud = torch.stack(clamped_ptcloud_coords_list, dim=-1)
        
        grid, grid_pt_weights, grid_pt_indexes = gridding.forward(
            min_x, max_x, min_y, max_y, min_z, max_z, clamped_ptcloud
        )
        # --- MODIFICATION END ---
        
        # print(grid.size())             # torch.Size(batch_size, scale_x * scale_y * scale_z)
        # print(grid_pt_weights.size())  # torch.Size(batch_size, n_pts, 8, 3)
        # print(grid_pt_indexes.size())  # torch.Size(batch_size, n_pts, 8)
        ctx.save_for_backward(grid_pt_weights, grid_pt_indexes)

        return grid

    @staticmethod
    def backward(ctx, grad_grid):
        grid_pt_weights, grid_pt_indexes = ctx.saved_tensors
        grad_ptcloud = gridding.backward(grid_pt_weights, grid_pt_indexes, grad_grid)
        # print(grad_ptcloud.size())   # torch.Size(batch_size, n_pts, 3)

        return None, grad_ptcloud


class Gridding(torch.nn.Module):
    def __init__(self, scales=(64, 64, 64)):
        super(Gridding, self).__init__()
        # --- MODIFICATION START ---
        if not (isinstance(scales, tuple) and len(scales) == 3 and all(isinstance(s, int) and s > 0 for s in scales)):
             raise ValueError("scales must be a tuple of 3 positive integers (Dx, Dy, Dz)")
        self.scales_tuple = scales # Store as tuple for GriddingFunction
        self.scales_arr = np.array(scales) # Keep np.array for scale_factors if preferred
        # Ensure scale_factors are derived correctly for the [-scale/2, scale/2] mapping
        self.scale_factors = torch.from_numpy(self.scales_arr / 2.0).float()
        if torch.cuda.is_available():
            self.scale_factors = self.scale_factors.cuda()
        # --- MODIFICATION END ---

    def forward(self, ptcloud):
        # ptcloud assumed in [-1, 1]^3, scale to grid coordinates
        # --- MODIFICATION START ---
        current_scale_factors = self.scale_factors
        if ptcloud.device != current_scale_factors.device:
             current_scale_factors = current_scale_factors.to(ptcloud.device)
        scaled_ptcloud = ptcloud * current_scale_factors.unsqueeze(0).unsqueeze(0) # Scale each dimension
        # --- MODIFICATION END ---
        
        _ptcloud = torch.split(scaled_ptcloud, 1, dim=0)
        grids = []
        for p in _ptcloud:
            non_zeros = torch.sum(p.abs(), dim=2).ne(0) # Check absolute sum for zeros
            if non_zeros.any():
                 p_filtered = p[non_zeros].unsqueeze(dim=0)
                 grids.append(GriddingFunction.apply(self.scales_tuple, p_filtered))
            else:
                 n_grid_vertices = np.prod(self.scales_arr)
                 grids.append(torch.zeros(1, n_grid_vertices, device=ptcloud.device, dtype=ptcloud.dtype))


        return torch.cat(grids, dim=0).contiguous()


class GriddingReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scales, grid): # grid shape: (B, Dx*Dy*Dz)
        # scales: tuple (scale_x, scale_y, scale_z)
        scale_x, scale_y, scale_z = scales
        # Ensure grid is flat
        if grid.ndim != 2:
             # Maybe add a warning or reshape, but expect flattened input
             raise ValueError(f"GriddingReverseFunction.forward expects flattened grid (B, N), got {grid.shape}")
        ptcloud = gridding.rev_forward(scale_x, scale_y, scale_z, grid)
        ctx.save_for_backward(torch.Tensor(scales).int(), grid, ptcloud) # Save scales as tensor
        return ptcloud

    @staticmethod
    def backward(ctx, grad_ptcloud):
        scales_tensor, grid, ptcloud = ctx.saved_tensors
        scales = scales_tensor.tolist() # Convert tensor back to list
        scale_x, scale_y, scale_z = scales[0], scales[1], scales[2]
        # Call CUDA backward, returns grad_grid with shape (B, Dx*Dy*Dz)
        grad_grid = gridding.rev_backward(scale_x, scale_y, scale_z, ptcloud, grid, grad_ptcloud)

        # Return gradient for scales (None) and grid (original flat shape)
        return None, grad_grid


class GriddingReverse(torch.nn.Module):
    def __init__(self, scales=(64, 64, 64)):
        super(GriddingReverse, self).__init__()
        # --- MODIFICATION START ---
        if not (isinstance(scales, tuple) and len(scales) == 3 and all(isinstance(s, int) and s > 0 for s in scales)):
             raise ValueError("scales must be a tuple of 3 positive integers (Dx, Dy, Dz)")
        self.scales = scales # Keep as tuple
        # Inverse scaling factors to map back roughly to [-1, 1]
        self.output_scaling_factors = torch.tensor([2.0 / s for s in scales], dtype=torch.float32)
        if torch.cuda.is_available():
            self.output_scaling_factors = self.output_scaling_factors.cuda()
        # --- MODIFICATION END ---

    def forward(self, grid):
        # grid expected shape: (batch_size, scale_x * scale_y * scale_z) or (batch_size, scale_x, scale_y, scale_z)
        if grid.ndim == 4:
            grid = grid.view(grid.size(0), -1) # Flatten if needed

        ptcloud = GriddingReverseFunction.apply(self.scales, grid)
        # Scale output point cloud dimensions
        # --- MODIFICATION START ---
        current_output_scaling = self.output_scaling_factors
        if ptcloud.device != current_output_scaling.device:
            current_output_scaling = current_output_scaling.to(ptcloud.device)
        ptcloud = ptcloud * current_output_scaling.unsqueeze(0).unsqueeze(0)
        # --- MODIFICATION END ---
        return ptcloud
