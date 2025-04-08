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
    def forward(ctx, scales, ptcloud):
        # scales: (scale_x, scale_y, scale_z)
        scale_x, scale_y, scale_z = scales
        min_x, max_x = -scale_x // 2, scale_x // 2 - 1
        min_y, max_y = -scale_y // 2, scale_y // 2 - 1
        min_z, max_z = -scale_z // 2, scale_z // 2 - 1

        grid, grid_pt_weights, grid_pt_indexes = gridding.forward(min_x, max_x, min_y, max_y, min_z, max_z, ptcloud)
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
        self.scales = np.array(scales)
        self.scale_factors = torch.from_numpy(self.scales / 2.0).float().cuda() # Use float tensor for scaling

    def forward(self, ptcloud):
        # ptcloud assumed in [-1, 1]^3, scale to grid coordinates
        scaled_ptcloud = ptcloud * self.scale_factors.unsqueeze(0).unsqueeze(0) # Scale each dimension
        _ptcloud = torch.split(scaled_ptcloud, 1, dim=0)
        grids = []
        for p in _ptcloud:
            non_zeros = torch.sum(p.abs(), dim=2).ne(0) # Check absolute sum for zeros
            # Ensure p has points after filtering before applying gridding
            if non_zeros.any():
                 p_filtered = p[non_zeros].unsqueeze(dim=0)
                 grids.append(GriddingFunction.apply(self.scales, p_filtered))
            else:
                 # Handle empty point cloud case, e.g., return zero grid
                 n_grid_vertices = np.prod(self.scales)
                 grids.append(torch.zeros(1, n_grid_vertices, device=ptcloud.device, dtype=ptcloud.dtype))


        return torch.cat(grids, dim=0).contiguous()


class GriddingReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scales, grid):
        # scales: (scale_x, scale_y, scale_z)
        scale_x, scale_y, scale_z = scales
        ptcloud = gridding.rev_forward(scale_x, scale_y, scale_z, grid)
        ctx.save_for_backward(torch.Tensor(scales).int(), grid, ptcloud) # Save scales as tensor
        return ptcloud

    @staticmethod
    def backward(ctx, grad_ptcloud):
        scales_tensor, grid, ptcloud = ctx.saved_tensors
        scales = scales_tensor.tolist() # Convert tensor back to list
        scale_x, scale_y, scale_z = scales[0], scales[1], scales[2]
        grad_grid = gridding.rev_backward(scale_x, scale_y, scale_z, ptcloud, grid, grad_ptcloud)
        # Reshape according to the actual grid dimensions
        grad_grid = grad_grid.view(-1, scale_x, scale_y, scale_z)
        return None, grad_grid


class GriddingReverse(torch.nn.Module):
    def __init__(self, scales=(64, 64, 64)):
        super(GriddingReverse, self).__init__()
        self.scales = scales
        # Inverse scaling factors to map back roughly to [-1, 1]
        self.output_scaling = torch.tensor([2.0 / s for s in scales], dtype=torch.float32).cuda()

    def forward(self, grid):
        # grid expected shape: (batch_size, scale_x * scale_y * scale_z) or (batch_size, scale_x, scale_y, scale_z)
        if grid.ndim == 4:
            grid = grid.view(grid.size(0), -1) # Flatten if needed

        ptcloud = GriddingReverseFunction.apply(self.scales, grid)
        # Scale output point cloud dimensions
        ptcloud = ptcloud * self.output_scaling.unsqueeze(0).unsqueeze(0)
        return ptcloud
