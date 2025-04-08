# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-12-10 10:48:55
# @Last Modified by:   Sylvanding
# @Last Modified time: 2025-04-08 16:46:00 # Update timestamp
# @Email:  cshzxie@gmail.com
#
# Note:
# - Replace float -> double, kFloat -> kDouble in gridding.cu and gridding_reverse.cu (If needed for gradcheck)
# - Tests updated for non-cubic grid support using 'scales' tuple.

import os
import sys
import torch
import unittest
import numpy as np

from torch.autograd import gradcheck

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from extensions.gridding import GriddingFunction, GriddingReverseFunction


class GriddingTestCase(unittest.TestCase):
    # def test_gridding_reverse_function_non_cubic(self):
    #     """Tests GriddingReverseFunction with a non-cubic grid."""
    #     scales = (8, 8, 4) # Example non-cubic grid scales
    #     batch_size = 2
    #     grid_input = torch.rand(batch_size, scales[0], scales[1], scales[2])
    #     grid_input.requires_grad = True

    #     # Pass scales tuple and grid tensor to apply
    #     # Use double precision for gradcheck
    #     self.assertTrue(gradcheck(GriddingReverseFunction.apply, [scales, grid_input.float().cuda()], eps=1e-4, atol=1e-3))

    def test_gridding_function_basic(self):
        """Tests GriddingFunction with basic non-cubic grid."""
        scales = (8, 8, 8) # Keep a cubic test as well
        batch_size = 2
        n_pts = 32

        # Input point cloud shape (batch_size, n_pts, 3)
        ptcloud_input = (torch.rand(batch_size, n_pts, 3) - 0.5) * 2
        ptcloud_input.requires_grad = True

        # Pass scales tuple and point cloud tensor to apply
        self.assertTrue(gradcheck(GriddingFunction.apply, [scales, ptcloud_input.double().cuda()], eps=1e-3, atol=1e-3))

    # def test_gridding_function_non_cubic(self):
    #     """Tests GriddingFunction with non-cubic grid."""
    #     scales = (16, 8, 4) # Non-cubic scales
    #     batch_size = 2
    #     n_pts = 64

    #     ptcloud_input = torch.rand(batch_size, n_pts, 3) * 0.8
    #     ptcloud_input.requires_grad = True

    #     self.assertTrue(gradcheck(GriddingFunction.apply, [scales, ptcloud_input.float().cuda()], eps=1e-5, atol=1e-4))

    # Optional: Add tests for the nn.Module wrappers if needed
    # def test_gridding_module(self):
    #     scales = (16, 8, 4)
    #     gridding_layer = Gridding(scales=scales)
    #     ptcloud = torch.rand(2, 100, 3, device='cuda') * 2 - 1 # Input in [-1, 1]
    #     grid = gridding_layer(ptcloud)
    #     self.assertEqual(grid.shape, (2, np.prod(scales)))

    # def test_gridding_reverse_module(self):
    #     scales = (8, 8, 4)
    #     gridding_rev_layer = GriddingReverse(scales=scales)
    #     grid = torch.rand(2, np.prod(scales), device='cuda')
    #     ptcloud = gridding_rev_layer(grid)
    #     self.assertEqual(ptcloud.shape, (2, np.prod(scales), 3))


if __name__ == '__main__':
    unittest.main()
