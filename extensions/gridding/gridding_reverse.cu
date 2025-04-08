/*
 * @Author: Haozhe Xie
 * @Date:   2019-11-21 16:42:18
 * @Last Modified by:   Haozhe Xie
 * @Last Modified time: 2020-06-17 15:00:21
 * @Email:  cshzxie@gmail.com
 */

#include <bits/stdc++.h>
#include <torch/extension.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_NUM_THREADS 512
#define EPS 1e-6

// Computer the number of threads needed in GPU
inline int get_n_threads(int n) {
  const int pow_2 = std::log(static_cast<float>(n)) / std::log(2.0);
  return max(min(1 << pow_2, CUDA_NUM_THREADS), 1);
}

static __device__ int compute_index(int offset_x,
                             int offset_y,
                             int offset_z,
                             int scale_y,
                             int scale_z) {
  return offset_x * scale_y * scale_z + offset_y * scale_z + offset_z;
}

__global__ void gridding_reverse_kernel(int scale_x,
                                        int scale_y,
                                        int scale_z,
                                        int n_pts,
                                        const float *__restrict__ grid,
                                        float *__restrict__ ptcloud) {
  int batch_index = blockIdx.x;
  int index       = threadIdx.x;
  int stride      = blockDim.x;

  ptcloud += batch_index * n_pts * 3;
  grid += batch_index * n_pts;

  int scale_yz = scale_y * scale_z;

  for (int j = index; j < n_pts; j += stride) {
    int x_offset = j / scale_yz;
    int y_offset = (j % scale_yz) / scale_z;
    int z_offset = j % scale_z;

    if (x_offset == 0 || y_offset == 0 || z_offset == 0 ||
        x_offset >= scale_x || y_offset >= scale_y || z_offset >= scale_z) {
        continue;
    }
    if (x_offset < 1 || y_offset < 1 || z_offset < 1) {
        continue;
    }

    int neighbor_indices[8] = {
      compute_index(x_offset - 1, y_offset - 1, z_offset - 1, scale_y, scale_z),
      compute_index(x_offset - 1, y_offset - 1, z_offset,     scale_y, scale_z),
      compute_index(x_offset - 1, y_offset,     z_offset - 1, scale_y, scale_z),
      compute_index(x_offset - 1, y_offset,     z_offset,     scale_y, scale_z),
      compute_index(x_offset,     y_offset - 1, z_offset - 1, scale_y, scale_z),
      compute_index(x_offset,     y_offset - 1, z_offset,     scale_y, scale_z),
      compute_index(x_offset,     y_offset,     z_offset - 1, scale_y, scale_z),
      j
    };

    float weights[8];
    float weights_sum = 0;
    for (size_t i = 0; i < 8; ++i) {
      weights[i] = grid[neighbor_indices[i]];
      weights_sum += weights[i];
    }

    if (weights_sum < EPS) {
      // If the weights sum is less than EPS, set the point cloud to the center of the grid
      // ptcloud[j * 3 + 0] = x_offset - scale_x / 2.0f + 0.5f;
      // ptcloud[j * 3 + 1] = y_offset - scale_y / 2.0f + 0.5f;
      // ptcloud[j * 3 + 2] = z_offset - scale_z / 2.0f + 0.5f;
      continue;
    }
    for (size_t i = 0; i < 8; ++i) {
      weights[i] /= weights_sum;
    }

    float x_offset_centered = x_offset - scale_x / 2.0f;
    float y_offset_centered = y_offset - scale_y / 2.0f;
    float z_offset_centered = z_offset - scale_z / 2.0f;

    ptcloud[j * 3 + 0] = 0.0f;
    ptcloud[j * 3 + 1] = 0.0f;
    ptcloud[j * 3 + 2] = 0.0f;

    ptcloud[j * 3 + 0] += weights[0] * (x_offset_centered - 0.5f);
    ptcloud[j * 3 + 1] += weights[0] * (y_offset_centered - 0.5f);
    ptcloud[j * 3 + 2] += weights[0] * (z_offset_centered - 0.5f);
    ptcloud[j * 3 + 0] += weights[1] * (x_offset_centered - 0.5f);
    ptcloud[j * 3 + 1] += weights[1] * (y_offset_centered - 0.5f);
    ptcloud[j * 3 + 2] += weights[1] * (z_offset_centered + 0.5f);
    ptcloud[j * 3 + 0] += weights[2] * (x_offset_centered - 0.5f);
    ptcloud[j * 3 + 1] += weights[2] * (y_offset_centered + 0.5f);
    ptcloud[j * 3 + 2] += weights[2] * (z_offset_centered - 0.5f);
    ptcloud[j * 3 + 0] += weights[3] * (x_offset_centered - 0.5f);
    ptcloud[j * 3 + 1] += weights[3] * (y_offset_centered + 0.5f);
    ptcloud[j * 3 + 2] += weights[3] * (z_offset_centered + 0.5f);
    ptcloud[j * 3 + 0] += weights[4] * (x_offset_centered + 0.5f);
    ptcloud[j * 3 + 1] += weights[4] * (y_offset_centered - 0.5f);
    ptcloud[j * 3 + 2] += weights[4] * (z_offset_centered - 0.5f);
    ptcloud[j * 3 + 0] += weights[5] * (x_offset_centered + 0.5f);
    ptcloud[j * 3 + 1] += weights[5] * (y_offset_centered - 0.5f);
    ptcloud[j * 3 + 2] += weights[5] * (z_offset_centered + 0.5f);
    ptcloud[j * 3 + 0] += weights[6] * (x_offset_centered + 0.5f);
    ptcloud[j * 3 + 1] += weights[6] * (y_offset_centered + 0.5f);
    ptcloud[j * 3 + 2] += weights[6] * (z_offset_centered - 0.5f);
    ptcloud[j * 3 + 0] += weights[7] * (x_offset_centered + 0.5f);
    ptcloud[j * 3 + 1] += weights[7] * (y_offset_centered + 0.5f);
    ptcloud[j * 3 + 2] += weights[7] * (z_offset_centered + 0.5f);
  }
}

torch::Tensor gridding_reverse_cuda_forward(int scale_x,
                                            int scale_y,
                                            int scale_z,
                                            torch::Tensor grid,
                                            cudaStream_t stream) {
  int batch_size = grid.size(0);
  int n_pts      = scale_x * scale_y * scale_z;

  torch::Tensor ptcloud =
    torch::zeros({batch_size, n_pts, 3}, torch::CUDA(torch::kFloat));

  gridding_reverse_kernel<<<batch_size, get_n_threads(n_pts), 0, stream>>>(
    scale_x, scale_y, scale_z, n_pts, grid.data_ptr<float>(), ptcloud.data_ptr<float>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in gridding_reverse_cuda_forward: %s\n", cudaGetErrorString(err));
  }
  return ptcloud;
}

__global__ void gridding_reverse_grad_kernel(
  int scale_x,
  int scale_y,
  int scale_z,
  int n_pts,
  const float *__restrict__ ptcloud,
  const float *__restrict__ grid,
  const float *__restrict__ grad_ptcloud,
  float *__restrict__ grad_grid) {
  int batch_index = blockIdx.x;
  int index       = threadIdx.x;
  int stride      = blockDim.x;

  ptcloud += batch_index * n_pts * 3;
  grid += batch_index * n_pts;
  grad_ptcloud += batch_index * n_pts * 3;
  grad_grid += batch_index * n_pts;

  int scale_yz = scale_y * scale_z;

  for (int j = index; j < n_pts; j += stride) {
    int x_offset = j / scale_yz;
    int y_offset = (j % scale_yz) / scale_z;
    int z_offset = j % scale_z;

    if (x_offset == 0 || y_offset == 0 || z_offset == 0 ||
        x_offset >= scale_x || y_offset >= scale_y || z_offset >= scale_z) {
        continue;
    }
    if (x_offset < 1 || y_offset < 1 || z_offset < 1) {
        continue;
    }

    int gvtx_indexes[8] = {
      compute_index(x_offset - 1, y_offset - 1, z_offset - 1, scale_y, scale_z),
      compute_index(x_offset - 1, y_offset - 1, z_offset,     scale_y, scale_z),
      compute_index(x_offset - 1, y_offset,     z_offset - 1, scale_y, scale_z),
      compute_index(x_offset - 1, y_offset,     z_offset,     scale_y, scale_z),
      compute_index(x_offset,     y_offset - 1, z_offset - 1, scale_y, scale_z),
      compute_index(x_offset,     y_offset - 1, z_offset,     scale_y, scale_z),
      compute_index(x_offset,     y_offset,     z_offset - 1, scale_y, scale_z),
      j
    };
    float weights[8] = {grid[gvtx_indexes[0]], grid[gvtx_indexes[1]],
                        grid[gvtx_indexes[2]], grid[gvtx_indexes[3]],
                        grid[gvtx_indexes[4]], grid[gvtx_indexes[5]],
                        grid[gvtx_indexes[6]], grid[gvtx_indexes[7]]};

    float weights_sum = 0;
    for (size_t i = 0; i < 8; ++i) {
      weights_sum += weights[i];
    }

    if (weights_sum < EPS) {
      continue;
    }
    for (size_t i = 0; i < 8; ++i) {
      weights[i] /= weights_sum;
    }

    float x_offset_centered = x_offset - scale_x / 2.0f;
    float y_offset_centered = y_offset - scale_y / 2.0f;
    float z_offset_centered = z_offset - scale_z / 2.0f;

    float center_coords[8][3] = {
        {x_offset_centered - 0.5f, y_offset_centered - 0.5f, z_offset_centered - 0.5f},
        {x_offset_centered - 0.5f, y_offset_centered - 0.5f, z_offset_centered + 0.5f},
        {x_offset_centered - 0.5f, y_offset_centered + 0.5f, z_offset_centered - 0.5f},
        {x_offset_centered - 0.5f, y_offset_centered + 0.5f, z_offset_centered + 0.5f},
        {x_offset_centered + 0.5f, y_offset_centered - 0.5f, z_offset_centered - 0.5f},
        {x_offset_centered + 0.5f, y_offset_centered - 0.5f, z_offset_centered + 0.5f},
        {x_offset_centered + 0.5f, y_offset_centered + 0.5f, z_offset_centered - 0.5f},
        {x_offset_centered + 0.5f, y_offset_centered + 0.5f, z_offset_centered + 0.5f}
    };

    float grad_pt_x = grad_ptcloud[j * 3 + 0];
    float grad_pt_y = grad_ptcloud[j * 3 + 1];
    float grad_pt_z = grad_ptcloud[j * 3 + 2];
    float pt_x = ptcloud[j * 3 + 0];
    float pt_y = ptcloud[j * 3 + 1];
    float pt_z = ptcloud[j * 3 + 2];

    for(int k = 0; k < 8; ++k) {
        float factor = (grad_pt_x * (center_coords[k][0] - pt_x) +
                        grad_pt_y * (center_coords[k][1] - pt_y) +
                        grad_pt_z * (center_coords[k][2] - pt_z)) / weights_sum;
        atomicAdd(&(grad_grid[gvtx_indexes[k]]), factor);
    }
  }
}

torch::Tensor gridding_reverse_cuda_backward(int scale_x,
                                             int scale_y,
                                             int scale_z,
                                             torch::Tensor ptcloud,
                                             torch::Tensor grid,
                                             torch::Tensor grad_ptcloud,
                                             cudaStream_t stream) {
  int batch_size = ptcloud.size(0);
  int n_pts      = ptcloud.size(1);

  TORCH_CHECK(n_pts == scale_x * scale_y * scale_z, "Point cloud size does not match scales");

  torch::Tensor grad_grid =
    torch::zeros({batch_size, n_pts}, torch::CUDA(torch::kFloat));

  gridding_reverse_grad_kernel<<<batch_size, get_n_threads(n_pts), 0, stream>>>(
    scale_x, scale_y, scale_z, n_pts, ptcloud.data_ptr<float>(), grid.data_ptr<float>(),
    grad_ptcloud.data_ptr<float>(), grad_grid.data_ptr<float>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in gridding_reverse_cuda_backward: %s\n", cudaGetErrorString(err));
  }
  return grad_grid;
}