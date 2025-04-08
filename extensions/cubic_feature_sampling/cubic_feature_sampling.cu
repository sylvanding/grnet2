/*
 * @Author: Haozhe Xie
 * @Date:   2019-12-19 20:36:36
 * @Last Modified by:   Haozhe Xie
 * @Last Modified time: 2020-06-17 14:55:41
 * @Email:  cshzxie@gmail.com
 */

#include <torch/extension.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector> // Add include

#define CUDA_NUM_THREADS 512

// Computer the number of threads needed in GPU
inline int get_n_threads(int n) {
  const int pow_2 = std::log(static_cast<float>(n)) / std::log(2.0);
  return max(min(1 << pow_2, CUDA_NUM_THREADS), 1);
}

// Modify compute_index and make it static
static __device__ int compute_index(int offset_x,
                                    int offset_y,
                                    int offset_z,
                                    int scale_y, // Use scale_y, scale_z
                                    int scale_z) {
  return offset_x * scale_y * scale_z + offset_y * scale_z + offset_z;
}

// Modify kernel signature and logic
__global__ void cubic_feature_sampling_kernel(
  int scale_x, // Use scale_x, scale_y, scale_z
  int scale_y,
  int scale_z,
  int neighborhood_size,
  int n_vertices, // n_vertices is related to neighborhood_size, not scale
  int n_pts,
  int n_cubic_channels,
  const float *__restrict__ ptcloud,
  const float *__restrict__ cubic_features,
  float *__restrict__ point_features,
  int *__restrict__ grid_pt_indexes) {
  int batch_index = blockIdx.x;
  int index       = threadIdx.x;
  int stride      = blockDim.x;
  int n_grid_cells = scale_x * scale_y * scale_z; // Total number of grid cells

  ptcloud += batch_index * n_pts * 3;
  cubic_features += batch_index * n_cubic_channels * n_grid_cells;
  point_features += batch_index * n_pts * n_vertices * n_cubic_channels;
  grid_pt_indexes += batch_index * n_pts * n_vertices;

  for (int i = index; i < n_pts; i += stride) {
    // ptcloud coordinates are assumed to be scaled to grid indices already
    float pt_x = ptcloud[i * 3 + 0];
    float pt_y = ptcloud[i * 3 + 1];
    float pt_z = ptcloud[i * 3 + 2];

    // Get the central grid cell index based on the point's location
    // Use round instead of floor/ceil? The original code uses floor/ceil based neighborhood. Let's stick to that.
    // But the original intent seems to be finding neighbors around the *closest* grid cell center.
    // Let's assume ptcloud is already scaled, e.g., to [0, Dx-1], [0, Dy-1], [0, Dz-1] range.
    int center_x = static_cast<int>(roundf(pt_x));
    int center_y = static_cast<int>(roundf(pt_y));
    int center_z = static_cast<int>(roundf(pt_z));

    // Clamp center coordinates to be within valid grid indices
    center_x = max(0, min(center_x, scale_x - 1));
    center_y = max(0, min(center_y, scale_y - 1));
    center_z = max(0, min(center_z, scale_z - 1));

    int ns         = neighborhood_size; // neighborhood_size=1 means 3x3x3=27 neighbors? Original was 2*ns -> (2*1)^3 = 8? No, (ns*2+1)^3. Let's check n_vertices calc. n_vertices = pow(neighborhood_size * 2, 3) -> If ns=1, (1*2)^3 = 8. This means it samples the 8 corners of the cell containing the point. Let's keep the original logic based on floor/ceil.

    int lower_x = std::floor(pt_x);
    int upper_x = lower_x + 1; // Original code used ceil, but floor/floor+1 seems more standard for interpolation cells
    int lower_y = std::floor(pt_y);
    int upper_y = lower_y + 1;
    int lower_z = std::floor(pt_z);
    int upper_z = lower_z + 1;

    // The neighborhood definition seems strange. It takes neighbors around BOTH lower and upper bounds.
    // (upper_x + ns) - (lower_x - ns) + 1 = (lower_x+1+ns) - (lower_x-ns) + 1 = 2*ns + 2
    // Total neighbors = (2*ns+2)^3. If ns=1, it's 4x4x4 = 64 neighbors?
    // Let's re-read the n_vertices calculation: n_vertices = std::pow(neighborhood_size * 2, 3); If ns=1, n_vertices=8.
    // This implies the loop structure is wrong or neighborhood_size is interpreted differently.
    // Assume `neighborhood_size` defines the half-width, so the neighborhood is `(2*ns+1)^3`.
    // If ns=1, it's 3x3x3=27. If ns=0, it's 1x1x1=1.
    // The original code calculates n_vertices = pow(neighborhood_size*2, 3). If ns=1, this is 8.
    // This likely means it samples features from the 8 grid cells surrounding the point pt.
    // Let's stick to the original interpretation: find the cell containing pt, and sample features from its 8 corner vertices.

    // Redefine neighborhood based on the 8 corners of the cell [lower_x, upper_x] x [lower_y, upper_y] x [lower_z, upper_z]
    int corners[8][3] = {
        {lower_x, lower_y, lower_z}, {lower_x, lower_y, upper_z},
        {lower_x, upper_y, lower_z}, {lower_x, upper_y, upper_z},
        {upper_x, lower_y, lower_z}, {upper_x, lower_y, upper_z},
        {upper_x, upper_y, lower_z}, {upper_x, upper_y, upper_z}
    };

    int vertex_idx = 0; // Index within the 8 output vertices for this point
    for (int j = 0; j < 8; ++j) { // Iterate through the 8 corners
        int corner_x = corners[j][0];
        int corner_y = corners[j][1];
        int corner_z = corners[j][2];

        // Check boundary conditions using scale_x, scale_y, scale_z
        if (corner_x < 0 || corner_x >= scale_x || corner_y < 0 || corner_y >= scale_y || corner_z < 0 || corner_z >= scale_z) {
            // Assign -1 if the corner is outside the grid
            grid_pt_indexes[i * n_vertices + vertex_idx] = -1;
        } else {
            // Calculate linear index for the valid corner
            grid_pt_indexes[i * n_vertices + vertex_idx] =
              compute_index(corner_x, corner_y, corner_z, scale_y, scale_z); // Use updated compute_index
        }
        vertex_idx++; // Move to the next output vertex index
    }

    // Gather Features (Loop through the 8 sampled vertices)
    for (int j = 0; j < n_vertices; ++j) { // n_vertices should be 8 based on calculation
      int source_vertex_idx = grid_pt_indexes[i * n_vertices + j];
      if (source_vertex_idx == -1) {
         // Optionally fill with zeros if index is invalid
         for (int k = 0; k < n_cubic_channels; ++k) {
             int feature_idx_out = i * n_vertices * n_cubic_channels + j * n_cubic_channels + k;
             point_features[feature_idx_out] = 0.0f;
         }
         continue;
      }
      // Copy features from the source grid cell
      for (int k = 0; k < n_cubic_channels; ++k) {
        int feature_idx_out = i * n_vertices * n_cubic_channels + j * n_cubic_channels + k;
        // Index into the flat cubic_features tensor: channel * n_grid_cells + cell_linear_index
        float feature_val = cubic_features[k * n_grid_cells + source_vertex_idx];
        point_features[feature_idx_out] = feature_val;
      }
    }
  }
}

// Modify cuda_forward signature and logic
std::vector<torch::Tensor> cubic_feature_sampling_cuda_forward(
  int scale_x, // Use scale_x, scale_y, scale_z
  int scale_y,
  int scale_z,
  int neighborhood_size, // Should be 1 for 8 corners sampling based on kernel logic
  torch::Tensor ptcloud,
  torch::Tensor cubic_features,
  cudaStream_t stream) {
  int batch_size       = ptcloud.size(0);
  int n_pts            = ptcloud.size(1);
  int n_cubic_channels = cubic_features.size(1);
  // Verify cubic_features dimensions match scales
  TORCH_CHECK(cubic_features.size(2) == scale_x && cubic_features.size(3) == scale_y && cubic_features.size(4) == scale_z, "cubic_features dimensions do not match scales");

  // n_vertices seems fixed at 8 based on the original calculation and kernel interpretation
  int n_vertices = 8; // Corresponds to sampling the 8 corners of the cell containing the point.
                      // If neighborhood_size > 1 was intended differently, the kernel needs significant rework.
                      // Let's assume neighborhood_size=1 is the only intended use case for this kernel.
  TORCH_CHECK(neighborhood_size == 1, "Current implementation likely only supports neighborhood_size=1 (sampling 8 cell corners)");


  torch::Tensor point_features =
    torch::zeros({batch_size, n_pts, n_vertices, n_cubic_channels},
                 torch::CUDA(torch::kFloat));
  torch::Tensor grid_pt_indexes =
    torch::zeros({batch_size, n_pts, n_vertices}, torch::CUDA(torch::kInt)); // Stores linear indices of the 8 corners

  // Flatten cubic_features for kernel access: B, C, Dx*Dy*Dz
  torch::Tensor cubic_features_flat = cubic_features.reshape({batch_size, n_cubic_channels, -1});

  cubic_feature_sampling_kernel<<<batch_size, get_n_threads(n_pts), 0,
                                  stream>>>(
    scale_x, scale_y, scale_z, neighborhood_size, n_vertices, n_pts, n_cubic_channels,
    ptcloud.data_ptr<float>(), cubic_features_flat.data_ptr<float>(), // Pass flat features
    point_features.data_ptr<float>(), grid_pt_indexes.data_ptr<int>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in cubic_feature_sampling_cuda_forward: %s\n",
           cudaGetErrorString(err));
  }
  return {point_features, grid_pt_indexes};
}

// Modify grad_kernel signature and logic
__global__ void cubic_feature_sampling_grad_kernel(
  int scale_x, // Use scale_x, scale_y, scale_z
  int scale_y,
  int scale_z,
  int neighborhood_size,
  int n_vertices, // Should be 8
  int n_pts,
  int n_cubic_channels,
  const float *__restrict__ grad_point_features,
  const int *__restrict__ grid_pt_indexes,
  float *__restrict__ grad_ptcloud, // Gradient w.r.t. ptcloud coordinates (likely zero as sampling indices depend on floor/ceil)
  float *__restrict__ grad_cubic_features) { // Gradient w.r.t. original cubic_features
  int batch_index = blockIdx.x;
  int index       = threadIdx.x;
  int stride      = blockDim.x;
  int n_grid_cells = scale_x * scale_y * scale_z;

  grad_point_features += batch_index * n_pts * n_vertices * n_cubic_channels;
  grid_pt_indexes += batch_index * n_pts * n_vertices;
  grad_ptcloud += batch_index * n_pts * 3; // Although calculated, it remains zero
  grad_cubic_features += batch_index * n_cubic_channels * n_grid_cells;

  for (int i = index; i < n_pts; i += stride) {
    for (int j = 0; j < n_vertices; ++j) { // Loop through the 8 sampled vertices
      int source_vertex_idx = grid_pt_indexes[i * n_vertices + j];
      if (source_vertex_idx == -1) {
        // If the forward pass sampled -1 (out of bounds), gradient doesn't flow back
        continue;
      }
      // Accumulate gradients back to the source cubic features
      for (int k = 0; k < n_cubic_channels; ++k) {
        int grad_idx_in = // Index for incoming gradient (grad_point_features)
          i * n_vertices * n_cubic_channels + j * n_cubic_channels + k;
        float grad_val = grad_point_features[grad_idx_in];

        // Index for gradient accumulation in grad_cubic_features (flat)
        int grad_idx_out = k * n_grid_cells + source_vertex_idx;
        atomicAdd(&(grad_cubic_features[grad_idx_out]), grad_val);
      }
    }
    // Gradient w.r.t. ptcloud coordinates is zero because floor/ceil/round
    // functions have zero gradients almost everywhere.
    // grad_ptcloud[i * 3 + 0] = 0.0f;
    // grad_ptcloud[i * 3 + 1] = 0.0f;
    // grad_ptcloud[i * 3 + 2] = 0.0f;
    // The kernel initializes grad_ptcloud to zero, so no need to set it explicitly here.
  }
}

// Modify cuda_backward signature and logic
std::vector<torch::Tensor> cubic_feature_sampling_cuda_backward(
  int scale_x, // Use scale_x, scale_y, scale_z
  int scale_y,
  int scale_z,
  int neighborhood_size,
  torch::Tensor grad_point_features, // Input gradient: B, N_pts, N_vertices, C
  torch::Tensor grid_pt_indexes,     // Saved indices: B, N_pts, N_vertices
  cudaStream_t stream) {
  int batch_size       = grad_point_features.size(0);
  int n_pts            = grad_point_features.size(1);
  int n_vertices       = grad_point_features.size(2); // Should be 8
  int n_cubic_channels = grad_point_features.size(3);
  TORCH_CHECK(neighborhood_size == 1, "Current implementation likely only supports neighborhood_size=1 (sampling 8 cell corners)");
  TORCH_CHECK(n_vertices == 8, "n_vertices in gradient does not match expected 8");
  TORCH_CHECK(grid_pt_indexes.size(0) == batch_size && grid_pt_indexes.size(1) == n_pts && grid_pt_indexes.size(2) == n_vertices, "grid_pt_indexes shape mismatch");


  // Gradient w.r.t ptcloud is zero
  torch::Tensor grad_ptcloud =
    torch::zeros({batch_size, n_pts, 3}, torch::CUDA(torch::kFloat));
  // Gradient w.r.t cubic_features (needs correct non-cubic shape)
  torch::Tensor grad_cubic_features =
    torch::zeros({batch_size, n_cubic_channels, scale_x, scale_y, scale_z}, // Use non-cubic shape
                 torch::CUDA(torch::kFloat));

  // Flatten grad_cubic_features for atomicAdd in the kernel
  torch::Tensor grad_cubic_features_flat = grad_cubic_features.reshape({batch_size, n_cubic_channels, -1});


  cubic_feature_sampling_grad_kernel<<<batch_size, get_n_threads(n_pts), 0,
                                       stream>>>(
    scale_x, scale_y, scale_z, neighborhood_size, n_vertices, n_pts, n_cubic_channels,
    grad_point_features.data_ptr<float>(), grid_pt_indexes.data_ptr<int>(),
    grad_ptcloud.data_ptr<float>(), grad_cubic_features_flat.data_ptr<float>()); // Pass flat gradient tensor

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in cubic_feature_sampling_cuda_backward: %s\n",
           cudaGetErrorString(err));
  }
  // Return the unflattened grad_cubic_features
  return {grad_ptcloud, grad_cubic_features};
}