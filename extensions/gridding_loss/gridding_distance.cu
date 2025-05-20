/*
 * @Author: Haozhe Xie
 * @Date:   2019-12-30 11:35:30
 * @Last Modified by:   Haozhe Xie
 * @Last Modified time: 2020-06-17 15:00:45
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

// Make compute_index static
static __device__ int compute_index(
  int offset_x, int offset_y, int offset_z, int len_y, int len_z) {
  return offset_x * len_y * len_z + offset_y * len_z + offset_z;
}

__device__ float compute_weight(float x, float x0) { return 1 - abs(x - x0); }

// Modify kernel for corrected grid shape
__global__ void gridding_dist_kernel(int n_grid_vertices,
                                     int n_pts,
                                     float min_x,
                                     float min_y,
                                     float min_z,
                                     int len_y,
                                     int len_z,
                                     const float *__restrict__ ptcloud,
                                     float *__restrict__ grid_weights, // Shape: (B, n_grid_vertices)
                                     float *__restrict__ grid_pt_weights, // Shape: (B, N_pts, 8, 3)
                                     int *__restrict__ grid_pt_indexes) { // Shape: (B, N_pts, 8), stores linear grid index
  int batch_index = blockIdx.x;
  int index       = threadIdx.x;
  int stride      = blockDim.x;

  ptcloud += batch_index * n_pts * 3;
  // grid_weights shape correction: remove the last dimension of 8
  grid_weights += batch_index * n_grid_vertices; // Now points to start of grid for this batch item
  grid_pt_weights += batch_index * n_pts * 8 * 3; // Correct offset: B*N*8*3 -> B*N*24
  grid_pt_indexes += batch_index * n_pts * 8; // Correct offset

  for (int j = index; j < n_pts; j += stride) {
    float pt_x = ptcloud[j * 3 + 0];
    float pt_y = ptcloud[j * 3 + 1];
    float pt_z = ptcloud[j * 3 + 2];

    // --- Calculate corner indices and weights (same as original gridding) ---
    int lower_x = std::floor(pt_x);
    int upper_x = lower_x + 1; // Use floor + 1 consistently
    int lower_y = std::floor(pt_y);
    int upper_y = lower_y + 1;
    int lower_z = std::floor(pt_z);
    int upper_z = lower_z + 1;

    int lx_offset = lower_x - min_x, ux_offset = upper_x - min_x;
    int ly_offset = lower_y - min_y, uy_offset = upper_y - min_y;
    int lz_offset = lower_z - min_z, uz_offset = upper_z - min_z;
    
    // If len_z is 1, uz_offset for indexing should be lz_offset
    // to prevent accessing out of bounds for the z-dimension.
    int indexing_uz_offset = uz_offset;
    if (len_z == 1) {
      indexing_uz_offset = lz_offset;
    }

    // LLL
    grid_pt_indexes[j * 8 + 0] = compute_index(lx_offset, ly_offset, lz_offset, len_y, len_z);
    grid_pt_weights[j * 24 + 0] = compute_weight(pt_x, lower_x);
    grid_pt_weights[j * 24 + 1] = compute_weight(pt_y, lower_y);
    grid_pt_weights[j * 24 + 2] = compute_weight(pt_z, lower_z);
    // LLU
    grid_pt_indexes[j * 8 + 1] = compute_index(lx_offset, ly_offset, indexing_uz_offset, len_y, len_z);
    grid_pt_weights[j * 24 + 3] = compute_weight(pt_x, lower_x);
    grid_pt_weights[j * 24 + 4] = compute_weight(pt_y, lower_y);
    grid_pt_weights[j * 24 + 5] = compute_weight(pt_z, upper_z);
    // LUL
    grid_pt_indexes[j * 8 + 2] = compute_index(lx_offset, uy_offset, lz_offset, len_y, len_z);
    grid_pt_weights[j * 24 + 6] = compute_weight(pt_x, lower_x);
    grid_pt_weights[j * 24 + 7] = compute_weight(pt_y, upper_y);
    grid_pt_weights[j * 24 + 8] = compute_weight(pt_z, lower_z);
    // LUU
    grid_pt_indexes[j * 8 + 3] = compute_index(lx_offset, uy_offset, indexing_uz_offset, len_y, len_z);
    grid_pt_weights[j * 24 + 9]  = compute_weight(pt_x, lower_x);
    grid_pt_weights[j * 24 + 10] = compute_weight(pt_y, upper_y);
    grid_pt_weights[j * 24 + 11] = compute_weight(pt_z, upper_z);
    // ULL
    grid_pt_indexes[j * 8 + 4] = compute_index(ux_offset, ly_offset, lz_offset, len_y, len_z);
    grid_pt_weights[j * 24 + 12] = compute_weight(pt_x, upper_x);
    grid_pt_weights[j * 24 + 13] = compute_weight(pt_y, lower_y);
    grid_pt_weights[j * 24 + 14] = compute_weight(pt_z, lower_z);
    // ULU
    grid_pt_indexes[j * 8 + 5] = compute_index(ux_offset, ly_offset, indexing_uz_offset, len_y, len_z);
    grid_pt_weights[j * 24 + 15] = compute_weight(pt_x, upper_x);
    grid_pt_weights[j * 24 + 16] = compute_weight(pt_y, lower_y);
    grid_pt_weights[j * 24 + 17] = compute_weight(pt_z, upper_z);
    // UUL
    grid_pt_indexes[j * 8 + 6] = compute_index(ux_offset, uy_offset, lz_offset, len_y, len_z);
    grid_pt_weights[j * 24 + 18] = compute_weight(pt_x, upper_x);
    grid_pt_weights[j * 24 + 19] = compute_weight(pt_y, upper_y);
    grid_pt_weights[j * 24 + 20] = compute_weight(pt_z, lower_z);
    // UUU
    grid_pt_indexes[j * 8 + 7] = compute_index(ux_offset, uy_offset, indexing_uz_offset, len_y, len_z);
    grid_pt_weights[j * 24 + 21] = compute_weight(pt_x, upper_x);
    grid_pt_weights[j * 24 + 22] = compute_weight(pt_y, upper_y);
    grid_pt_weights[j * 24 + 23] = compute_weight(pt_z, upper_z);
    // --- End of corner indices and weights calculation ---
  }

  __syncthreads(); // Wait for all threads to compute weights and indices

  // --- Accumulate weights onto the grid (corrected) ---
  for (int j = index; j < n_pts; j += stride) {
    // Iterate through the 8 corners influenced by point j
    for (int k = 0; k < 8; ++k) {
        int gvtx_idx = grid_pt_indexes[j * 8 + k]; // Get linear index of the k-th corner
        // Check if index is valid (within grid boundaries defined implicitly by min/max and len)
        if (gvtx_idx >= 0 && gvtx_idx < n_grid_vertices) {
            // Calculate the trilinear weight for this corner
            float weight_prod = grid_pt_weights[j * 24 + k*3 + 0] * // wx
                                grid_pt_weights[j * 24 + k*3 + 1] * // wy
                                grid_pt_weights[j * 24 + k*3 + 2];  // wz
            // Atomically add this weight to the corresponding grid vertex
            atomicAdd(&(grid_weights[gvtx_idx]), weight_prod);
        }
    }
  }
  // --- End of grid weight accumulation ---
}

// Modify cuda_forward for corrected grid shape
std::vector<torch::Tensor> gridding_distance_cuda_forward(float min_x,
                                                          float max_x,
                                                          float min_y,
                                                          float max_y,
                                                          float min_z,
                                                          float max_z,
                                                          torch::Tensor ptcloud,
                                                          cudaStream_t stream) {
  int batch_size      = ptcloud.size(0);
  int n_pts           = ptcloud.size(1);
  // Calculate grid dimensions based on float bounds (potential for off-by-one if not careful)
  // It's generally safer to work with integer scales/sizes directly if possible.
  // Assuming the Python side calculated min/max to ensure integer dimensions.
  int len_x           = static_cast<int>(roundf(max_x - min_x + 1));
  int len_y           = static_cast<int>(roundf(max_y - min_y + 1));
  int len_z           = static_cast<int>(roundf(max_z - min_z + 1));
  int n_grid_vertices = len_x * len_y * len_z;

  // Correct grid_weights shape: remove last dimension of 8
  torch::Tensor grid_weights =
    torch::zeros({batch_size, n_grid_vertices}, torch::CUDA(torch::kFloat));
  // grid_pt_weights shape seems correct: (B, N_pts, 8 corners, 3 weights per corner)
  torch::Tensor grid_pt_weights =
    torch::zeros({batch_size, n_pts, 8, 3}, torch::CUDA(torch::kFloat));
  // grid_pt_indexes stores the linear index of the 8 corners for each point
  torch::Tensor grid_pt_indexes =
    torch::zeros({batch_size, n_pts, 8}, torch::CUDA(torch::kInt));

  gridding_dist_kernel<<<batch_size, get_n_threads(n_pts), 0, stream>>>(
    n_grid_vertices, n_pts, min_x, min_y, min_z, len_y, len_z,
    ptcloud.data_ptr<float>(), grid_weights.data_ptr<float>(),
    grid_pt_weights.data_ptr<float>(), grid_pt_indexes.data_ptr<int>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in gridding_distance_cuda_forward: %s\n",
           cudaGetErrorString(err));
  }
  // Return corrected grid_weights shape
  return {grid_weights, grid_pt_weights, grid_pt_indexes};
}

// Modify grad_kernel for corrected grid shape
__global__ void gridding_dist_grad_kernel(
  int n_grid_vertices,
  int n_pts,
  const float *__restrict__ grid_pt_weights, // (B, N_pts, 8, 3)
  const int *__restrict__ grid_pt_indexes,   // (B, N_pts, 8) linear indices
  const float *__restrict__ grad_grid,       // Input grad, shape (B, n_grid_vertices)
  float *__restrict__ grad_ptcloud) {        // Output grad, shape (B, N_pts, 3)
  int batch_index = blockIdx.x;
  int index       = threadIdx.x;
  int stride      = blockDim.x;

  grid_pt_weights += batch_index * n_pts * 8 * 3; // Offset: B*N*24
  grid_pt_indexes += batch_index * n_pts * 8;     // Offset: B*N*8
  // Correct grad_grid offset (shape is now B, n_grid_vertices)
  grad_grid += batch_index * n_grid_vertices;
  grad_ptcloud += batch_index * n_pts * 3;

  // Accumulate gradient for each point based on the gradients of the grid cells it affects
  for (int j = index; j < n_pts; j += stride) {
      float grad_x = 0.0f;
      float grad_y = 0.0f;
      float grad_z = 0.0f;

      // Iterate through the 8 corners influenced by point j
      for (int k = 0; k < 8; ++k) {
          int gvtx_idx = grid_pt_indexes[j * 8 + k]; // Get linear index of the k-th corner

          // Check if index is valid
          if (gvtx_idx >= 0 && gvtx_idx < n_grid_vertices) {
              // Get the gradient flowing back from this grid vertex
              float grad_vtx = grad_grid[gvtx_idx];

              // Get the weights associated with this corner for point j
              float w_x = grid_pt_weights[j * 24 + k*3 + 0];
              float w_y = grid_pt_weights[j * 24 + k*3 + 1];
              float w_z = grid_pt_weights[j * 24 + k*3 + 2];

              // Determine sign based on which corner it is (relative to pt's floor coords)
              // d(weight_prod) / d(pt_x) = sign_x * w_y * w_z
              // sign_x is -1 if corner_x = floor(pt_x), +1 if corner_x = ceil(pt_x) = floor(pt_x)+1
              // Similarly for y and z.
              int corner_type = k; // 0=LLL, 1=LLU, 2=LUL, 3=LUU, 4=ULL, 5=ULU, 6=UUL, 7=UUU
              float sign_x = (corner_type & 4) ? 1.0f : -1.0f; // Check 3rd bit (Uxx)
              float sign_y = (corner_type & 2) ? 1.0f : -1.0f; // Check 2nd bit (xUx)
              float sign_z = (corner_type & 1) ? 1.0f : -1.0f; // Check 1st bit (xxU)

              // Accumulate gradient contributions
              grad_x += grad_vtx * sign_x * w_y * w_z;
              grad_y += grad_vtx * w_x * sign_y * w_z;
              grad_z += grad_vtx * w_x * w_y * sign_z;
          }
      }
      // Write the final gradient for the point
      // Use atomicAdd if multiple threads might write to the same grad_ptcloud[j*3+c]?
      // No, each thread calculates the full gradient for points j, j+stride, etc. No overlap.
      grad_ptcloud[j * 3 + 0] = grad_x;
      grad_ptcloud[j * 3 + 1] = grad_y;
      grad_ptcloud[j * 3 + 2] = grad_z;
  }
}

// Modify cuda_backward for corrected grid shape
torch::Tensor gridding_distance_cuda_backward(torch::Tensor grid_pt_weights,
                                              torch::Tensor grid_pt_indexes,
                                              torch::Tensor grad_grid, // Expected shape (B, n_grid_vertices)
                                              cudaStream_t stream) {
  int batch_size      = grad_grid.size(0);
  // Check grad_grid dimension (should be 2D now)
  TORCH_CHECK(grad_grid.dim() == 2, "grad_grid is expected to be 2D (B, n_grid_vertices)");
  int n_grid_vertices = grad_grid.size(1);
  int n_pts           = grid_pt_indexes.size(1);
  TORCH_CHECK(grid_pt_weights.size(0) == batch_size && grid_pt_weights.size(1) == n_pts && grid_pt_weights.size(2) == 8 && grid_pt_weights.size(3) == 3, "grid_pt_weights shape mismatch");
  TORCH_CHECK(grid_pt_indexes.size(0) == batch_size && grid_pt_indexes.size(1) == n_pts && grid_pt_indexes.size(2) == 8, "grid_pt_indexes shape mismatch");


  torch::Tensor grad_ptcloud =
    torch::zeros({batch_size, n_pts, 3}, torch::CUDA(torch::kFloat));

  gridding_dist_grad_kernel<<<batch_size, get_n_threads(n_pts), 0, stream>>>(
    n_grid_vertices, n_pts, grid_pt_weights.data_ptr<float>(),
    grid_pt_indexes.data_ptr<int>(), grad_grid.data_ptr<float>(), // Pass corrected grad_grid
    grad_ptcloud.data_ptr<float>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in gridding_distance_cuda_backward: %s\n",
           cudaGetErrorString(err));
  }
  return grad_ptcloud;
}
