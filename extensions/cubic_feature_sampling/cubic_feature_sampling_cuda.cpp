/*
 * @Author: Haozhe Xie
 * @Date:   2019-12-19 17:04:38
 * @Last Modified by:   Haozhe Xie
 * @Last Modified time: 2020-06-17 14:50:22
 * @Email:  cshzxie@gmail.com
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector> // Add include

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor") // Use TORCH_CHECK
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous") // Use TORCH_CHECK
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// Modify forward signature
std::vector<torch::Tensor> cubic_feature_sampling_cuda_forward(
  int scale_x, int scale_y, int scale_z, // Use scales
  int neighborhood_size,
  torch::Tensor ptcloud,
  torch::Tensor cubic_features,
  cudaStream_t stream);

// Modify backward signature
std::vector<torch::Tensor> cubic_feature_sampling_cuda_backward(
  int scale_x, int scale_y, int scale_z, // Use scales
  int neighborhood_size,
  torch::Tensor grad_point_features,
  torch::Tensor grid_pt_indexes,
  cudaStream_t stream);

// Modify forward wrapper signature
std::vector<torch::Tensor> cubic_feature_sampling_forward(
  int scale_x, int scale_y, int scale_z, // Use scales
  int neighborhood_size,
  torch::Tensor ptcloud,
  torch::Tensor cubic_features) {
  CHECK_INPUT(ptcloud);
  CHECK_INPUT(cubic_features);
  // Add check for cubic_features dimension
  TORCH_CHECK(cubic_features.dim() == 5, "cubic_features is expected to be 5D (B, C, Dx, Dy, Dz)");
  TORCH_CHECK(cubic_features.size(2) == scale_x && cubic_features.size(3) == scale_y && cubic_features.size(4) == scale_z, "cubic_features dimensions mismatch scales");


  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  return cubic_feature_sampling_cuda_forward(scale_x, scale_y, scale_z, neighborhood_size, ptcloud,
                                             cubic_features, stream);
}

// Modify backward wrapper signature
std::vector<torch::Tensor> cubic_feature_sampling_backward(
  int scale_x, int scale_y, int scale_z, // Use scales
  int neighborhood_size,
  torch::Tensor grad_point_features,
  torch::Tensor grid_pt_indexes) {
  CHECK_INPUT(grad_point_features);
  CHECK_INPUT(grid_pt_indexes);
   // Add check for grad_point_features dimension
  TORCH_CHECK(grad_point_features.dim() == 4, "grad_point_features is expected to be 4D (B, N_pts, N_vertices, C)");
  TORCH_CHECK(grid_pt_indexes.dim() == 3, "grid_pt_indexes is expected to be 3D (B, N_pts, N_vertices)");


  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  return cubic_feature_sampling_cuda_backward(
    scale_x, scale_y, scale_z, neighborhood_size, grad_point_features, grid_pt_indexes, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Update bindings
  m.def("forward", &cubic_feature_sampling_forward,
        "Cubic Feature Sampling forward (CUDA)");
  m.def("backward", &cubic_feature_sampling_backward,
        "Cubic Feature Sampling backward (CUDA)");
}