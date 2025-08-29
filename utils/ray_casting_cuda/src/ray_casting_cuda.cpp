#include "ray_casting_cuda_kernel.h"
#include <vector>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

std::vector<at::Tensor> ray_casting_cuda(
    at::Tensor ray_start,      // [B, 3], float32
    at::Tensor ray_end,        // [B, 3], float32
    at::Tensor pc_range_min,   // [3], float32
    at::Tensor voxel_size,     // [3], float32
    at::Tensor spatial_shape,  // [3], int32
    int max_voxels_per_ray
) {
    int B = ray_start.size(0);
    // printf("B: %d\n", B);
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(ray_start.device());
    auto voxel_indices = torch::zeros({B, max_voxels_per_ray, 3}, options_int);
    auto voxel_nums = torch::zeros({B}, options_int);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ray_casting_cuda_launcher(
        ray_start.data_ptr<float>(),
        ray_end.data_ptr<float>(),
        pc_range_min.data_ptr<float>(),
        voxel_size.data_ptr<float>(),
        spatial_shape.data_ptr<int>(),
        max_voxels_per_ray,
        voxel_indices.data_ptr<int>(),
        voxel_nums.data_ptr<int>(),
        B,
        stream
    );
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA stream synchronization failed");
    }
    return {voxel_indices, voxel_nums};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ray_casting_cuda", &ray_casting_cuda, "Ray Casting CUDA (batch)");
}
