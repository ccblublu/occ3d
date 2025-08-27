#pragma once
#include <vector>
#include <cuda_runtime.h>

void ray_casting_cuda_launcher(
    const float* ray_start,     // [B, 3]
    const float* ray_end,       // [B, 3]
    const float* pc_range_min,  // [3]
    const float* voxel_size,    // [3]
    const int* spatial_shape,   // [3]
    int max_voxels_per_ray,     // 每条射线最多穿过多少体素
    int* voxel_indices,         // [B, max_voxels_per_ray, 3] 输出
    int* voxel_nums,            // [B] 每条射线实际穿过体素数
    int batch_size,             // B
    cudaStream_t stream
);
