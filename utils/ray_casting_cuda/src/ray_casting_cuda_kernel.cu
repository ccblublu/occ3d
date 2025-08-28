#include "ray_casting_cuda_kernel.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cfloat>

__device__ inline float signf(float x) {
    return (x > 0) - (x < 0);
}

__global__ void ray_casting_kernel(
    const float* ray_start,     // [B, 3]
    const float* ray_end,       // [B, 3]
    const float* pc_range_min,  // [3]
    const float* voxel_size,    // [3]
    const int* spatial_shape,   // [3]
    int max_voxels_per_ray,
    int* voxel_indices,         // [B, max_voxels_per_ray, 3]
    int* voxel_nums,            // [B]
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Load input
    float start[3], end[3], range_min[3], vsize[3];
    for (int i = 0; i < 3; ++i) {
        start[i] = ray_start[idx * 3 + i];
        end[i] = ray_end[idx * 3 + i];
        range_min[i] = pc_range_min[i];
        vsize[i] = voxel_size[i];
    }
    int shape[3] = {spatial_shape[0], spatial_shape[1], spatial_shape[2]};

    // Compute in voxel coordinate
    float new_start[3], new_end[3], ray[3], step[3], tDelta[3], tMax[3];
    bool ray_nonzero[3];
    for (int i = 0; i < 3; ++i) {
        new_start[i] = start[i] - range_min[i];
        new_end[i] = end[i] - range_min[i];
        ray[i] = new_end[i] - new_start[i];
        step[i] = signf(ray[i]);
        ray_nonzero[i] = fabs(ray[i]) > 1e-6f;
        tDelta[i] = ray_nonzero[i] ? (step[i] * vsize[i]) / ray[i] : FLT_MAX;
    }
    // if (idx == 0) printf("ray_nonzero: %d %d %d\n",
            // ray_nonzero[0], ray_nonzero[1], ray_nonzero[2]);
    // Epsilon adjustment
    float eps = 1e-9f;
    for (int i = 0; i < 3; ++i) {
        new_start[i] += step[i] * vsize[i] * eps;
        new_end[i] -= step[i] * vsize[i] * eps;
    }
    // if (idx == 0) printf("new_start: %f %f %f, new_end: %f %f %f\n",
            // new_start[0], new_start[1], new_start[2], new_end[0], new_end[1], new_end[2]);
    // Current and last voxel
    int cur_voxel[3], last_voxel[3];
    for (int i = 0; i < 3; ++i) {
        cur_voxel[i] = floorf(new_start[i] / vsize[i]);
        last_voxel[i] = floorf(new_end[i] / vsize[i]);
    }
    // if (idx == 0) printf("cur_voxel: %d %d %d, last_voxel: %d %d %d\n",
            // cur_voxel[0], cur_voxel[1], cur_voxel[2], last_voxel[0], last_voxel[1], last_voxel[2]);

    // tMax init
    for (int i = 0; i < 3; ++i) {
        tMax[i] = FLT_MAX;
        if (ray_nonzero[i]) {
            float cur_coordinate = cur_voxel[i] * vsize[i];
            if (step[i] < 0 && cur_coordinate < new_start[i])
                tMax[i] = cur_coordinate;
            else
                tMax[i] = cur_coordinate + vsize[i] * step[i];
            tMax[i] = (tMax[i] - new_start[i]) / ray[i];
        }
    }

    int voxel_count = 0;
    bool valid = true;
    while (valid) {
        // if (idx == 0) printf("cur_voxel: %d %d %d, last_voxel: %d %d %d\n",
            // cur_voxel[0], cur_voxel[1], cur_voxel[2], last_voxel[0], last_voxel[1], last_voxel[2]);
        // printf("cur_voxel: %d %d %d\n", cur_voxel[0], cur_voxel[1], cur_voxel[2]);
        // Find min tMax axis
        int min_axis = 0;
        float min_val = tMax[0];
        for (int k = 1; k < 3; ++k)
            if (tMax[k] < min_val) {
                min_axis = k;
                min_val = tMax[k];
            }

        // Update voxel
        cur_voxel[min_axis] += (int)step[min_axis];

        // Check bounds
        if (cur_voxel[min_axis] < 0 || cur_voxel[min_axis] >= shape[min_axis])
            break;

        // Save voxel
        if (voxel_count < max_voxels_per_ray) {
            int base = idx * max_voxels_per_ray * 3 + voxel_count * 3;
            voxel_indices[base + 0] = cur_voxel[0];
            voxel_indices[base + 1] = cur_voxel[1];
            voxel_indices[base + 2] = cur_voxel[2];
            voxel_count++;
        }

        // Update tMax
        tMax[min_axis] += tDelta[min_axis];

        // Check finish
        valid = false;
        for (int k = 0; k < 3; ++k)
            if (step[k] * (last_voxel[k] - cur_voxel[k]) > 0.5f)
                valid = true;
    }
    voxel_nums[idx] = voxel_count;
}

void ray_casting_cuda_launcher(
    const float* ray_start,
    const float* ray_end,
    const float* pc_range_min,
    const float* voxel_size,
    const int* spatial_shape,
    int max_voxels_per_ray,
    int* voxel_indices,
    int* voxel_nums,
    int batch_size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    ray_casting_kernel<<<blocks, threads, 0, stream>>>(
        ray_start, ray_end, pc_range_min, voxel_size, spatial_shape,
        max_voxels_per_ray, voxel_indices, voxel_nums, batch_size
    );
}
