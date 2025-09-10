#include "ray_casting_cuda_kernel.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <stdio.h>
#include <cfloat>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

// 使用更快的符号函数
__device__ __forceinline__ float fast_signf(float x) {
    return __float_as_int(x) >> 31 ? -1.0f : (x > 0.0f ? 1.0f : 0.0f);
}
struct bool3 {
    bool x, y, z;
};


// 完整的动态并行度版本（适用于不均匀工作负载）
__global__ void ray_casting_kernel_dynamic(
    const float* __restrict__ ray_start,     // [B, 3]
    const float* __restrict__ ray_end,       // [B, 3]
    const float* __restrict__ pc_range_min,  // [3]
    const float* __restrict__ voxel_size,    // [3]
    const int* __restrict__ spatial_shape,   // [3]
    int max_voxels_per_ray,
    int* __restrict__ voxel_indices,         // [B, max_voxels_per_ray, 3]
    int* __restrict__ voxel_nums,            // [B]
    int batch_size,
    int* __restrict__ work_queue,            // [batch_size] 工作队列
    int* __restrict__ queue_counter          // [1] 队列计数器
) {
    // 使用共享内存存储常量数据和工作队列管理
    __shared__ float s_range_min[3];
    __shared__ float s_voxel_size[3];
    __shared__ int s_spatial_shape[3];
    __shared__ int s_local_queue_start;
    __shared__ int s_threads_working;
    
    // Block内的线程ID
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // 初始化共享内存
    if (tid == 0) {
        // 加载常量数据到共享内存
        for (int i = 0; i < 3; ++i) {
            s_range_min[i] = pc_range_min[i];
            s_voxel_size[i] = voxel_size[i];
            s_spatial_shape[i] = spatial_shape[i];
        }
        
        // 原子操作获取这个block的工作范围
        s_local_queue_start = atomicAdd(queue_counter, block_size);
        s_threads_working = min(block_size, batch_size - s_local_queue_start);
    }
    __syncthreads();
    
    // 检查当前线程是否有工作
    if (tid >= s_threads_working) {
        return;
    }
    
    // 从工作队列获取实际的ray索引
    int queue_idx = s_local_queue_start + tid;
    if (queue_idx >= batch_size) {
        return;
    }
    
    int idx = work_queue[queue_idx];
    if (idx >= batch_size) {
        return;
    }

    // 使用寄存器存储频繁访问的数据
    float3 start, end, ray, new_start, new_end;
    float3 step, tDelta, tMax, vsize, range_min;
    int3 cur_voxel, last_voxel, shape;
    bool3 ray_nonzero;
    
    // 向量化加载射线起点和终点
    const float3 *start_ptr = (const float3*)(ray_start + idx * 3);
    const float3 *end_ptr = (const float3*)(ray_end + idx * 3);
    start = *start_ptr;
    end = *end_ptr;
    
    // 从共享内存加载常量数据
    range_min = make_float3(s_range_min[0], s_range_min[1], s_range_min[2]);
    vsize = make_float3(s_voxel_size[0], s_voxel_size[1], s_voxel_size[2]);
    shape = make_int3(s_spatial_shape[0], s_spatial_shape[1], s_spatial_shape[2]);

    // 将射线转换到体素坐标系
    new_start = make_float3(start.x - range_min.x, start.y - range_min.y, start.z - range_min.z);
    new_end = make_float3(end.x - range_min.x, end.y - range_min.y, end.z - range_min.z);
    ray = make_float3(new_end.x - new_start.x, new_end.y - new_start.y, new_end.z - new_start.z);
    
    // 计算步长方向
    step.x = fast_signf(ray.x);
    step.y = fast_signf(ray.y);
    step.z = fast_signf(ray.z);
    
    // 检查射线在各轴上是否为零
    const float epsilon = 1e-6f;
    ray_nonzero.x = fabsf(ray.x) > epsilon;
    ray_nonzero.y = fabsf(ray.y) > epsilon;
    ray_nonzero.z = fabsf(ray.z) > epsilon;
    
    // 计算tDelta - 沿射线移动一个体素所需的参数增量
    tDelta.x = ray_nonzero.x ? fabsf(vsize.x / ray.x) : FLT_MAX;
    tDelta.y = ray_nonzero.y ? fabsf(vsize.y / ray.y) : FLT_MAX;
    tDelta.z = ray_nonzero.z ? fabsf(vsize.z / ray.z) : FLT_MAX;

    // 添加epsilon调整避免数值误差
    const float eps = 1e-9f;
    new_start.x += step.x * vsize.x * eps;
    new_start.y += step.y * vsize.y * eps;
    new_start.z += step.z * vsize.z * eps;
    new_end.x -= step.x * vsize.x * eps;
    new_end.y -= step.y * vsize.y * eps;
    new_end.z -= step.z * vsize.z * eps;

    // 计算当前体素和最后体素的索引
    cur_voxel.x = __float2int_rd(new_start.x / vsize.x);
    cur_voxel.y = __float2int_rd(new_start.y / vsize.y);
    cur_voxel.z = __float2int_rd(new_start.z / vsize.z);
    
    last_voxel.x = __float2int_rd(new_end.x / vsize.x);
    last_voxel.y = __float2int_rd(new_end.y / vsize.y);
    last_voxel.z = __float2int_rd(new_end.z / vsize.z);

    // 初始化tMax - 到达下一个体素边界的参数值
    if (ray_nonzero.x) {
        float cur_coord = cur_voxel.x * vsize.x;
        float next_boundary = (step.x > 0) ? (cur_coord + vsize.x) : cur_coord;
        if (step.x < 0 && cur_coord < new_start.x) {
            next_boundary = cur_coord;
        }
        tMax.x = (next_boundary - new_start.x) / ray.x;
    } else {
        tMax.x = FLT_MAX;
    }
    
    if (ray_nonzero.y) {
        float cur_coord = cur_voxel.y * vsize.y;
        float next_boundary = (step.y > 0) ? (cur_coord + vsize.y) : cur_coord;
        if (step.y < 0 && cur_coord < new_start.y) {
            next_boundary = cur_coord;
        }
        tMax.y = (next_boundary - new_start.y) / ray.y;
    } else {
        tMax.y = FLT_MAX;
    }
    
    if (ray_nonzero.z) {
        float cur_coord = cur_voxel.z * vsize.z;
        float next_boundary = (step.z > 0) ? (cur_coord + vsize.z) : cur_coord;
        if (step.z < 0 && cur_coord < new_start.z) {
            next_boundary = cur_coord;
        }
        tMax.z = (next_boundary - new_start.z) / ray.z;
    } else {
        tMax.z = FLT_MAX;
    }

    // 主要的DDA遍历循环
    int voxel_count = 0;
    int base_idx = idx * max_voxels_per_ray * 3;
    
    // 先保存起始体素（如果在边界内）
    if (cur_voxel.x >= 0 && cur_voxel.x < shape.x &&
        cur_voxel.y >= 0 && cur_voxel.y < shape.y &&
        cur_voxel.z >= 0 && cur_voxel.z < shape.z &&
        voxel_count < max_voxels_per_ray) {
        
        voxel_indices[base_idx + voxel_count * 3 + 0] = cur_voxel.x;
        voxel_indices[base_idx + voxel_count * 3 + 1] = cur_voxel.y;
        voxel_indices[base_idx + voxel_count * 3 + 2] = cur_voxel.z;
        voxel_count++;
    }
    
    // DDA主循环
    while (voxel_count < max_voxels_per_ray) {
        // 找到tMax最小的轴（下一个要跨越的体素边界）
        int min_axis;
        
        if (tMax.x <= tMax.y && tMax.x <= tMax.z) {
            min_axis = 0;
        } else if (tMax.y <= tMax.z) {
            min_axis = 1;
        } else {
            min_axis = 2;
        }

        // 沿最小tMax的轴移动到下一个体素
        if (min_axis == 0) {
            cur_voxel.x += (int)step.x;
            // 检查是否超出边界
            if (cur_voxel.x < 0 || cur_voxel.x >= shape.x) break;
            tMax.x += tDelta.x;
        } else if (min_axis == 1) {
            cur_voxel.y += (int)step.y;
            if (cur_voxel.y < 0 || cur_voxel.y >= shape.y) break;
            tMax.y += tDelta.y;
        } else {
            cur_voxel.z += (int)step.z;
            if (cur_voxel.z < 0 || cur_voxel.z >= shape.z) break;
            tMax.z += tDelta.z;
        }

        // 检查是否仍在其他轴的边界内
        if (cur_voxel.x < 0 || cur_voxel.x >= shape.x ||
            cur_voxel.y < 0 || cur_voxel.y >= shape.y ||
            cur_voxel.z < 0 || cur_voxel.z >= shape.z) {
            break;
        }

        // 保存当前体素
        voxel_indices[base_idx + voxel_count * 3 + 0] = cur_voxel.x;
        voxel_indices[base_idx + voxel_count * 3 + 1] = cur_voxel.y;
        voxel_indices[base_idx + voxel_count * 3 + 2] = cur_voxel.z;
        voxel_count++;

        // 检查是否到达终点体素
        bool reached_end = true;
        if (step.x != 0 && step.x * (last_voxel.x - cur_voxel.x) > 0) reached_end = false;
        if (step.y != 0 && step.y * (last_voxel.y - cur_voxel.y) > 0) reached_end = false;
        if (step.z != 0 && step.z * (last_voxel.z - cur_voxel.z) > 0) reached_end = false;
        
        if (reached_end) break;
    }
    
    // 保存该射线遍历的体素数量
    voxel_nums[idx] = voxel_count;
}

// 动态版本的启动器
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
    cudaStream_t stream = 0
) {
    // 分配工作队列和计数器
    int* d_work_queue;
    int* d_queue_counter;
    
    cudaMalloc(&d_work_queue, batch_size * sizeof(int));
    cudaMalloc(&d_queue_counter, sizeof(int));
    
    // 初始化工作队列：简单按顺序排列（可以根据需要预排序）
    thrust::device_ptr<int> queue_ptr(d_work_queue);
    thrust::sequence(thrust::cuda::par.on(stream), queue_ptr, queue_ptr + batch_size);
    
    // 初始化队列计数器为0
    cudaMemsetAsync(d_queue_counter, 0, sizeof(int), stream);
    
    // 获取设备属性以优化线程配置
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // 根据设备能力选择线程配置
    int threads = (prop.major >= 7) ? 512 : 256;
    int blocks = (batch_size + threads - 1) / threads;
    
    // 限制block数量避免过度创建
    int max_blocks = prop.multiProcessorCount * 4;  // 每个SM 4个block
    blocks = min(blocks, max_blocks);
    
    // 启动kernel
    ray_casting_kernel_dynamic<<<blocks, threads, 0, stream>>>(
        ray_start, ray_end, pc_range_min, voxel_size, spatial_shape,
        max_voxels_per_ray, voxel_indices, voxel_nums, batch_size,
        d_work_queue, d_queue_counter
    );
    
    // 检查错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(error));
    }
    
    // 清理临时内存
    cudaFree(d_work_queue);
    cudaFree(d_queue_counter);
}

// 辅助函数：创建预排序的工作队列（可选优化）
void create_sorted_work_queue(
    const float* ray_start,
    const float* ray_end,
    int batch_size,
    int* work_queue,
    cudaStream_t stream = 0
) {
    // 这里可以根据射线长度等特征对工作进行预排序
    // 让相似复杂度的任务在同一个warp中执行
    // 简单实现：直接按顺序排列
    thrust::device_ptr<int> queue_ptr(work_queue);
    thrust::sequence(thrust::cuda::par.on(stream), queue_ptr, queue_ptr + batch_size);
    
    // TODO: 可以添加更复杂的排序逻辑
    // 例如：按射线长度排序，按起始位置排序等
}

// 高级版本：支持预排序工作队列的启动器
void ray_casting_cuda_launcher_dynamic_advanced(
    const float* ray_start,
    const float* ray_end,
    const float* pc_range_min,
    const float* voxel_size,
    const int* spatial_shape,
    int max_voxels_per_ray,
    int* voxel_indices,
    int* voxel_nums,
    int batch_size,
    bool use_sorted_queue = false,
    cudaStream_t stream = 0
) {
    int* d_work_queue;
    int* d_queue_counter;
    
    cudaMalloc(&d_work_queue, batch_size * sizeof(int));
    cudaMalloc(&d_queue_counter, sizeof(int));
    
    if (use_sorted_queue) {
        create_sorted_work_queue(ray_start, ray_end, batch_size, d_work_queue, stream);
    } else {
        thrust::device_ptr<int> queue_ptr(d_work_queue);
        thrust::sequence(thrust::cuda::par.on(stream), queue_ptr, queue_ptr + batch_size);
    }
    
    cudaMemsetAsync(d_queue_counter, 0, sizeof(int), stream);
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    int threads = (prop.major >= 7) ? 512 : 256;
    int blocks = (batch_size + threads - 1) / threads;
    int max_blocks = prop.multiProcessorCount * 4;
    blocks = min(blocks, max_blocks);
    
    ray_casting_kernel_dynamic<<<blocks, threads, 0, stream>>>(
        ray_start, ray_end, pc_range_min, voxel_size, spatial_shape,
        max_voxels_per_ray, voxel_indices, voxel_nums, batch_size,
        d_work_queue, d_queue_counter
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(error));
    }
    
    cudaFree(d_work_queue);
    cudaFree(d_queue_counter);
}