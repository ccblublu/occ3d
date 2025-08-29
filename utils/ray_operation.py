import numpy as np
from numba import cuda, float32, int32, njit
import math
import torch
import ray_casting_cuda 
import gc

def ray_casting(ray_start, ray_end, pc_range_min, voxel_size, spatial_shape):
    # import time
    # start = time.time()

    ray_starts = torch.from_numpy(np.array(ray_start[..., :3])).to(torch.float32)
    ray_ends = torch.from_numpy(np.array(ray_end[..., :3])).to(torch.float32)
    pc_range_min = torch.from_numpy(np.array(pc_range_min)).to(torch.float32).to('cuda')
    voxel_size = torch.from_numpy(np.array(voxel_size)).to(torch.float32).to('cuda')
    spatial_shape = torch.from_numpy(np.array(spatial_shape)).to(torch.int32).to('cuda')
    max_voxels_per_ray = (spatial_shape).sum().to(torch.int32).to('cuda')
    max_length = 1000000
    output = []
    for i in range(int(np.ceil(ray_starts.shape[0]/max_length))):
        voxel_indices, voxel_nums = ray_casting_cuda.ray_casting_cuda(ray_starts[i*max_length:(i+1)*max_length].to('cuda'), ray_ends[i*max_length:(i+1)*max_length].to('cuda'), pc_range_min, voxel_size, spatial_shape.to(torch.int32), max_voxels_per_ray)
        result = [voxel_index[:voxel_num].cpu().numpy() for voxel_index, voxel_num in zip(voxel_indices, voxel_nums)]
        torch.cuda.empty_cache()
        output.extend(result)
    # print('time0:', time.time()-start)
    # voxel_indices, voxel_nums = ray_casting_cuda.ray_casting_cuda(ray_starts, ray_ends, pc_range_min, voxel_size, spatial_shape, max_voxels_per_ray)
    # start = time.time()
    # pc_range_min_ = np.array([-40.,-40.,-1.], dtype=np.float32)
    # voxel_size_ = np.array([0.4,0.4,0.4], dtype=np.float32)
    # spatial_shape_ = np.array([200,200,16], dtype=np.int32)
    # # for index in range(10000):
    # while True:
    #     index = 0
    #     ray_start_ = ray_start[index, :3].astype(np.float32)
    #     ray_end_ = ray_end[index, :3].astype(np.float32)
    #     ray_casting_(ray_start_, ray_end_, pc_range_min_, voxel_size_, spatial_shape_)
    # # # 返回单个射线的结果
    # # print('time1:', time.time()-start)
    # # output = [voxel_index[:voxel_num] for voxel_index, voxel_num in zip(voxel_indices, voxel_nums)]
    # torch.cuda.empty_cache()
    gc.collect()
    return output

# @njit
def ray_casting_(ray_start, ray_end, pc_range_min, voxel_size, spatial_shape):
    """
    code reproduction of occ3d Algorithm 1: Ray Casting
    不优化要跑一万年
    虽然引入时是点云坐标系，但后续的处理和输出都是在体素坐标系，那在输入的时候，直接按照体素坐标系来计算不就行了，只是相当于一个规则化的range而已

    """
    # ray_start = np.array(ray_start[:3]) #! 点云坐标
    # ray_end = np.array(ray_end[:3]) #! 传感器坐标
    # pc_range_min = np.array(pc_range[3:]) 
    # if not isinstance(voxel_size, list):
    #     voxel_size = [voxel_size, voxel_size, voxel_size]
    # voxel_size = np.array(voxel_size)

    # Adjust ray start and end by subtracting pc_range min values
    new_ray_start = ray_start - pc_range_min #! coords: points
    new_ray_end = ray_end - pc_range_min #! coords: points 

    # Initialize arrays
    ray = new_ray_end - new_ray_start #! coords: points
    step = np.sign(ray).astype(np.int32)  # Use sign function for step direction
    print(step)
    # Handle zero ray components
    ray_nonzero = np.abs(ray) > 1e-12
    tDelta = np.full(3, 1e10)  # Large value for zero-ray components
    tDelta[ray_nonzero] = (step[ray_nonzero] * voxel_size[ray_nonzero]) / ray[ray_nonzero] #? voxel_size 取倒数？

    # Adjust ray start and end with small epsilon
    #! 不包含边界！
    eps = 1e-9
    new_ray_start += step * voxel_size * eps
    new_ray_end -= step * voxel_size * eps

    # Calculate current and last voxel indices
    print(new_ray_start, new_ray_end)

    cur_voxel = np.floor(new_ray_start.astype(np.float32) / voxel_size).astype(np.int32)
    last_voxel = np.floor(new_ray_end.astype(np.float32) / voxel_size).astype(np.int32)
    print(cur_voxel, last_voxel)

    # Initialize tMax values
    tMax = np.full(3, 1e10)
    free_voxel = []
    for k in range(3):
        if ray_nonzero[k]:
            cur_coordinate = cur_voxel[k] * voxel_size[k]
            if step[k] < 0 and cur_coordinate < new_ray_start[k]:
                tMax[k] = cur_coordinate
            else:
                tMax[k] = cur_coordinate + voxel_size[k] * step[k]
            tMax[k] = (tMax[k] - new_ray_start[k]) / ray[k]
    while np.any(step * (last_voxel - cur_voxel) > 0.5):#! 沿一个轴走完
        # Find the axis with the minimum tMax
        min_axis = np.argmin(tMax)

        # Update voxel index for the selected axis
        cur_voxel[min_axis] += step[min_axis]

        # Check if the new voxel is within bounds
        if cur_voxel[min_axis] < 0 or cur_voxel[min_axis] >= spatial_shape[min_axis]:
            break

        # Update tMax for the selected axis
        tMax[min_axis] += tDelta[min_axis]

        # Yield the current voxel
        free_voxel.append(cur_voxel.copy())

    return free_voxel