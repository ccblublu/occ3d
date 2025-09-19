import os
import argparse
from pathlib import Path
import time
import colorsys
import cv2
import torch
import numpy as np
import mayavi.mlab as mlab

# colors = np.array(
#     [
#         [0, 0, 0, 255],
#         [255, 120, 50, 255],  # barrier              orangey
#         [255, 192, 203, 255],  # bicycle              pink
#         [255, 255, 0, 255],  # bus                  yellow
#         [0, 150, 245, 255],  # car                  blue
#         [0, 255, 255, 255],  # construction_vehicle cyan
#         [200, 180, 0, 255],  # motorcycle           dark orange
#         [255, 0, 0, 255],  # pedestrian           red
#         [255, 240, 150, 255],  # traffic_cone         light yellow
#         [135, 60, 0, 255],  # trailer              brown
#         [160, 32, 240, 255],  # truck                purple
#         [255, 0, 255, 255],  # driveable_surface    dark pink
#         # [175,   0,  75, 255],       # other_flat           dark red
#         [139, 137, 137, 255],
#         [75, 0, 75, 255],  # sidewalk             dard purple
#         [150, 240, 80, 255],  # terrain              light green
#         [230, 230, 250, 255],  # manmade              white
#         [0, 175, 0, 255],  # vegetation           green
#         [0, 255, 127, 255],  # ego car              dark cyan
#         [255, 99, 71, 255],
#         [0, 191, 255, 255]
#     ]
# ).astype(np.uint8)

colors = np.array(
    [
        [0, 0, 0, 255],
        [0, 150, 245, 255],  # car                  blue
        [160, 32, 240, 255],  # truck                purple
        [255, 255, 0, 255],  # bus                  yellow
        [0, 191, 255, 255],
        [200, 180, 0, 255],  # motorcycle           dark orange
        [255, 192, 203, 255],  # bicycle              pink
        [0, 255, 127, 255],  # ego car              dark cyan
        [255, 0, 0, 255],  # pedestrian           red
        [230, 230, 250, 255],  # manmade              white
        [0, 255, 255, 255],  # construction_vehicle cyan
        [255, 240, 150, 255],  # traffic_cone         light yellow
        [135, 60, 0, 255],  # trailer              brown
        [139, 137, 137, 255],
        [139, 137, 137, 255],
        [75, 0, 75, 255],  # sidewalk             dard purple
        [175,   0,  75, 255],       # other_flat           dark red
        [255, 120, 50, 255],  # barrier              orangey
        [255, 0, 255, 255],  # driveable_surface    dark pink
        # [0, 175, 0, 255],  # vegetation           green
        [150, 240, 80, 255],  # terrain              light green
    ]
).astype(np.uint8)
threshold = 0.65
voxel_size = 0.4
pc_range = [-40, -40, -1, 40, 40, 5.4]
X, Y, Z = 200, 200, 16


def gridcloud3d(B, Z, Y, X, device='cpu'):
    # we want to sample for each location in the grid
    grid_z, grid_y, grid_x = meshgrid3d(B, Z, Y, X, device=device)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    z = torch.reshape(grid_z, [B, -1])

    # pdb.set_trace()
    # these are B x N
    xyz = torch.stack([x, y, z], dim=2)
    # here is stack in order with xyz
    # this is B x N x 3

    # pdb.set_trace()
    return xyz

def meshgrid3d(B, Z, Y, X, stack=False, device='cuda'):
    # returns a meshgrid sized B x Z x Y x X

    grid_z = torch.linspace(0.0, Z-1, Z, device=device)
    grid_z = torch.reshape(grid_z, [1, Z, 1, 1])
    grid_z = grid_z.repeat(B, 1, Y, X)

    grid_y = torch.linspace(0.0, Y-1, Y, device=device)
    grid_y = torch.reshape(grid_y, [1, 1, Y, 1])
    grid_y = grid_y.repeat(B, Z, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=device)
    grid_x = torch.reshape(grid_x, [1, 1, 1, X])
    grid_x = grid_x.repeat(B, Z, Y, 1)
    # here repeat is in the order with ZYX

    if stack:
        # note we stack in xyz order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid
    else:
        return grid_z, grid_y, grid_x


def visualize_occ_dict(output_dict, offscreen=True, render_w=1600):
    mlab.options.offscreen = offscreen
    for i, file in enumerate(Path(output_dict).glob("*/labels.npz")):
        data_dict = np.load(file, allow_pickle=True)
        xyz = gridcloud3d(1, Z, Y, X, device='cpu')
        xyz_min = np.array(pc_range[:3])
        xyz_max = np.array(pc_range[3:])
        occ_size = np.array([X, Y, Z])
        xyz = xyz / occ_size * (xyz_max - xyz_min) + xyz_min + 0.5 * voxel_size
        xyz = xyz.reshape(Z, Y, X, 3).permute(2, 1, 0, 3).numpy()
        occs = data_dict['semantics']
        mask_lidar = data_dict['mask_lidar'].astype(bool)
        mask_camera = data_dict['mask_camera'].astype(bool)
        occ_mask = occs[mask_camera] != 17
        xyz_class = np.concatenate(
            [xyz[mask_camera][occ_mask], occs[mask_camera][occ_mask][:, None]],
            axis=1)

        fov_voxels = xyz_class
        # cam_positions, focal_positions = [], []
        if i == 0:
            figure = mlab.figure(size=(render_w, render_w / 16 * 9), bgcolor=(1, 1, 1))

            plt_plot_fov = mlab.points3d(
                fov_voxels[:, 0],
                fov_voxels[:, 1],
                fov_voxels[:, 2],
                fov_voxels[:, 3],
                colormap="viridis",
                scale_factor=voxel_size - 0.05 * voxel_size,
                mode="cube",
                opacity=1.0,
                vmin=0,
                vmax=19,
            )
            plt_plot_fov.glyph.scale_mode = "scale_by_vector"
            plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
        mlab.view(azimuth=180, elevation=60, distance=70, focalpoint=(0,0,0) )

        plt_plot_fov.mlab_source.reset(x=fov_voxels[:, 0], y=fov_voxels[:, 1], z=fov_voxels[:, 2], scalars=fov_voxels[:, 3])
        mlab.draw()
        mlab.process_ui_events()
        time.sleep(0.2)
    mlab.show()


    # scene = figure.scene
    # pos = scene.camera.position
    # pos[2] -= 50  # 向下移动 2 个单位
    # scene.camera.position = pos
    # mlab.view(azimuth=180, elevation=60, distance=70, focalpoint=(0,0,0) )
    # az, el, dist, fpt = mlab.view()
    # print(f"当前视角：方位角={az}, 仰角={el}, 距离={dist}, 焦点={fpt}")

    # mlab.show()
def generate_35_category_colors(num):
    """
    生成35种类别的归一化三色数组(RGB格式)
    
    返回:
        colors (np.ndarray): 形状为(35, 3)的数组，每行是一个归一化的RGB颜色
    """
    # 使用HSV色彩空间生成更均匀分布的颜色
    colors = []
    
    # 生成不同色调的颜色 (0-1范围)
    for i in range(num):
        # 使用黄金角分布，确保颜色尽可能均匀分布
        hue = (i * 0.618033988749895) % 1.0  # 黄金角近似值
        
        # 固定饱和度和亮度，确保颜色清晰可辨
        saturation = 0.8
        value = 0.9
        
        # 转换为RGB
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append([r, g, b])
    
    # 转换为numpy数组
    colors = np.array(colors)
    
    # 确保所有值在[0,1]范围内
    colors = np.clip(colors, 0.0, 1.0)
    
    return colors
# print(generate_35_category_colors(35)[0])


if __name__=='__main__':
    # gtfile_path = '/media/chen/data/OpenOcc/Occpancy3D/gts/scene-0061/7626dde27d604ac28a0240bdd54eba7a/labels.npz'
    # gtfile_path = '/media/chen/data/OpenOcc/Occpancy3D/gts/scene-0061'
    # visualize_occ_dict(gtfile_path, offscreen=False)
    render_w = 1600
    # from utils.ops import generate_35_category_colors
    # colors = generate_35_category_colors(35) * 255
    # root = Path("/home/chen/workspace/occ/Occ3D/runs/debug")
    # for i, file in enumerate(sorted(root.glob("*.npy"))):
    #     points = np.load(file)
    #     points = points.astype(np.float32)
    #     points[:,:3] = (points[:, :3] + 0.5 ) * voxel_size + np.array(pc_range[:3])
    #     # voxel_size=1
    #     print(points[:,:3].min(0), points[:, :3].max(0))
    #     print(points[:,3].min(0), points[:, 3].max(0))
    #     if i == 0:
    #         figure = mlab.figure(size=(render_w, render_w / 16 * 9), bgcolor=(1, 1, 1))

    #         plt_plot_fov = mlab.points3d(
    #             points[:, 0],
    #             points[:, 1],
    #             points[:, 2],
    #             points[:, 3],
    #             colormap="viridis",
    #             scale_factor=voxel_size - 0.05 * voxel_size,
    #             mode="cube",
    #             opacity=1.0,
    #             vmin=0,
    #             vmax=19,
    #         )
    #         plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    #         plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
    #     mlab.view(azimuth=180, elevation=70, distance=70, focalpoint=(0,0,-7) )
    #     # az, el, dist, fpt = mlab.view()
    #     # print(f"当前视角：方位角={az}, 仰角={el}, 距离={dist}, 焦点={fpt}")

    #     plt_plot_fov.mlab_source.reset(x=points[:, 0], y=points[:, 1], z=points[:, 2], scalars=points[:, 3])
    #     mlab.draw()
    #     mlab.process_ui_events()
    #     # mlab.show()
    #     mlab.savefig(f"viz/occ-{i}.png")
    #     # time.sleep(100000)
    #     # break
    # mlab.show()
    pos = {"front_narrow": [0, 1920], "front_wide": [0, 1920*2], "left_front": [1080, 0], "left_back": [1080*2, 0], "right_front": [1080, 1920*3], "right_back": [1080*2, 1920*3], "back": [1080*3, 1920*1.5]}
    seg_root = Path("/media/chen/090/train_data/GACRT014_1729079698/raw_data/seg_pred")
    root = Path("/home/chen/workspace/occ/Occ3D/viz")
    h = int(1080 * 4)
    w = int(1920 * 4)
    for i in range(40):
        file = root / f"occ-{i}.png"
        image = cv2.imread(str(file))
        image = cv2.resize(image, (1920*2, 1080*2))
        blank = np.full((h, w, 3), 150, dtype=np.uint8)
        blank[1080:1080*3, 1920:1920*3, :] = image
        for k, v in pos.items():
            v = np.array(v, dtype=int)
            file = list((seg_root / k).glob(f"*_{i}.jpg"))[0]
            image = cv2.imread(str(file))
            image = cv2.resize(image, (1920, 1080))
            blank[v[0]:v[0]+1080, v[1]:v[1]+1920, :] = image

        # h, w, c = image.shape
        if i == 0:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 格式
            video = cv2.VideoWriter("./viz/occ.mp4", fourcc, 2, (w, h))
        video.write(blank)
        # break
    video.release()
