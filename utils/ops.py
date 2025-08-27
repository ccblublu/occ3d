import numpy as np
import colorsys
import open3d as o3d

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


def rotate_yaw(yaw):
    if yaw > np.pi:
        yaw -= 2 * np.pi
    elif yaw < -np.pi:
        yaw += 2 * np.pi
    return np.array(
        [[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]],
        dtype=np.float32,
    )




def viz_occ(points, labels, save=True, viz=True, name="tmp", voxel_size=0.4):
    colors = generate_35_category_colors(35)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.colors = o3d.utility.Vector3dVector(colors[labels.astype(int)])
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd, voxel_size=voxel_size
    )
    if viz:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[0, 0, 0]
        )  # create coordinate frame
        vis.add_geometry(mesh_frame)
        vis.add_geometry(voxel_grid)
        vis.run()
    if save:
        o3d.io.write_voxel_grid(f"./viz/{name}.ply", voxel_grid)
    return pcd

def viz_mesh(mesh, gt_box=None, save=True):

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0]
    )  # create coordinate frame
    vis.add_geometry(mesh_frame)
    if isinstance(mesh, list):
        for m in mesh:
            vis.add_geometry(m)
    else:
        vis.add_geometry(mesh)
    if gt_box is not None:
        gt_box[0, 2] += 0.5 * gt_box[0, 5]
        box3d = o3d.geometry.OrientedBoundingBox(
            gt_box[0, 0:3], np.eye(3), gt_box[0, 3:6]
        )
        line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
        vis.add_geometry(line_set)
    vis.run()
    if save:
        o3d.io.write_triangle_mesh(f"./viz/tmp.ply", mesh)
    return


if __name__ == '__main__':
    colors = generate_35_category_colors(35)
    import cv2
    import matplotlib.pyplot as plt
    blank = np.ones((100*36, 100, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        cv2.rectangle(blank, (0, i * 100), (100, i * 100 + 100), (color * 255).astype(np.uint8).tolist(), -1)
        cv2.putText(blank, f'{i}', (10, i * 100 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    blank = cv2.cvtColor(blank, cv2.COLOR_RGB2BGR)
    cv2.imwrite('colors.png', blank)