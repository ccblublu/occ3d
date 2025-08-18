from collections import defaultdict
from os import path as osp
import pickle
from tkinter import _flatten
import pyquaternion
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from functools import partial


import vdbfusion
import open3d as o3d
import numpy as np
from torch.utils.data import Dataset
from nuscenes import NuScenes
from mmcv import load as mmcv_load
from mmdet3d.datasets import build_dataset
from mmcv import track_iter_progress
from mmdet3d.datasets import DATASETS
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmcv.utils import Registry, build_from_cfg
from mmdet3d.core.bbox import get_box_type
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.core.bbox import box_np_ops
import pypatchworkpp

from ops.mmdetection3d.tools.misc.browse_dataset import show_det_data, show_result
# @DATASETS.register_module()


def rotate_yaw(yaw):
    if yaw > np.pi:
        yaw -= 2 * np.pi
    elif yaw < -np.pi:
        yaw += 2 * np.pi
    return np.array(
        [[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]],
        dtype=np.float32,
    )


class LiDARInstance3DBoxeswithID(LiDARInstance3DBoxes):
    def __init__(self, tensor, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0), ids=None):
        super().__init__(tensor, box_dim, with_yaw, origin)
        self.ids = ids

class SeqNuscene(Dataset):
    def __init__(self, data_root, anno_file, nuscenes_version="v1.0-trainval"):
        self.data_root = data_root
        self.anno_file = anno_file
        nuscenes_version = nuscenes_version
        # dataroot = './data/nuscenes/'
        self.nuscenes = NuScenes(nuscenes_version, data_root)
        anno_data = mmcv_load(osp.join(data_root, anno_file))
        self.anno_metadata = anno_data['metadata']
        self.anno_infos = anno_data['infos']
        self.seq_data = self.get_seq_data()


    def get_seq_data(self):
        seq_data = defaultdict(list)
        for id, info in enumerate(self.anno_infos):
            token = info['token']
            sample = self.nuscenes.get('sample', token)
            seq_data[sample['scene_token']].append(info)
        seq_data = dict(seq_data)
        out = []
        for key, value in seq_data.items():
            # seq_data[key] = sorted(value, key=lambda x: x['timestamp'])
            out.append(init_dataset(value, key))
        return out

    def __len__(self):
        return len(self.seq_data)
    
    def __getitem__(self, index):
        return self.seq_data[index]
    
@DATASETS.register_module()
class SequenceDataset(Dataset):
    def __init__(self, data, token=None, pipeline=None, **kwargs):
        super().__init__()
        self.data_infos = self.merge_sweeps(data)
        self.data_infos = sorted(self.data_infos, key=lambda x: x['timestamp'])
        self.start_lidar2global = self.get_lidar2global(0)
        self.token = token

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

    def get_lidar2global(self, index):
        # pyquaternion.Quaternion(self.data[index]["lidar2ego_rotation"])
        lidar2ego = pyquaternion.Quaternion(self.data_infos[index]["lidar2ego_rotation"]).transformation_matrix
        lidar2ego[:3, 3] = self.data_infos[index]["lidar2ego_translation"]
        # ego2global = np.eye(4)
        ego2global = pyquaternion.Quaternion(self.data_infos[index]["ego2global_rotation"]).transformation_matrix
        ego2global[:3, 3] = self.data_infos[index]["ego2global_translation"]
        lidar2global = ego2global @ lidar2ego
        return lidar2global

    def merge_sweeps(self, data):
        '''
        process lidar sweeps as key frame
        used_infos:
            trans_matrix...
            token
            lidar_path
            timestamp
            axis_align_matrix*
            annos:
                gt_boxes
                gt_names
                gt_inst_token
            cams

        '''
        data = sorted(data, key=lambda x: x['timestamp'])
        out = []
        for i in range(len(data)):
            info = data[i]
            # print(info['token'])
            sweeps = info.pop("sweeps")
            info['annos'] = {'gt_boxes': info.pop('gt_boxes'), 'gt_names':info.pop('gt_names'), "gt_inst_token":info.pop('gt_inst_token')}
            info["is_key_frame"] = True
            out.append(info)
            for sweep in sweeps:
                sweep['lidar_path'] = sweep['data_path']    
                sweep['lidar2ego_translation'] = sweep['sensor2ego_translation']
                sweep['lidar2ego_rotation'] = sweep['sensor2ego_rotation']
                sweep["is_key_frame"] = False
                out.append(sweep)
        return out
    
    def __getitem__(self, index):
        return self.data_infos[index] 

    def __len__(self):
        return len(self.data_infos)
    
    def pre_pipeline(self, results):
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = LiDARInstance3DBoxes
        results['box_mode_3d'] = 0

    def get_data_info(self, index):
        info = self.data_infos[index]
        
        lidar2global = self.get_lidar2global(index)
        lidar2start = np.linalg.inv(self.start_lidar2global) @ lidar2global
        info["axis_align_matrix"] = lidar2start
        # sample_idx = info['sample_idx']
        # sample_idx = index
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            # sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )
        # if not self.test_mode:
        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos
            # if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
                # return None
        # print(f"token:{info['token']}, is key frame:{info['is_key_frame']}")
        return input_dict
    
    def get_ann_info(self, index):
        info = self.data_infos[index]
        gt_bboxes_3d = info["annos"]['gt_boxes']
        gt_names_3d = info["annos"]['gt_names']
        gt_bboxes_id = info["annos"]['gt_inst_token']

        # turn original box type to target box type
        gt_bboxes_3d = LiDARInstance3DBoxeswithID(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            ids=gt_bboxes_id,
            origin=(0.5, 0.5, 0.5)).convert_to(0) #! (0.5, 0.5, 0),
        
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_names_3d,
            gt_bboxes_id=gt_bboxes_id,
            gt_names=gt_names_3d,
            axis_align_matrix=info['axis_align_matrix'],
            )
        return anns_results
    
def init_dataset(data, token):
    dataset_cfg = dict(
        type="SequenceDataset", data=data, token=token)
    dataset_cfg.update(
        # use_valid_flag=True,
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5),
            dict(type="RemoveSelfCenter", radius=2.5),
            dict(
                type='RemoveGround',
            ),
            # dict(
            #     type='LoadPointsFromMultiSweeps',
            #     sweeps_num=10,
            #     use_dim=[0, 1, 2, 3, 4],
            #     pad_empty_sweeps=True,
            #     remove_close=True),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(type='GlobalAlignmentwithGT',
                 ),
            

        ])
    
    dataset = build_dataset(dataset_cfg)
    return dataset


def get_mesh(points, voxel_size=0.1, sdf_trunc=0.3, min_weight=0):
    # min_weight = 0
    # voxel_size = 0.08
    # sdf_trunc = 0.1
    vdb_volume = vdbfusion.VDBVolume(voxel_size=voxel_size,
                                    sdf_trunc=sdf_trunc,
                                    space_carving=False)
    for i, point in enumerate(tqdm(points)):
        if len(point) > 0:
            vdb_volume.integrate(point[:, :3].astype(np.float64), np.eye(4))
    vertices, triangles = vdb_volume.extract_triangle_mesh(fill_holes=True, min_weight=min_weight)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles),
        )
    mesh.compute_vertex_normals()
    return mesh

def get_ground_mesh(ground_points, grid_size=5):
    ground_points = ground_points[np.logical_and(ground_points[:, 2] > -1.5, ground_points[:, 2] < 1.5)]
    region_range = np.stack([ground_points[:, :3].min(axis=0), ground_points[:, :3].max(axis=0)])
    x_edges = np.arange(region_range[0, 0], region_range[1, 0], grid_size)
    y_edges = np.arange(region_range[0, 1], region_range[1, 1], grid_size)
    region_grid = np.meshgrid(x_edges, y_edges, indexing='ij')
    grid_indices = np.stack((
        np.digitize(ground_points[:, 0], bins=x_edges) - 1,
        np.digitize(ground_points[:, 1], bins=y_edges) - 1,), axis=-1)
    ground_points2grid_map = defaultdict(list)
    for i, idx in enumerate(grid_indices):
        pos = (x_edges[idx[0]], y_edges[idx[1]])
        ground_points2grid_map[pos].append(i)
    # ret = []
    # for x_, y_ in :

    def thread_worker(i, region_grid=None, ground_points2grid_map=None, points=None):
        x_, y_ = region_grid[i]
        idx = np.array(ground_points2grid_map[(x_, y_)])
        if len(idx) < 2:
            return None
        points_ = ground_points[np.array(ground_points2grid_map[(x_, y_)])]
        print(points_.shape)
        ground_pcd = o3d.geometry.PointCloud()
        ground_pcd.points = o3d.utility.Vector3dVector(points_[:, :3])    # show_result(points=all_points, gt_bboxes=None, pred_bboxes=None, out_dir="./viz", filename=example['sample_idx'], show=True, snapshot=True)
        ground_pcd.estimate_normals()
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        ground_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(ground_pcd, depth=9)
        ground_mesh.compute_vertex_normals()
        return ground_mesh
    region_grid = np.stack((region_grid), -1).reshape(-1, 2)
    thread_func = partial(thread_worker, region_grid=region_grid, ground_points2grid_map=ground_points2grid_map, points=ground_points)
    ret = []
    for i in range(len(region_grid)):
        mesh_ = thread_func(i)
        if mesh_ is not None:
            ret.append(mesh_)
    # ret = thread_map(thread_func, range(len(region_grid)), max_workers=1, chunksize=20)
    # 
        # ret.append(ground_mesh)
    return ret
# def get_voxel(mesh_points, voxel_size=0.1):
    # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(mesh_points,
    #                                                         voxel_size=0.05)


def main(data_path, info_path):

    dataset = SeqNuscene(data_path, info_path, nuscenes_version="v1.0-mini")
    for i in track_iter_progress(list(range(len(dataset)))):
        seq_dataset = dataset[i]
        obj_points_bank = {}
        ground_points_bank = {}
        nonground_points_bank = {}
        for j in track_iter_progress(list(range(len(seq_dataset)))):
            input_dict = seq_dataset.get_data_info(j)
            seq_dataset.pre_pipeline(input_dict)
            example = seq_dataset.pipeline(input_dict)
            
            annos = example['ann_info']
            image_idx = example['sample_idx']
            points = example['points'].tensor.numpy()
            gt_boxes_3d = annos['gt_bboxes_3d'].tensor
            names = annos['gt_names']
            gt_boxes_id = annos['gt_bboxes_id']
            obj_point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d.numpy())

            #! 区分地面/非地面, 前景(标注目标)/背景点云
            background_points_indices = np.where(~obj_point_indices.any(axis=1))[0]
            ground_points_indices = example['ground_idx']
            nonground_points_indices = example['nonground_idx']
            background_and_ground_points_indices = np.intersect1d(ground_points_indices, background_points_indices, assume_unique=True)
            background_and_nonground_points_indices = np.intersect1d(nonground_points_indices, background_points_indices, assume_unique=True)
            # left_points = example['points'][ground_points_indices]
            # left_points = points[background_points_indices]
            nonground_points_bank[example['timestamp']] =  points[background_and_nonground_points_indices]
            ground_points_bank[example['timestamp']] = points[background_and_ground_points_indices]
            # show_result(points=points, gt_bboxes=gt_boxes_3d.clone(), pred_bboxes=None, out_dir="./viz", filename=example['sample_idx'], show=True, snapshot=True)
            
            for i, (obj_id, box) in enumerate(zip(gt_boxes_id, gt_boxes_3d.numpy())):
                if obj_id not in obj_points_bank:
                    obj_points_bank[obj_id] = {'points':[], "size": box[3:6]}
                gt_points = points[obj_point_indices[:, i]]
                gt_points[:, :3] -= box[:3]
                gt_points[:, :3] = gt_points[:, :3] @ rotate_yaw(box[6]).T
                obj_points_bank[obj_id]['points'].append(gt_points)
                obj_points_bank[obj_id]['size'] = np.maximum(obj_points_bank[obj_id]['size'], box[3:6])
                pass
        
        #! 序列背景点云拼接验证
        # all_points = list(background_points_bank.values())
        # all_points = np.vstack(all_points)

        #! 地面点二次筛选
        all_points = list(nonground_points_bank.values())
        PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(pypatchworkpp.Parameters())
        batch_id = np.concatenate([np.full(len(points), i) for i, points in enumerate(all_points)])
        all_points_arr = np.vstack(all_points)
        PatchworkPLUSPLUS.estimateGround(all_points_arr[:, :4])
        nonground_idx = PatchworkPLUSPLUS.getNongroundIndices()
        nonground_points = all_points_arr[nonground_idx]
        ground_idx = PatchworkPLUSPLUS.getGroundIndices()
        batch_id = batch_id[nonground_idx]
        all_points = [nonground_points[batch_id == i] for i in range(len(nonground_points_bank))]
        #! 背景非地面点云mesh重建
        nonground_mesh = get_mesh(all_points)
        
        
        # viz_mesh(nonground_mesh)
        ground_points = list(ground_points_bank.values())
        ground_points = np.vstack(ground_points)
        ground_points = np.concatenate((ground_points, all_points_arr[ground_idx]))
        ground_mesh = get_ground_mesh(ground_points)
        viz_mesh(ground_mesh)
        for obj_id, data in obj_points_bank.items():
            points = data['points']
            size = data['size']
            mesh = get_mesh(points)
            # points = np.vstack(points)
            show_result(points=np.vstack(points), gt_bboxes=None, pred_bboxes=None, out_dir="./viz", filename=example['sample_idx'], show=True, snapshot=True)
            o3d.visualization.draw_geometries([mesh])  
            o3d.io.write_triangle_mesh(f"./viz/{obj_id}.ply", mesh)


def viz_mesh(mesh, gt_box=None, save=True):

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])  # create coordinate frame
    vis.add_geometry(mesh_frame)
    if isinstance(mesh, list):
        for m in mesh:
            vis.add_geometry(m)
    else:
        vis.add_geometry(mesh)
    if gt_box is not None:
        gt_box[0, 2] += 0.5 * gt_box[0, 5]
        box3d = o3d.geometry.OrientedBoundingBox(gt_box[0, 0:3], np.eye(3), gt_box[0, 3:6])
        line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
        vis.add_geometry(line_set)
    vis.run()
    # o3d.visualization.draw_geometries([mesh, mesh, mesh])  
    if save:
        o3d.io.write_triangle_mesh(f"./viz/tmp.ply", mesh)
    return 

if __name__ == '__main__':
    from ops.mmdetection3d.tools.data_converter.nuscenes_converter import create_nuscenes_infos
    create_nuscenes_infos(
        # '/media/chen/Elements/nuScenes/raw/Trainval/',
        "/media/chen/data/nusecnes/v1.0-mini",
        info_prefix="test",
        version='v1.0-mini',
        get_seg=True)

    main('/media/chen/data/nusecnes/v1.0-mini/', 'test_infos_train.pkl')
    obj_points_bank = pickle.load(open("tmp.pkl", "rb"))
    for obj_id, data in obj_points_bank.items():
    # if True:
        # obj_id = "e91afa15647c4c4994f19aeb302c7179"
        # data = obj_points_bank[obj_id]
        points = data['points']
        gt_box = np.zeros((1, 7))
        gt_box[0, 3:6] = data['size']
        # size = data['size']
        mesh = get_mesh(points)
        # mesh 重建
        viz_mesh(mesh, gt_box.copy(), save=True)
        # 原始点云体素化
        viz_points = np.vstack(points)
        mesh_points = o3d.geometry.PointCloud()
        mesh_points.points = o3d.utility.Vector3dVector(viz_points[:,:3])
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(mesh_points,
                                                            voxel_size=0.05)
        viz_mesh(voxel_grid, gt_box.copy(), save=False)
        
        # 原始点云
        show_result(points=viz_points, gt_bboxes=gt_box.copy(), pred_bboxes=None, out_dir="./viz", filename=obj_id, show=True, snapshot=True)
        
        mesh_points = mesh.sample_points_poisson_disk(10000)
        points = np.asarray(mesh_points.points)
        points = points[np.logical_and.reduce([points[:, 0] >= -gt_box[0, 3] / 2, points[:, 0] <= gt_box[0, 3] / 2, points[:, 1] >= -gt_box[0, 4] / 2, points[:, 1] <= gt_box[0, 4] / 2, points[:, 2] >= 0, points[:, 2] <= gt_box[0, 5] ])]
        mesh_points.points = o3d.utility.Vector3dVector(np.vstack((points, viz_points[:,:3])))
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(mesh_points,
                                                            voxel_size=0.05)
        
        # mesh点云+原始点云
        viz_points = np.asarray(mesh_points.points)
        show_result(points=viz_points, gt_bboxes=gt_box.copy(), pred_bboxes=None, out_dir="./viz", filename=obj_id, show=True, snapshot=True)
        # (mesh点云+原始点云i)mesh重建
        viz_mesh(voxel_grid, gt_box.copy(), save=False)

        # mesh = get_mesh([viz_points, ])
        # viz_mesh(mesh, gt_box.copy(), save=True)
        
        
