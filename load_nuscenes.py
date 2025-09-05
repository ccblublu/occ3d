from collections import defaultdict
from os import path as osp
import pickle
from tkinter import _flatten
import pyquaternion
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from typing import List

import torch
import imagesize
import open3d as o3d
import numpy as np
from scipy import stats
from torch.utils.data import Dataset
from nuscenes import NuScenes
from mmcv import load as mmcv_load
from mmdet3d.datasets import build_dataset, DATASETS
from mmcv import track_iter_progress
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmcv.utils import Registry, build_from_cfg
from mmdet3d.core.bbox import LiDARInstance3DBoxes, get_box_type, box_np_ops
from mmdet3d.datasets.pipelines import Compose
from numba import njit, prange
from mmseg.apis import inference_segmentor
import vdbfusion
import pypatchworkpp

from ops.mmdetection3d.tools.misc.browse_dataset import show_det_data, show_result
from utils.ops import generate_35_category_colors, rotate_yaw, viz_mesh, viz_occ
from utils.ray_operation import ray_casting, camera_ray_occ, image_guided_voxel_refinement, project_voxel2pixel
from utils.nuScenes_infos import *
from ops.segmentation.image_segmentaiton import init_model
from utils import data_pipeline
# import ray_casting_cuda


class LiDARInstance3DBoxeswithID(LiDARInstance3DBoxes):
    def __init__(
        self, tensor, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0), ids=None
    ):
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
        self.anno_metadata = anno_data["metadata"]
        self.anno_infos = anno_data["infos"]
        self.seq_data = self.get_seq_data()

    def get_seq_data(self):
        seq_data = defaultdict(list)
        for id, info in enumerate(self.anno_infos):
            token = info["token"]
            sample = self.nuscenes.get("sample", token)
            seq_data[sample["scene_token"]].append(info)
        seq_data = dict(seq_data)
        out = []
        for key, value in seq_data.items():
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
        self.data_infos = sorted(self.data_infos, key=lambda x: x["timestamp"])
        # self.start_lidar2global = self.get_lidar2global(0)
        self.start_ego2global = self.get_ego2global(0)
        self.token = token
        self.key_frame_index = self.get_key_frame_index()
        self.timestamps = np.array([info["timestamp"] for info in self.data_infos])

        self.fineidx2coarseidx = np.vectorize(NUSC_FINEIDX2COARSEIDX.get)
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

    def get_lidar2global(self, index):
        lidar2global = self.get_ego2global(index) @ self.get_lidar2ego(index)
        return lidar2global

    def get_ego2global(self, index):
        ego2global = pyquaternion.Quaternion(
            self.data_infos[index]["ego2global_rotation"]
        ).transformation_matrix
        ego2global[:3, 3] = self.data_infos[index]["ego2global_translation"]
        return ego2global

    def get_lidar2ego(self, index):
        lidar2ego = pyquaternion.Quaternion(
            self.data_infos[index]["lidar2ego_rotation"]
        ).transformation_matrix
        lidar2ego[:3, 3] = self.data_infos[index]["lidar2ego_translation"]
        return lidar2ego

    def get_key_frame_index(self):

        return np.array(
            [i for i, info in enumerate(self.data_infos) if info["is_key_frame"]]
        )

    def merge_sweeps(self, data):
        """
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

        """
        data = sorted(data, key=lambda x: x["timestamp"])
        out = []
        key_frame_timestamps = []
        for i in range(len(data)):
            info = data[i]
            key_frame_timestamps.append(info["timestamp"])
            # print(info['token'])
            sweeps = info.pop("sweeps")
            info["annos"] = {
                "gt_boxes": info.pop("gt_boxes"),
                "gt_names": info.pop("gt_names"),
                "gt_inst_token": info.pop("gt_inst_token"),
            }
            info["is_key_frame"] = True
            out.append(info)
            for sweep in sweeps:
                # ? 排查timestamp重叠的问题: nuscenes数据集中的sweeps的最后一帧为关键帧，需要额外剔除
                if sweep["timestamp"] in key_frame_timestamps:
                    continue
                sweep["lidar_path"] = sweep["data_path"]
                sweep["lidar2ego_translation"] = sweep["sensor2ego_translation"]
                sweep["lidar2ego_rotation"] = sweep["sensor2ego_rotation"]
                sweep["is_key_frame"] = False
                out.append(sweep)
        return out

    def __getitem__(self, index):
        return self.data_infos[index]

    def __len__(self):
        return len(self.data_infos)

    def pre_pipeline(self, results):
        results["img_fields"] = []
        results["bbox3d_fields"] = []
        results["pts_mask_fields"] = []
        results["pts_seg_fields"] = []
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []
        results["box_type_3d"] = LiDARInstance3DBoxes
        results["box_mode_3d"] = 0

    def get_data_info(self, index):
        info = self.data_infos[index]

        # lidar2global = self.get_lidar2global(index)
        ego2global = self.get_ego2global(index)
        ego2start = np.linalg.inv(self.start_ego2global) @ ego2global
        lidar2ego = self.get_lidar2ego(index)
        info["axis_align_matrix"] = ego2start @ lidar2ego

        input_dict = dict(
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            is_key_frame=info["is_key_frame"],
            timestamp=info["timestamp"],
            lidar2ego=lidar2ego,
            cams=info["cams"],
        )

        annos = self.get_ann_info(index)
        input_dict["ann_info"] = annos

        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        gt_bboxes_3d = info["annos"]["gt_boxes"]
        gt_names_3d = info["annos"]["gt_names"]
        gt_bboxes_id = info["annos"]["gt_inst_token"]
        if info["is_key_frame"]:
            gt_pts_semantic_mask = np.fromfile(
                info["pts_semantic_mask_path"], dtype=np.uint8
            )
            gt_pts_semantic_mask = self.fineidx2coarseidx(gt_pts_semantic_mask)
        else:
            gt_pts_semantic_mask = None
        # turn original box type to target box type
        gt_bboxes_3d = LiDARInstance3DBoxeswithID(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            ids=gt_bboxes_id,
            origin=(0.5, 0.5, 0.5),
        ).convert_to(
            0
        )  #! (0.5, 0.5, 0),

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_names_3d,
            gt_bboxes_id=gt_bboxes_id,
            gt_names=gt_names_3d,
            axis_align_matrix=info["axis_align_matrix"],
            gt_pts_semantic_mask=gt_pts_semantic_mask,
        )
        return anns_results


def init_dataset(data, token):
    dataset_cfg = dict(type="SequenceDataset", data=data, token=token)
    dataset_cfg.update(
        # use_valid_flag=True,
        pipeline=[
            dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5),
            dict(type="RemoveSelfCenter", radius=2.5),
            dict(type="GetGround"),
            dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
            dict(type="GetBoxPointIndices"),            
            dict(type="GlobalAlignmentwithGT"), #! 自由度太少，框有误差，先单帧取点云再拼接
        
        ]
    )

    dataset = build_dataset(dataset_cfg)
    return dataset


def get_mesh(points, voxel_size=0.1, sdf_trunc=0.3, min_weight=0):
    # min_weight = 0
    # voxel_size = 0.08
    # sdf_trunc = 0.1
    vdb_volume = vdbfusion.VDBVolume(
        voxel_size=voxel_size, sdf_trunc=sdf_trunc, space_carving=False
    )
    for i, point in enumerate(points):
        if len(point) > 0:
            vdb_volume.integrate(point[:, :3].astype(np.float64), np.eye(4))
    vertices, triangles = vdb_volume.extract_triangle_mesh(
        fill_holes=True, min_weight=min_weight
    )
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles),
    )
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=3)

    mesh.compute_vertex_normals()
    return mesh


def get_ground_mesh(ground_points, grid_size=5, max_workers=1):

    region_range = np.stack(
        [ground_points[:, :3].min(axis=0), ground_points[:, :3].max(axis=0)]
    )
    x_edges = np.arange(region_range[0, 0], region_range[1, 0], grid_size)
    y_edges = np.arange(region_range[0, 1], region_range[1, 1], grid_size)
    region_grid = np.meshgrid(x_edges, y_edges, indexing="ij")
    grid_indices = np.stack(
        (
            np.digitize(ground_points[:, 0], bins=x_edges) - 1,
            np.digitize(ground_points[:, 1], bins=y_edges) - 1,
        ),
        axis=-1,
    )
    ground_points2grid_map = defaultdict(list)
    for i, idx in enumerate(grid_indices):
        pos = (x_edges[idx[0]], y_edges[idx[1]])
        ground_points2grid_map[pos].append(i)

    def thread_worker(i, region_grid=None, ground_points2grid_map=None, points=None):
        x_, y_ = region_grid[i]
        idx = np.array(ground_points2grid_map[(x_, y_)])
        if len(idx) < 2:
            return None
        points_ = ground_points[np.array(ground_points2grid_map[(x_, y_)])]
        # print(points_.shape)
        ground_pcd = o3d.geometry.PointCloud()
        ground_pcd.points = o3d.utility.Vector3dVector(points_[:, :3])
        ground_pcd.estimate_normals()
        (
            ground_mesh,
            densities,
        ) = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            ground_pcd, depth=9
        )
        ground_mesh.compute_vertex_normals()
        return ground_mesh

    region_grid = np.stack((region_grid), -1).reshape(-1, 2)
    # thread_func = partial(
    #     thread_worker,
    #     region_grid=region_grid,
    #     ground_points2grid_map=ground_points2grid_map,
    #     points=ground_points,
    # )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        ret = []
        for i in range(len(region_grid)):
            ret.append(
                executor.submit(
                    thread_worker,
                    i,
                    region_grid=region_grid,
                    ground_points2grid_map=ground_points2grid_map,
                    points=ground_points,
                )
            )
    # ret = thread_map(thread_func, range(len(region_grid)), max_workers=1, chunksize=20)

    return [m for m in ret if m is not None]


def propagate_labels_with_knn(
    keyframe_points, keyframe_labels, non_keyframe_points, k=5, distance_threshold=0.5
):
    """
    使用KNN将关键帧的分割标签传播到非关键帧点云

    参数:
        keyframe_points (np.ndarray): (N, 3) 关键帧点云坐标
        keyframe_labels (np.ndarray): (N,) 关键帧点云的分割标签
        non_keyframe_points (np.ndarray): (M, 3) 非关键帧点云坐标
        k (int): KNN中使用的最近邻数量
        distance_threshold (float): 最大距离阈值，超过此距离的点不分配标签

    返回:
        non_keyframe_labels (np.ndarray): (M,) 非关键帧点云的预测标签
    """
    # 1. 创建KNN模型并拟合关键帧数据
    knn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree")
    knn.fit(keyframe_points)
    # 2. 为非关键帧点云找到k个最近邻
    distances, indices = knn.kneighbors(non_keyframe_points)
    # 3. 获取最近邻点的标签
    neighbor_labels = keyframe_labels[indices]
    # 4. 使用多数投票确定标签
    # 对于每个点，统计k个邻居中最常见的标签
    non_keyframe_labels = np.zeros(
        len(non_keyframe_points), dtype=keyframe_labels.dtype
    )
    # 处理每个点
    for i in range(len(non_keyframe_points)):
        # 检查最小距离是否在阈值内
        if distances[i].min() > distance_threshold:
            # 超出距离阈值的点分配默认标签（例如背景或未定义）
            non_keyframe_labels[i] = 0
            continue
        # 获取邻居标签
        labels = neighbor_labels[i]
        # 多数投票
        unique_labels, counts = np.unique(labels, return_counts=True)
        counts[unique_labels == 0] = -1  # 将背景标签的计数设为-1，使其不会影响多数投票
        non_keyframe_labels[i] = unique_labels[np.argmax(counts)]

    return non_keyframe_labels


def fine2coarseidx(fine_name):
    coarse_name = NUSC_FINE2COARSE[fine_name]
    coarse_idx = NUSC_COARSE2IDX[coarse_name]
    return coarse_idx


class Nuscenes2Occ3D:
    def __init__(self, nuscenes_version="v1.0-mini"):
        pass
        self.get_mesh_sample = False
        self.colors = generate_35_category_colors(35)
        self.pc_range = np.array([40, 40, 5.4, -40, -40, -1])
        self.voxel_size = 0.4
        self.load_offline = True
        # self.load_offline = False
        self.eps = 1e-5
        self.camera_list = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        # self.cv_seg_model = init_model()
        self.cv_seg_model = init_model() if not self.load_offline else None
        # self.clasees_map = {v: k for k, v in enumerate(self.cv_seg_model.CLASSES)}
        self.ade20k2nuscidx = np.vectorize(ADE20K2NUSC.get)

    def get_semantic(self, image_path):
        if self.load_offline:    
            result = np.load(osp.join("viz", osp.basename(image_path) + ".npy"))
        else:
            result = inference_segmentor(self.model, image_path)# 在线推理显存吃不消
        return result

    def main(self, data_path, info_path):
        self.dataset = SeqNuscene(data_path, info_path, nuscenes_version="v1.0-mini")
        for i in track_iter_progress(list(range(len(self.dataset)))):
            self.seq_run(i)

    def seq_run(self, seq_id):
        # self.dataset = SeqNuscene(data_path, info_path, nuscenes_version="v1.0-mini")
        # for i in track_iter_progress(list(range(len(self.dataset)))):
        self.seq_dataset = self.dataset[seq_id]
        self.obj_points_bank = {}
        self.obj_info_bank = {}
        self.timestamp2token = {}
        self.cams_bank = {}
        # ground_points_bank = {}
        # nonground_points_bank = {}
        self.background_points_bank = {}
        self.frame_info_bank = {}
        self.key_frame_indices = self.seq_dataset.key_frame_index
        self.timestamps = self.seq_dataset.timestamps
        self.key_frame_timestamps = self.timestamps[self.key_frame_indices]
        self.waiting_for_key_frame = defaultdict(list)
        if self.load_offline:
            self.save_and_load_offline()
            all_points = list(self.background_points_bank.values())
            all_points_arr = np.load("viz/all_points_arr.npy")
        else:
            for j in track_iter_progress(list(range(len(self.seq_dataset)))):
                self.get_frame_data(j)
            all_points = list(self.background_points_bank.values())
            all_points_arr = np.vstack(all_points).astype(np.float32)
            all_points_arr = self.filter_noise(all_points_arr)
        # self.sample_dynamic_objects()
        # self.get_seq_image_segmentation()
        # exit()

        batch_id = np.concatenate(
            [
                np.full(len(points), i)
                for i, points in enumerate(list(self.background_points_bank.values()))
            ]
        )

        #! 先采样再重建
        if self.get_mesh_sample:
            mesh_points = self.mesh_and_sample(all_points_arr)

        #! 还原至单帧自车坐标系
        self.get_key_frame_data(all_points_arr, batch_id)

    def save_and_load_offline(self):
        print("\nloading offline data……")
        keys = [
            "obj_points_bank",
            "obj_info_bank",
            "timestamp2token",
            "background_points_bank",
            "frame_info_bank",
            "cams_bank"
        ]
        # for key in keys:
        #     with open(f"viz/{key}.pkl", "wb") as f:
        #         pickle.dump(getattr(self, key), f)
        for k in keys:
            with open(f"viz/{k}.pkl", "rb") as f:
                info = pickle.load(f)
                setattr(self, k, info)
        # self.obj_points_bank = self.obj_points_bank.astype(np.float32)
        # self.background_points_bank = self.background_points_bank.astype(np.float32)

    def put_dynamic_objects(self, timestamp):
        frame_obj_points = []
        for obj_info in self.obj_info_bank[timestamp]:
            obj_id = obj_info["obj_id"]
            box = obj_info["box"]

            raw_points = np.concatenate(self.obj_points_bank[obj_id]["points"])[:, :3]
            if len(raw_points) < 0:
                mesh_points = self.obj_points_bank[obj_id]["sampled_points"]
                points = np.vstack([raw_points, mesh_points])
            else:
                points = raw_points
            points_ego = points @ rotate_yaw(box[6]) + box[:3]
            label = self.obj_points_bank[obj_id]["label"]
            points = np.concatenate(
                [points_ego, np.full((len(points), 1), label)], axis=1
            )
            frame_obj_points.append(points)

        return np.concatenate(frame_obj_points, axis=0)

    def get_key_frame_data(self, all_points_arr, batch_id):
        root = "/media/chen/data/OpenOcc/Occpancy3D/gts/scene-0061"
        all_coords = []
        
        #! 获取序列下的自车坐标
        lidars2start = np.array(
            [i["lidar2start"] for i in self.frame_info_bank.values()]
        )
        lidars2egos = np.array([i["lidar2ego"] for i in self.frame_info_bank.values()])
        spatial_shape = ((self.pc_range[:3] - self.pc_range[3:]) / self.voxel_size).astype(np.int32)

        for i, timestamp in enumerate(tqdm(self.key_frame_timestamps)):

            lidar2start = self.frame_info_bank[timestamp]["lidar2start"]
            lidar2ego = self.frame_info_bank[timestamp]["lidar2ego"]
            cameras = self.cams_bank[timestamp]
            
            ego2cars = (
                lidars2egos
                @ np.linalg.inv(lidars2start)
                @ lidar2start
                @ np.linalg.inv(lidar2ego)
            )

            # ego2lidars = (
            #     np.linalg.inv(lidars2start)
            #     @ lidar2start
            #     @ np.linalg.inv(lidar2ego)
            # )
            #! 获取当前帧下的序列轨迹坐标点
            trajectory_pose = np.linalg.inv(ego2cars)[:, :3, -1]
            trajectory_voxel = np.floor((trajectory_pose - np.array(self.pc_range[3:])) / self.voxel_size).astype(np.int32)
            car_gemotry = np.stack(np.meshgrid([-1,0,1],[-1,0,1],[-1,0,1]), -1).reshape(-1,1,3)
            trajectory_voxel = (trajectory_voxel + car_gemotry).reshape(-1,3)
            ego_sem = np.full(len(trajectory_voxel), 31, dtype=np.int32)


            local_points = all_points_arr[:, [0, 1, 2, -1]].copy()
            local_points[:, :3] = (
                local_points[:, :3] - lidar2start[:3, 3]
            ) @ lidar2start[
                :3, :3
            ]  #! at lidar coordinate
            local_points[:, :3] = (
                local_points[:, :3] @ lidar2ego[:3, :3].T + lidar2ego[:3, 3]
            )
            dynamic_points = self.put_dynamic_objects(timestamp)
            local_points = np.concatenate([local_points, dynamic_points], axis=0)
            local_batch_id = np.concatenate([batch_id, np.full(len(dynamic_points), list(self.background_points_bank.keys()).index(timestamp))], axis=0)
            points_origin = trajectory_pose[local_batch_id]  #! broadcast yyds

            #! 范围需要二次确认后统一： 范围确认无误，但在体素化时，需要转到第一象限进行操作以确保【取整】的一致性
            # range_mask = np.logical_and.reduce([np.abs(local_points[:, 0]) < 40, np.abs(local_points[:, 1]) < 40, local_points[:, 2] > -1, local_points[:, 2] < 5.4])
            range_mask = np.logical_and.reduce(
                [
                    local_points[:, 0] < self.pc_range[0] - self.eps,
                    local_points[:, 1] < self.pc_range[1] - self.eps,
                    local_points[:, 2] < self.pc_range[2] - self.eps,
                    local_points[:, 0] > self.pc_range[3] + self.eps,
                    local_points[:, 1] > self.pc_range[4] + self.eps,
                    local_points[:, 2] > self.pc_range[5] + self.eps,
                ]
            )

            local_points = local_points[range_mask]
            points_origin = points_origin[range_mask]
            # frame_mask = batch_id[range_mask] == 0
            # local_points = local_points[frame_mask]
            # points_origin = points_origin[frame_mask]
            # viz_occ(local_points[:, :3], local_points[:, -1], name=str(timestamp))
            print(f"get points voxelization: {timestamp}")
            occ_coords, voxel_labels_ = assign_voxel_labels_vectorized(
                local_points[:, :3],
                local_points[:, -1],
                voxel_size=self.voxel_size,
                pc_range=self.pc_range,
            )
            # relative_voxel_coords_ = ((voxel_coords_ - self.pc_range[3:]) / self.voxel_size - 0.5).astype(int)

            occ_dense_label = np.full((spatial_shape), 17, dtype=np.int32) #! 17 free
            occ_dense_label[occ_coords[:, 0], occ_coords[:, 1], occ_coords[:, 2]] = voxel_labels_

            # print(f"calculate lidar visibility: {timestamp}")
            lidar_voxel_state = calculate_lidar_visibility(local_points, points_origin, occ_coords, np.array(self.pc_range)[3:], np.array([self.voxel_size, self.voxel_size, self.voxel_size]), spatial_shape)
            camera_voxel_state, free_voxels, cam_id_count = calculate_camera_visibility(cameras, lidar_voxel_state, np.array(self.pc_range)[3:], np.array([self.voxel_size, self.voxel_size, self.voxel_size]), spatial_shape)
            semantics_adjust = self.segmentation_refine(cameras, free_voxels, cam_id_count, occ_dense_label, camera_voxel_state)
            
            # occ_gt_path = f"{root}/{self.timestamp2token[timestamp]}/labels.npz"
            # occ_gt_info = np.load(occ_gt_path)
            # occ_gt = occ_gt_info["semantics"].reshape(-1)
            # gt_lidar_mask = occ_gt_info["mask_lidar"]
            # x = np.arange(-40, 40, self.voxel_size) + self.voxel_size / 2
            # y = np.arange(-40, 40, self.voxel_size) + self.voxel_size / 2
            # z = np.arange(-1, 5.4, self.voxel_size) + self.voxel_size / 2
            # grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing="ij")
            # coords = np.stack([grid_x, grid_y, grid_z], axis=-1)
            # coords = coords.reshape(-1, 3)
            # mask_pred = (camera_voxel_state == 0).reshape(-1)
            # trajectory_voxel = (trajectory_voxel + 0.5 ) * self.voxel_size + self.pc_range[3:] 
            # # with_car_voxel = np.concatenate([coords[mask_pred], trajectory_voxel])
            # # with_car_sem = np.concatenate([voxel_state.reshape(-1)[mask_pred], ego_sem], dtype=np.int32)
            # viz_occ(
            #     # with_car_voxel,
            #     # with_car_sem,
            #     coords[mask_pred],
            #     camera_voxel_state.reshape(-1)[mask_pred],
            #     name=f"{timestamp}-free-gen",
            #     viz=False,
            #     voxel_size=self.voxel_size,
            # )
            # # mask = occ_gt != 17
            # mask = gt_lidar_mask.reshape(-1).astype(bool)
            # viz_occ(
            #     coords[mask],
            #     occ_gt[mask],
            #     name=f"{timestamp}-free-gt",
            #     viz=False,
            #     voxel_size=self.voxel_size,
            # )

            # coords = (coords - lidar2ego[:3, 3]) @ lidar2ego[:3, :3] @ lidar2start[
            #     :3, :3
            # ].T + lidar2start[:3, 3]
            print(f"get image segmentation: {timestamp}")

    
    def segmentation_refine(self, cam_infos, free_voxels, cam_id_count, voxel_labels_, camera_voxel_state):
        start_pos = 0
        semantics_adjust = np.full_like(voxel_labels_, -1)
        for cam_type, cam_info in cam_infos.items():
            max_u, max_v = cam_id_count[cam_type]
            point_count = max_u * max_v
            semantics = self.get_semantic(cam_info["data_path"]) #! h*w
            nusc_semantics = self.ade20k2nuscidx(semantics)
            free_voxel = free_voxels[start_pos:start_pos + point_count]
            semantics_adjust = image_guided_voxel_refinement(
                semantics_adjust,
                nusc_semantics,
                free_voxel,
                voxel_labels_,
                # max_u,
                max_v,
            )
            start_pos += point_count
            blank_sem = np.zeros_like(nusc_semantics)
            image = project_voxel2pixel(
                blank_sem,
                free_voxel,
                voxel_labels_,
                camera_voxel_state,
                max_v
            )
        return semantics_adjust



    def sample_dynamic_objects(
        self,
    ):

        for obj_id, data in self.obj_points_bank.items():
            points = data["points"]
            if sum([len(i) for i in points]) == 0:
                self.obj_points_bank[obj_id]["sampled_points"] = np.zeros((0, 3))
                continue
            gt_box = np.zeros((1, 7))
            gt_box[0, 3:6] = data["size"]
            #! mesh 重建
            mesh = get_mesh(points)
            # viz_mesh(mesh, gt_box.copy(), save=True)
            #! 原始点云体素化
            viz_points = np.vstack(points)
            mesh_points = o3d.geometry.PointCloud()
            mesh_points.points = o3d.utility.Vector3dVector(viz_points[:, :3])
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
                mesh_points, voxel_size=0.05
            )
            # viz_mesh(voxel_grid, gt_box.copy(), save=False)

            # 原始点云
            # show_result(
            #     points=viz_points,
            #     gt_bboxes=gt_box.copy(),
            #     pred_bboxes=None,
            #     out_dir="./viz",
            #     filename=obj_id,
            #     show=True,
            #     snapshot=True,
            # )

            mesh_points = mesh.sample_points_poisson_disk(len(viz_points) * 2)
            points = np.asarray(mesh_points.points)
            points = points[
                np.logical_and.reduce(
                    [
                        points[:, 0] >= -gt_box[0, 3] / 2,
                        points[:, 0] <= gt_box[0, 3] / 2,
                        points[:, 1] >= -gt_box[0, 4] / 2,
                        points[:, 1] <= gt_box[0, 4] / 2,
                        points[:, 2] >= 0,
                        points[:, 2] <= gt_box[0, 5],
                    ]
                )
            ]
            # mesh_points.points = o3d.utility.Vector3dVector(
            #     np.vstack((points, viz_points[:, :3]))
            # )
            self.obj_points_bank[obj_id]["sampled_points"] = points
            # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            #     mesh_points, voxel_size=0.05
            # )

            # mesh点云+原始点云
            # viz_points = np.asarray(mesh_points.points)
            # show_result(
            #     points=viz_points,
            #     gt_bboxes=gt_box.copy(),
            #     pred_bboxes=None,
            #     out_dir="./viz",
            #     filename=obj_id,
            #     show=True,
            #     snapshot=True,
            # )
            # (mesh点云+原始点云i)mesh重建
            # viz_mesh(voxel_grid, gt_box.copy(), save=False)

            # mesh = get_mesh([viz_points, ])
            # viz_mesh(mesh, gt_box.copy(), save=True)

    def filter_noise(self, all_points_arr):
        """噪音点云二次赋值"""
        noise_points_index = np.where(all_points_arr[:, -1] == 0)[0]
        comfirm_points_index = np.setdiff1d(
            np.arange(len(all_points_arr)), noise_points_index
        )
        noise_points_labels = propagate_labels_with_knn(
            all_points_arr[comfirm_points_index, :3],
            all_points_arr[comfirm_points_index, -1],
            all_points_arr[noise_points_index, :3],
            distance_threshold=0.5,
        )
        all_points_arr[noise_points_index, -1] = noise_points_labels

        return all_points_arr

    def mesh_and_sample(self, all_points_arr):
        sample_points, sample_labels = assign_voxel_labels_vectorized(
            all_points_arr[:, :3], all_points_arr[:, -1], voxel_size=0.01
        )
        scene_mesh = get_mesh([sample_points])
        # # viz_mesh(scene_mesh, save=True)
        mesh_pcd = scene_mesh.sample_points_poisson_disk(len(all_points_arr))
        mesh_points = np.asarray(mesh_pcd.points)
        mesh_point_labels = propagate_labels_with_knn(
            all_points_arr[:, :3],
            all_points_arr[:, -1],
            mesh_points,
            distance_threshold=0.5,
        )
        mesh_points = np.concatenate(
            [mesh_points, mesh_point_labels.reshape(-1, 1)], axis=-1
        )

        return mesh_points

    def viz_points(self, all_points, obj_id="tmp"):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(
            self.colors[all_points[:, -1].astype(int)]
        )
        o3d.io.write_point_cloud(f"./viz/{obj_id}.ply", pcd)

    def get_frame_data(self, index):
        input_dict = self.seq_dataset.get_data_info(index)
        self.seq_dataset.pre_pipeline(input_dict)
        example = self.seq_dataset.pipeline(input_dict)
        annos = example["ann_info"]
        timestamp = example["timestamp"]
        image_idx = example["sample_idx"]

        self.timestamp2token[timestamp] = image_idx
        self.frame_info_bank[timestamp] = {
            "lidar2start": annos["axis_align_matrix"],
            "lidar2ego": example["lidar2ego"],
        }
        points = example["points"].tensor.numpy()[:, :3].astype(np.float32)  #! (n, 3)
        gt_pts_semantic_mask = annos["gt_pts_semantic_mask"]  #!(n, )

        if gt_pts_semantic_mask is None:
            gt_pts_semantic_mask = np.zeros(len(points), dtype=np.uint8)
        points = np.concatenate((points, gt_pts_semantic_mask[:, None]), axis=1)
        gt_boxes_3d = annos["gt_bboxes_3d"].tensor
        names = annos["gt_names"]
        gt_boxes_id = annos["gt_bboxes_id"]
        obj_point_indices = example['obj_point_indices']
        # obj_point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d.numpy())

        #! 区分地面/非地面, 前景(标注目标)/背景点云
        background_points_indices = np.where(~obj_point_indices.any(axis=1))[0]
        ground_points_indices = example["ground_idx"]
        nonground_points_indices = example["nonground_idx"]
        background_and_ground_points_indices = np.intersect1d(
            ground_points_indices, background_points_indices, assume_unique=True
        )
        background_and_nonground_points_indices = np.intersect1d(
            nonground_points_indices, background_points_indices, assume_unique=True
        )
        # points_bank[timestamp] = points[background_points_indices]

        # left_points = example['points'][ground_points_indices]
        # left_points = points[background_points_indices]
        # nonground_points_bank[example['timestamp']] =  points[background_and_nonground_points_indices]
        # ground_points_bank[example['timestamp']] = points[background_and_ground_points_indices]

        if not example["is_key_frame"]:
            closed_key_frame_idx = np.argmin(
                np.abs(self.key_frame_timestamps - timestamp)
            )
            keyframe_timesamp = self.key_frame_timestamps[closed_key_frame_idx]

            if keyframe_timesamp not in self.background_points_bank:
                self.waiting_for_key_frame[keyframe_timesamp].append(timestamp)
            else:
                # keyframe_timesamp = key_frame_timestamps[closed_key_frame_idx]
                # keyframe_points = np.concatenate((nonground_points_bank[keyframe_timesamp], ground_points_bank[keyframe_timesamp]))
                keyframe_points = self.background_points_bank[keyframe_timesamp]
                keyframe_labels = keyframe_points[:, -1]
                labels = propagate_labels_with_knn(
                    keyframe_points[:, :3], keyframe_labels, points[:, :3]
                )
                points[:, -1] = labels
        elif timestamp in self.waiting_for_key_frame:
            non_key_frames = self.waiting_for_key_frame.pop(timestamp)
            for non_key_frames_timestamp in non_key_frames:
                non_keyframe_points = self.background_points_bank[
                    non_key_frames_timestamp
                ]
                non_keyframe_labels = propagate_labels_with_knn(
                    points[background_points_indices, :3],
                    points[background_points_indices, -1],
                    non_keyframe_points[:, :3],
                )
                self.background_points_bank[non_key_frames_timestamp][
                    :, -1
                ] = non_keyframe_labels

        self.background_points_bank[timestamp] = points[background_points_indices]
        self.cams_bank[timestamp] = example["cams"]
        # show_result(points=points, gt_bboxes=gt_boxes_3d.clone(), pred_bboxes=None, out_dir="./viz", filename=example['sample_idx'], show=True, snapshot=True)
        self.obj_info_bank[timestamp] = []
        for i, (obj_id, box) in enumerate(zip(gt_boxes_id, gt_boxes_3d.numpy())):
            self.obj_info_bank[timestamp].append({"obj_id": obj_id, "box": box})
            if obj_id not in self.obj_points_bank:
                self.obj_points_bank[obj_id] = {
                    "points": [],
                    "size": box[3:6],
                    "label": NUSC_COARSE2IDX[NUSC_FINE2COARSE.get(names[i], names[i])],
                }
            gt_points = points[obj_point_indices[:, i]]
            gt_points[:, :3] -= box[:3]
            gt_points[:, :3] = gt_points[:, :3] @ rotate_yaw(box[6]).T
            self.obj_points_bank[obj_id]["points"].append(gt_points)
            self.obj_points_bank[obj_id]["size"] = np.maximum(
                self.obj_points_bank[obj_id]["size"], box[3:6]
            )
        return

    def get_seq_image_segmentation(self):
        for i, timestamp in enumerate(tqdm(self.key_frame_timestamps)):
            cameras = self.cams_bank[timestamp]
            for camera_name, camera_data in cameras.items():
                result = inference_segmentor(self.cv_seg_model, camera_data["data_path"])[0]
                np.save(osp.join(f"./viz", osp.basename(camera_data["data_path"])),
                    result,
                )
                # self.cv_seg_model.show_result(camera_data["data_path"], result, show=True)

def assign_voxel_labels_vectorized(
    points, labels, voxel_size=0.05, pc_range=[40, 40, 5.4, -40, -40, -1]
):
    """
    使用向量化操作高效地为体素分配标签

    参数:
    points: 点云坐标 (N, 3)
    labels: 点云分割标签 (N,)
    voxel_size: 体素大小

    返回:
    voxel_coords: 体素中心坐标 (M, 3)
    voxel_labels: 体素标签 (M,)
    """
    # 计算每个点所在的体素索引
    zero_point = points - np.array(pc_range[3:])
    voxel_indices = np.floor(zero_point / voxel_size).astype(int)
    # 使用字典存储每个体素中的标签
    voxel_dict = defaultdict(list)
    # 将每个点的标签添加到对应体素的列表中
    for i, idx in enumerate(voxel_indices):
        voxel_key = tuple(idx)
        voxel_dict[voxel_key].append(i)
        # voxel_dict[voxel_key].append(labels[i])
    # 提取体素坐标和标签
    occ_coords = []
    voxel_labels = []
    # voxel_counts = []
    for voxel_key, idx_list in voxel_dict.items():
        # 计算体素中心坐标
        # center = (np.array(voxel_key) + 0.5) * voxel_size
        center = np.array(voxel_key)
        occ_coords.append(center)
        # 使用多数投票确定体素标签
        label_list = labels[idx_list]
        unique, counts = np.unique(label_list, return_counts=True)
        counts[unique == 0] = 0  # 将背景标签的计数设为-1，使其不会影响多数投票
        counts[unique == 11] = min((unique == 11).sum(), 10)
        voxel_labels.append(unique[np.argmax(counts)])
        # voxel_counts.append(len(label_list))

    return (
        np.array(occ_coords),
        np.array(voxel_labels),
        # dict(voxel_dict),
    )


def calculate_lidar_visibility(ray_start, ray_end, occ_coords, pc_range_min, voxel_size, spatial_shape):
    """
    计算单帧下的点云可见性(全序列计算)

    input:
        voxel_free_count
        voxel_occ_count
    
    voxel_state[voxel_free_count > 0] = 0
    voxel_state[voxel_occ_count > 0] = 1
    output:
        voxel_state
    """
    free_voxels = ray_casting(
            ray_start,
            ray_end,
            pc_range_min,
            voxel_size,
            spatial_shape
        )
    # voxel_points = ((voxel_coords - pc_range_min) / voxel_size - 0.5).astype(int)
    
    free_voxel = np.concatenate(free_voxels) 
    voxel_state =  np.full((spatial_shape), -1, dtype=np.int32)
    voxel_state[free_voxel[:,0], free_voxel[:,1], free_voxel[:,2]] = 0
    voxel_state[occ_coords[:,0], occ_coords[:,1], occ_coords[:,2]] = 1
    return voxel_state


def calculate_camera_visibility(cam_infos, lidar_voxel_state, pc_range_min, voxel_size, spatial_shape):
    """
    计算单帧下的相机可见度(单帧计算)

    input:
        Images: (K, w, h)
        P_cam: (K, 4, 4)
        P_cam2ego: (K, 4, 4)
        P_ego2global: (K, 4, 4)
        P_intrinsics: (K, 4, 4)
        voxel_state: (H, W, Z)
        voxel_label: (H, W, Z)
        pc_range_min: (3,)
        voxel_size: (3,)
        spatial_shape: (3,)
        
    """
    DEPTH_MAX = 1e2
    update_voxel_state =  np.full((spatial_shape), -1, dtype=np.int32)


    uv2points = []
    origins = []
    cam_id_count = {}
    for cam_type, cam_info in cam_infos.items():        
        u, v = imagesize.get(cam_info['data_path'])
        intrinsic = cam_info['cam_intrinsic']
        cam2ego = pyquaternion.Quaternion(cam_info['sensor2ego_rotation']).transformation_matrix
        cam2ego[:3, 3] = cam_info['sensor2ego_translation']
        image_coords = np.stack(np.meshgrid(np.arange(u), np.arange(v), indexing='ij'), axis=-1)
        image_coords = image_coords.reshape(-1, 2)
        image_coords = np.concatenate([image_coords, np.ones((image_coords.shape[0], 1), dtype=np.int32)], axis=-1, dtype=np.int32)
        depth = np.full(u * v, DEPTH_MAX, dtype=np.float32)
        uvs = image_coords * depth[:, None]
        cam_coords = uvs @ np.linalg.inv(intrinsic.T)
        ego_coords = cam_coords @ cam2ego[:3, :3].T + cam2ego[:3, 3]
        # origin = np.zeros((4, 4), dtype=np.float32)
        # origin[3, 3] = 1
        ego_origin = cam2ego[None, :3, 3]
        uv2points.append(ego_coords)
        origins.append(ego_origin.repeat(ego_coords.shape[0], 0))
        cam_id_count[cam_type] = (u, v)

    uv2points = np.concatenate(uv2points, axis=0)
    origins = np.concatenate(origins, axis=0)
    ray_start = origins[:, :3]
    ray_end = uv2points[:, :3]
    free_voxels = ray_casting(ray_start, ray_end, pc_range_min, voxel_size, spatial_shape)

    #!: 相机伪点云传播机制已验证，需要按照顺序计算遮挡与不可见
    update_voxel_state = camera_ray_occ(free_voxels, lidar_voxel_state, update_voxel_state)

    return update_voxel_state, free_voxels, cam_id_count

if __name__ == "__main__":
    from ops.mmdetection3d.tools.data_converter.nuscenes_converter import (
        create_nuscenes_infos,
    )

    # create_nuscenes_infos(
    #     # '/media/chen/Elements/nuScenes/raw/Trainval/',
    #     "/media/chen/data/nusecnes/v1.0-mini",
    #     info_prefix="test",
    #     version="v1.0-mini",
    #     get_seg=True,
    # )
    nusc2occ = Nuscenes2Occ3D()
    nusc2occ.main("/media/chen/data/nusecnes/v1.0-mini/", "test_infos_train.pkl")
    # main("/media/chen/data/nusecnes/v1.0-mini/", "test_infos_train.pkl")
