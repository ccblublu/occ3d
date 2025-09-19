from collections import OrderedDict
import numpy as np
from mmdet3d.datasets.builder import PIPELINES
from mmdet3d.core.bbox import box_np_ops
import pypatchworkpp
import torch
from pointcept.models import build_model
from utils.nuScenes_infos import NUSC_COARSE2IDX

@PIPELINES.register_module()
class GetBoxPointIndices:
    def __init__(self):
        pass

    def __call__(self, input_dict):
        if "gt_bboxes_3d" in input_dict:
            gt_boxes_3d = input_dict["gt_bboxes_3d"].tensor.numpy()
            points = input_dict["points"].tensor.numpy()
            obj_point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)
            input_dict["obj_point_indices"] = obj_point_indices
        else:
            input_dict["obj_point_indices"] = None
        return input_dict


@PIPELINES.register_module()
class RemoveSelfCenter(object):
    def __init__(self, radius=2):
        self.radius = radius

    def __call__(self, input_dict):

        dist = np.linalg.norm(input_dict["points"].tensor.numpy()[:, :3], axis=1)
        keep_indices = np.where(dist > self.radius)[0]
        input_dict["points"] = input_dict["points"][keep_indices]
        if input_dict["is_key_frame"]:
            if "gt_pts_semantic_mask" in  input_dict["ann_info"] and input_dict["ann_info"]["gt_pts_semantic_mask"] is not None:
                input_dict["ann_info"]["gt_pts_semantic_mask"] = input_dict["ann_info"][
                "gt_pts_semantic_mask"
                ][keep_indices]
        return input_dict


@PIPELINES.register_module()
class GetGround(object):
    def __init__(self):
        params = pypatchworkpp.Parameters()
        self.PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)

    def __call__(self, input_dict):
        points = input_dict["points"].tensor.numpy()[:, :4]
        self.PatchworkPLUSPLUS.estimateGround(points)
        nonground_idx = self.PatchworkPLUSPLUS.getNongroundIndices()
        ground_idx = self.PatchworkPLUSPLUS.getGroundIndices()
        input_dict["nonground_idx"] = nonground_idx
        input_dict["ground_idx"] = ground_idx

        return input_dict


@PIPELINES.register_module()
class GlobalAlignmentwithGT:
    def __init__(self):
        pass

    def __call__(self, input_dict):
        assert (
            "axis_align_matrix" in input_dict["ann_info"].keys()
        ), "axis_align_matrix is not provided in GlobalAlignment"

        lidar2start = input_dict["ann_info"]["axis_align_matrix"]
        assert lidar2start.shape == (
            4,
            4,
        ), f"invalid shape {lidar2start.shape} for axis_align_matrix"
        rot_mat = lidar2start[:3, :3]
        trans_vec = lidar2start[:3, -1]

        # self._check_rot_mat(rot_mat)
        if "gt_bboxes_3d" in input_dict and len(input_dict["gt_bboxes_3d"].tensor) != 0:
            points, rot_mat_T = input_dict["gt_bboxes_3d"].rotate(
                rot_mat.T, input_dict["points"]
            )
            input_dict["gt_bboxes_3d"].translate(trans_vec)
            points.translate(trans_vec)
            input_dict["points"] = points

        else:
            # self._rot_points(input_dict, rot_mat)
            input_dict["points"].rotate(rot_mat.T)
            input_dict["points"].translate(trans_vec)
            # self._trans_points(input_dict, trans_vec)

        input_dict["pcd_rotation"] = rot_mat_T
        input_dict["pcd_trans"] = trans_vec

        return input_dict


@PIPELINES.register_module()
class PtsSegment:

    def __init__(
        self,
        checkpoint_path,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
        num_classes=19,
        # keys=("coord", "normal"),
    ):
        self.grid_size = grid_size
        self.hash_type = hash_type
        self.mode = mode
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement
        # self.keys = keys

        model_config = dict(
            type="DefaultSegmentorV2",
            # num_classes=16,
            num_classes=num_classes,
            backbone_out_channels=64,
            backbone=dict(
                type="PT-v3m1",
                in_channels=4,
                order=["z", "z-trans", "hilbert", "hilbert-trans"],
                stride=(2, 2, 2, 2),
                enc_depths=(2, 2, 2, 6, 2),
                enc_channels=(32, 64, 128, 256, 512),
                enc_num_head=(2, 4, 8, 16, 32),
                enc_patch_size=(1024, 1024, 1024, 1024, 1024),
                dec_depths=(2, 2, 2, 2),
                dec_channels=(64, 64, 128, 256),
                dec_num_head=(4, 4, 8, 16),
                dec_patch_size=(1024, 1024, 1024, 1024),
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                attn_drop=0.0,
                proj_drop=0.0,
                drop_path=0.3,
                shuffle_orders=True,
                pre_norm=True,
                enable_rpe=False,
                enable_flash=True,
                upcast_attention=False,
                upcast_softmax=False,
                cls_mode=False,
                pdnorm_bn=False,
                pdnorm_ln=False,
                pdnorm_decouple=True,
                pdnorm_adaptive=False,
                pdnorm_affine=True,
                pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
            ),
        )
        # seg_map = {
        #     "bicycle": "bicycle",
        #     "building": "manmade",
        #     "bus": "bus",
        #     "car": "car",
        #     "cone": "traffic_cone",
        #     "crowd": "pedestrian",
        #     "curbside": "sidewalk",
        #     "fence": "barrier",
        #     "motorcycle": "motorcycle",
        #     "other_ground": "other_flat",
        #     "other_object": "barrier",
        #     "other_structure": "manmade",
        #     "pedestrian": "pedestrian",
        #     "pole": "barrier",
        #     "road": "driveable_surface",
        #     "tree": "vegetation",
        #     "tricycle": "bicycle",
        #     "truck": "truck",
        #     "vegetation": "vegetation",
        # }
        # self.seg_index_map = {
        #     i: NUSC_COARSE2IDX[v]
        #     for i, (k, v) in enumerate(seg_map.items())
        # }

        # names = [
        #     "barrier",
        #     "bicycle",
        #     "bus",
        #     "car",
        #     "construction_vehicle",
        #     "motorcycle",
        #     "pedestrian",
        #     "traffic_cone",
        #     "trailer",
        #     "truck",
        #     "driveable_surface",
        #     "other_flat",
        #     "sidewalk",
        #     "terrain",
        #     "manmade",
        #     "vegetation",
        # ]
        # self.seg_index_map = {i:i + 1 for i, v in enumerate(names)} # ignore 0
        seg_map = {
            "bicycle": 5,
            "building": 8,
            "bus": 2,
            "car": 0,
            "cone": 9,
            "crowd": 10,
            "curbside": 11,
            "fence": 12,
            "motorcycle": 4,
            "other_ground": 13,
            "other_object": 14,
            "other_structure": 15,
            "pedestrian": 7,
            "pole": 16,
            "road": 17,
            "tree": 18,
            "tricycle": 6,
            "truck": 1,
            "vegetation": 19,
        }
        self.seg_index_map = {i: v for i, (k, v) in enumerate(seg_map.items())}
        self.seg_index_map_v = np.vectorize(self.seg_index_map.get)
        self.model = build_model(model_config).cuda()
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage.cuda())
        weight = OrderedDict()
        for key, value in checkpoint["state_dict"].items():
            weight[key.replace("module.", "")] = value
        load_state_info = self.model.load_state_dict(weight, strict=True)
        self.model.eval()
        print(load_state_info)
        # self.transform_pipeline = [
        #     dict(type="PointClip", point_cloud_range=(-80, -60, -3, 80, 60, 6)),
        #     dict(type="RemoveGround", ),
        #     # dict(type="FilterbyName", names=names, filter_name=od_objects ),
        #     # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
        #     # dict(type="RandomScale", scale=[1, 1]),
        #     # dict(type="RandomFlip", p=0.5),
        #     # dict(type="RandomJitter", sigma=0.005, clip=0.02),
        #     dict(type="Add", keys_dict={"condition": "nuScenes"}),
        #     dict(type="GridSample",grid_size=0.05,hash_type="fnv",mode="train",keys=("coord", "strength"),return_grid_coord=True,),
        #     dict(type="ToTensor"),
        #     dict(type="Collect",keys=("coord", "grid_coord", "condition"),feat_keys=("coord", "strength"),),
        # ]
    @torch.inference_mode()
    def __call__(self, input_dict):
        points = input_dict["points"].tensor.numpy()[:, :4]
        non_ground_points = points[input_dict["nonground_idx"]]
        data = dict()
        grid_infos = self.grid_sample(non_ground_points)
        data["coord"] = non_ground_points[:, :3][grid_infos['idx_unique']]
        data["strength"] = non_ground_points[:, 3:4][
            grid_infos['idx_unique']] / 255
        data["grid_coord"] = grid_infos['grid_coord']
        data['feat'] = np.concatenate([data["coord"], data["strength"]],
                                      axis=-1)
        data['offset'] = np.array([data["coord"].shape[0]])
        for k, v in data.items():
            data[k] = torch.from_numpy(v).cuda(non_blocking=True)
        data['condition'] = "nuScenes"
        output = self.model(data)
        pred = output["seg_logits"].argmax(dim=-1).data.cpu().numpy()
        pred = self.seg_index_map_v(pred)
        input_dict["seg_logits"] = pred[grid_infos['inverse']]
        return input_dict

    def grid_sample(self, points):
        # points = data_dict["points"].tensor.numpy()[:, :3]
        output = dict()
        scaled_coord = points[:, :3] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort,
                                      return_inverse=True,
                                      return_counts=True)
        idx_select = (np.cumsum(np.insert(count, 0, 0)[0:-1]) +
                      np.random.randint(0, count.max(), count.size) % count)
        idx_unique = idx_sort[idx_select]
        output["idx_unique"] = idx_unique
        # if "sampled_index" in data_dict:
        #     # for ScanNet data efficient, we need to make sure labeled point is sampled.
        #     idx_unique = np.unique(
        #         np.append(idx_unique, data_dict["sampled_index"]))
        #     mask = np.zeros_like(data_dict["segment"]).astype(bool)
        #     mask[data_dict["sampled_index"]] = True
        #     data_dict["sampled_index"] = np.where(mask[idx_unique])[0]
        if self.return_inverse:
            output["inverse"] = np.zeros_like(inverse)
            output["inverse"][idx_sort] = inverse
        if self.return_grid_coord:
            output["grid_coord"] = grid_coord[idx_unique]
        if self.return_min_coord:
            output["min_coord"] = min_coord.reshape([1, 3])
        if self.return_displacement:
            displacement = (scaled_coord - grid_coord - 0.5
                            )  # [0, 1] -> [-0.5, 0.5] displacement to center
            if self.project_displacement:
                displacement = np.sum(displacement * output["normal"],
                                      axis=-1,
                                      keepdims=True)
            output["displacement"] = displacement[idx_unique]
        # for key in self.keys:
        # output[key] = output[key][idx_unique]
        return output

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0],
                                                               dtype=np.uint64)
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys
