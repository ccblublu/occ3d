from load_nuscenes import *


class SeqOntime(SeqNuscene):

    def __init__(self, data_root, anno_file):
        # super().__init__(data_root, anno_file)#! ignore nuscenes
        self.data_root = data_root
        self.anno_file = anno_file
        anno_data = mmcv_load(osp.join(data_root, anno_file))
        self.anno_metadata = anno_data["metainfo"]
        self.anno_infos = anno_data["data_list"]
        self.seq_data = self.get_seq_data()

    def get_seq_data(self):
        seq_data = defaultdict(list)
        for id, info in enumerate(self.anno_infos):
            token = info["token"]
            scene_token = token.split("-")[0]
            seq_data[scene_token].append(info)
        seq_data = dict(seq_data)
        out = []
        self.seq_token = []
        for key, value in seq_data.items():
            out.append(self.init_dataset(value, key))
            self.seq_token.append(key)
        return out

    def init_dataset(self, data, token):
        dataset_cfg = dict(type="SequenceDatasetOntime", data=data, token=token)
        dataset_cfg.update(
            pipeline=[
                dict(type="LoadPointsFromFile",
                     coord_type="LIDAR",
                     load_dim=5,
                     use_dim=5),
                dict(type="PointsRangeFilter", point_cloud_range=[-200,-200,-10,200,200,10]),
                dict(type="RemoveSelfCenter", radius=2.5),
                dict(type="GetGround"),
                dict(type="LoadAnnotations3D",
                     with_bbox_3d=True,
                     with_label_3d=True),
                dict(type="GetBoxPointIndices"),
                dict(
                    type="PtsSegment",
                    checkpoint_path="/home/chen/workspace/points_segmentation/PointTransformerV3/Pointcept/runs/ontime_wo_ground/model/model_best.pth",
                    # checkpoint_path="/media/chen/workspace/workspace/points_segmentation/PointTransformerV3/Pointcept/models/model_best.pth", # nuscenes
                    grid_size=0.05,
                    hash_type="fnv",
                    mode="train",
                    return_inverse=True,
                    return_grid_coord=True,
                    return_min_coord=False,
                    return_displacement=False,
                    project_displacement=False,
                ),
                dict(type="GlobalAlignmentwithGT"),  #! 自由度太少，框有误差，先单帧取点云再拼接
            ])
        dataset = build_dataset(dataset_cfg)
        return dataset


class Ontime2Occ3D(Nuscenes2Occ3D):

    def __init__(self, data_path):
        super().__init__(data_path)

    def main(self, info_path):
        self.dataset = SeqOntime(self.data_path, info_path)
        for i in track_iter_progress(list(range(len(self.dataset)))):
            self.seq_token = self.dataset.get_token(i)
            self.sqe_save_path = self.save_path / self.seq_token
            self.sqe_save_path.mkdir(exist_ok=True, parents=True)
            self.seq_run(i)

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
        points = example["points"].tensor.numpy()[:, :3].astype(
            np.float32)  #! (n, 3)
        gt_boxes_3d = annos["gt_bboxes_3d"].tensor
        names = annos["gt_names"]
        gt_boxes_id = annos["gt_bboxes_id"]
        obj_point_indices = example["obj_point_indices"]
        # obj_point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d.numpy())

        #! 区分地面/非地面, 前景(标注目标)/背景点云
        background_points_indices = np.where(~obj_point_indices.any(axis=1))[0]
        # ground_points_indices = example["ground_idx"]
        nonground_points_indices = example["nonground_idx"]
        # background_and_ground_points_indices = np.intersect1d(
        #     ground_points_indices, background_points_indices, assume_unique=True
        # )
        # background_and_nonground_points_indices = np.intersect1d(
        #     nonground_points_indices, background_points_indices, assume_unique=True
        # )
        if "gt_pts_semantic_mask" in annos:
            gt_pts_semantic_mask = annos["gt_pts_semantic_mask"]  #!(n, )
            if gt_pts_semantic_mask is None:
                gt_pts_semantic_mask = np.zeros(len(points), dtype=np.uint8)
            points = np.concatenate((points, gt_pts_semantic_mask[:, None]),
                                    axis=1)
            if not example["is_key_frame"]:
                closed_key_frame_idx = np.argmin(
                    np.abs(self.key_frame_timestamps - timestamp))
                keyframe_timesamp = self.key_frame_timestamps[
                    closed_key_frame_idx]

                if keyframe_timesamp not in self.background_points_bank:
                    self.waiting_for_key_frame[keyframe_timesamp].append(
                        timestamp)
                else:
                    keyframe_points = self.background_points_bank[
                        keyframe_timesamp]
                    keyframe_labels = keyframe_points[:, 3]
                    labels = propagate_labels_with_knn(keyframe_points[:, :3],
                                                       keyframe_labels,
                                                       points[:, :3])
                    points[:, 3] = labels
            elif timestamp in self.waiting_for_key_frame:
                non_key_frames = self.waiting_for_key_frame.pop(timestamp)
                for non_key_frames_timestamp in non_key_frames:
                    non_keyframe_points = self.background_points_bank[
                        non_key_frames_timestamp]
                    non_keyframe_labels = propagate_labels_with_knn(
                        points[background_points_indices, :3],
                        points[background_points_indices, 3],
                        non_keyframe_points[:, :3],
                    )
                    self.background_points_bank[
                        non_key_frames_timestamp][:, 3] = non_keyframe_labels
        if "seg_logits" in example:
            seg_logits = example["seg_logits"] + 1
            pred_sem = np.full_like(points[:, 0:1], 18,
                                    dtype=np.int32)  # full ground idx
            pred_sem[nonground_points_indices] = seg_logits.reshape(-1, 1)
            points = np.concatenate((points, pred_sem), axis=1)
        self.background_points_bank[timestamp] = points[
            background_points_indices]
        self.cams_bank[timestamp] = example["cams"]
        # show_result(points=points, gt_bboxes=gt_boxes_3d.clone(), pred_bboxes=None, out_dir="./viz", filename=example['sample_idx'], show=True, snapshot=True)
        self.obj_info_bank[timestamp] = []
        for i, (obj_id, box) in enumerate(zip(gt_boxes_id,
                                              gt_boxes_3d.numpy())):
            self.obj_info_bank[timestamp].append({
                "obj_id": obj_id,
                "box": box
            })
            if obj_id not in self.obj_points_bank:
                self.obj_points_bank[obj_id] = {
                    "points": [],
                    "size": box[3:6],
                    "label": names[i] + 1,
                }
            gt_points = points[obj_point_indices[:, i]]
            gt_points[:, :3] -= box[:3]
            gt_points[:, :3] = gt_points[:, :3] @ rotate_yaw(box[6]).T
            self.obj_points_bank[obj_id]["points"].append(gt_points)
            self.obj_points_bank[obj_id]["size"] = np.maximum(
                self.obj_points_bank[obj_id]["size"], box[3:6])
        return

    def filter_dynamic_points(self, points, batch_id, idx=-1):
        # mask = np.logical_or(points[:, idx] >= 1, points[:, idx] <= 8)
        mask = points[:, idx] >= 8 + 1
        return points[mask], batch_id[mask]

@DATASETS.register_module()
class SequenceDatasetOntime(SequenceDataset):

    def __init__(self, data, token, pipeline=None, **kwargs):
        super().__init__(data, token, pipeline, **kwargs)

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
            sweeps = info.pop("lidar_sweeps")
            info["annos"] = {
                "gt_boxes": [],
                "gt_names": [],
                "gt_inst_token": [],
            }
            for item in info['instances']:
                info["annos"]["gt_boxes"].append(item['bbox_3d'])
                info["annos"]["gt_names"].append(item['bbox_label_3d'])
                info["annos"]["gt_inst_token"].append(item['obj_ids'])
            info['lidar_path'] = info['lidar_points']['lidar_path']
            info["is_key_frame"] = True
            out.append(info)
            # for sweep in sweeps:
            #     # ? 排查timestamp重叠的问题: nuscenes数据集中的sweeps的最后一帧为关键帧，需要额外剔除
            #     if sweep["timestamp"] in key_frame_timestamps:
            #         continue
            #     sweep['token'] = sweep['sample_data_token'].replace("-lidar-", "-")
            #     sweep["lidar_path"] = sweep["lidar_points"]['lidar_path']
            #     sweep["ego2global"] = sweep["lidar_points"]["ego2global"]
            #     # sweep["lidar2ego_rotation"] = sweep["sensor2ego_rotation"]
            #     sweep["is_key_frame"] = False
            #     out.append(sweep)
        return out

    def get_ego2global(self, index):
        ego2global = self.data_infos[index]["ego2global"]
        return ego2global

    def get_lidar2ego(self, index):
        lidar2ego = np.eye(4)
        return lidar2ego

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
            cams=info.get("images", None),
        )

        annos = self.get_ann_info(index)
        input_dict["ann_info"] = annos

        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        gt_bboxes_3d = np.asarray(info["annos"]["gt_boxes"])
        gt_names_3d = info["annos"]["gt_names"]
        gt_bboxes_id = info["annos"]["gt_inst_token"]
        # if info["is_key_frame"]:
        # gt_pts_semantic_mask = np.fromfile(info["pts_semantic_mask_path"], dtype=np.uint8)
        # gt_pts_semantic_mask = self.fineidx2coarseidx(gt_pts_semantic_mask)
        # gt_pts_semantic_mask = None
        # else:
        #     gt_pts_semantic_mask = None
        # turn original box type to target box type
        gt_bboxes_3d = LiDARInstance3DBoxeswithID(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            ids=gt_bboxes_id,
            origin=(0.5, 0.5, 0.5),
        ).convert_to(0)  #! (0.5, 0.5, 0),

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_names_3d,
            gt_bboxes_id=gt_bboxes_id,
            gt_names=gt_names_3d,
            axis_align_matrix=info["axis_align_matrix"],
            # gt_pts_semantic_mask=gt_pts_semantic_mask,
        )
        return anns_results
if __name__ == "__main__":
    ontime2occ = Ontime2Occ3D("/media/chen/090/train_data")
    ontime2occ.main("GACRT014_1729079698_infos.pkl")
