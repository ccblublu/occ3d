from collections import defaultdict
from os import path as osp
import pyquaternion
from scipy.spatial.transform import Rotation as R



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

from mmdetection3d.tools.misc.browse_dataset import show_det_data, show_result
# @DATASETS.register_module()

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
            out.append(info)
            for sweep in sweeps:
                sweep['lidar_path'] = sweep['data_path']    
                sweep['lidar2ego_translation'] = sweep['sensor2ego_translation']
                sweep['lidar2ego_rotation'] = sweep['sensor2ego_rotation']
                out.append(sweep)
        return out
    
    def __getitem__(self, index):
        return self.data_infos[index] 

    def __len__(self):
        return len(self.data_infos)
    
    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
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
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - ann_info (dict): Annotation info.
        """
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
        return input_dict
    
    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
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

def main(data_path, info_path):
    dataset = SeqNuscene(data_path, info_path, nuscenes_version="v1.0-mini")
    for i in track_iter_progress(list(range(len(dataset)))):
        seq_dataset = dataset[i]
        for j in track_iter_progress(list(range(len(seq_dataset)))):
            input_dict = seq_dataset.get_data_info(j)
            seq_dataset.pre_pipeline(input_dict)
            example = seq_dataset.pipeline(input_dict)        
            annos = example['ann_info']
            image_idx = example['sample_idx']
            points = example['points'].tensor.numpy()
            gt_boxes_3d = annos['gt_bboxes_3d'].tensor
            names = annos['gt_names']
            show_result(points=points, gt_bboxes=gt_boxes_3d.clone(), pred_bboxes=None, out_dir="./viz", filename=example['sample_idx'], show=True, snapshot=True)



if __name__ == '__main__':
    from mmdetection3d.tools.data_converter.nuscenes_converter import create_nuscenes_infos
    create_nuscenes_infos(
        '/media/chen/Elements/nuScenes/raw/Trainval/',
        info_prefix="test",)
        # version='v1.0-mini',)

    # main('/media/chen/data/nusecnes/v1.0-mini/', 'test_infos_train.pkl')