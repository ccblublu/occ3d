
import numpy as np
from mmdet3d.datasets.builder import PIPELINES
from mmdet3d.core.bbox import box_np_ops
import pypatchworkpp


@PIPELINES.register_module()
class GetBoxPointIndices:
    def __init__(self):
        pass

    def __call__(self, input_dict):
        if 'gt_bboxes_3d' in input_dict:
            gt_boxes_3d = input_dict["gt_bboxes_3d"].tensor.numpy()
            points = input_dict['points'].tensor.numpy()
            obj_point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)
            input_dict['obj_point_indices'] = obj_point_indices
        else:
            input_dict['obj_point_indices'] = None
        return input_dict
    
@PIPELINES.register_module()
class RemoveSelfCenter(object):
    def __init__(self, radius=2):
        self.radius = radius
    
    def __call__(self, input_dict):
        
        dist = np.linalg.norm(input_dict["points"].tensor.numpy()[:,:3], axis=1)
        keep_indices = np.where(dist > self.radius)[0]
        input_dict['points'] = input_dict['points'][keep_indices]
        if input_dict['is_key_frame']:
            input_dict["ann_info"]['gt_pts_semantic_mask'] = input_dict["ann_info"]['gt_pts_semantic_mask'][keep_indices]
        return input_dict
    
@PIPELINES.register_module()
class GetGround(object):
    def __init__(self):
        params = pypatchworkpp.Parameters()
        self.PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)
        

    def __call__(self, input_dict):
        points = input_dict['points'].tensor.numpy()[:, :4]
        self.PatchworkPLUSPLUS.estimateGround(points)
        nonground_idx = self.PatchworkPLUSPLUS.getNongroundIndices()
        ground_idx = self.PatchworkPLUSPLUS.getGroundIndices()
        input_dict['nonground_idx'] = nonground_idx
        input_dict['ground_idx'] = ground_idx

        return input_dict
    
@PIPELINES.register_module()
class GlobalAlignmentwithGT:
    def __init__(self):
        pass

    def __call__(self, input_dict):
        assert 'axis_align_matrix' in input_dict['ann_info'].keys(), \
            'axis_align_matrix is not provided in GlobalAlignment'

        lidar2start = input_dict['ann_info']['axis_align_matrix']
        assert lidar2start.shape == (4, 4), \
            f'invalid shape {lidar2start.shape} for axis_align_matrix'
        rot_mat = lidar2start[:3, :3]
        trans_vec = lidar2start[:3, -1]

        # self._check_rot_mat(rot_mat)
        if 'gt_bboxes_3d' in input_dict and \
                len(input_dict['gt_bboxes_3d'].tensor) != 0:
            points, rot_mat_T = input_dict['gt_bboxes_3d'].rotate(
                rot_mat.T, input_dict['points'])
            input_dict['gt_bboxes_3d'].translate(trans_vec)
            points.translate(trans_vec)
            input_dict['points'] = points

        else:
            # self._rot_points(input_dict, rot_mat)
            input_dict['points'].rotate(rot_mat.T)
            input_dict['points'].translate(trans_vec)
            # self._trans_points(input_dict, trans_vec)

        input_dict['pcd_rotation'] = rot_mat_T
        input_dict['pcd_trans'] = trans_vec
        
        return input_dict
