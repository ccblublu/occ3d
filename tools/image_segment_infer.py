import os
from pathlib import Path
import pickle
import cv2
from tqdm import tqdm

from nuscenes import NuScenes
import numpy as np
from ops.segmentation.image_segmentaiton import init_model
from mmseg.apis import inference_segmentor
from utils.ops import generate_35_category_colors

cv_seg_model = init_model()

data_root = Path("/media/chen/090/nuScenes")
anno_file = "test_infos_val.pkl"
with open(data_root / anno_file, "rb")as f:
    anno = pickle.load(f)['infos']
nuscenes = NuScenes("v1.0-trainval", data_root)
for frame in tqdm(anno):
    sample = nuscenes.get("sample", frame['token'])
    seq_token = sample["scene_token"]
    out_path = data_root / f"seg_pred/{seq_token}"
    out_path.mkdir(parents=True, exist_ok=True)
    for cam_type, cam_info in frame['cams'].items():
        if (out_path / f"{Path(cam_info['data_path']).name}.npy").exists():
            continue
        result = inference_segmentor(cv_seg_model, cam_info["data_path"])[0]
        file_path = Path("")
        np.save(out_path / Path(cam_info["data_path"]).name, result)



# data_root = Path(
#     "/media/chen/090/train_data/GACRT014_1729079698/raw_data/image_undistortion"
# )
# for file in data_root.glob("*/*.jpg"):
#     out_path = data_root.parent / f"seg_pred/{file.parent.name}"
#     if (out_path / file.name).exists():
#         continue
#     cam_type = file.parent.name
#     if cam_type in ["front_wide", "front_narrow"]:
#         continue
#     result = inference_segmentor(cv_seg_model, file)[0]
#     image = cv_seg_model.show_result(
#         str(file), [result],
#         palette=generate_35_category_colors(len(cv_seg_model.CLASSES)) * 255,
#         show=False)
#     out_path.mkdir(parents=True, exist_ok=True)
#     cv2.imwrite(str(out_path / file.name), image)
