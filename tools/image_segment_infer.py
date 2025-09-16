import os
from pathlib import Path 
import pickle
from tqdm import tqdm 

from nuscenes import NuScenes
import numpy as np
from ops.segmentation.image_segmentaiton import init_model
from mmseg.apis import inference_segmentor

cv_seg_model = init_model()

data_root = Path("/media/chen/data/nusecnes/v1.0-mini/")
anno_file = "test_infos_train.pkl"
with open(data_root / anno_file, "rb")as f:
    anno = pickle.load(f)['infos']
nuscenes = NuScenes(data_root.name, data_root)
for frame in tqdm(anno):
    sample = nuscenes.get("sample", frame['token'])
    seq_token = sample["scene_token"]
    out_path = data_root / f"seg_pred/{seq_token}"
    out_path.mkdir(parents=True, exist_ok=True)
    for cam_type, cam_info in frame['cams'].items():
        result = inference_segmentor(cv_seg_model, cam_info["data_path"])[0]
        file_path = Path("")
        np.save(out_path / Path(cam_info["data_path"]).name, result)