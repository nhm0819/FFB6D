import argparse
from datasets.neuromeka.neuromeka_dataset import Dataset as NM_Dataset
from common import Config, ConfigRandLA
import models.pytorch_utils as pt_utils
from utils.basic_utils import Basic_Utils

import os
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import tqdm

parser = argparse.ArgumentParser(description="Arg parser")

parser.add_argument("-cls", type=str, default='doorstop')
parser.add_argument("-batch_size", type=int, default=1)
parser.add_argument(
    '--dataset_dir', type=str, default="/mnt/data"
)
parser.add_argument('--n_keypoints', type=int, default=50)

args = parser.parse_args()

config = Config(ds_name='neuromeka', dataset_dir="/home/nhm/work/FFB6D/ffb6d/datasets", cls_type=args.cls,
                batch_size=1, cad_file='ply', kps_extractor='SIFT', n_keypoints=args.n_keypoints)
bs_utils = Basic_Utils(config)

if __name__ == "__main__":
    ds_type = 'real'  # render, fuse, real
    obj_id = config.neuromeka_obj_dict[args.cls]
    test_ds = NM_Dataset(ds_type, cls_type=args.cls, num_renders=150, num_fuse=150)

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=1
    )
    # K = np.array([[700., 0., 320.],
    #               [0., 700., 240.],
    #               [0., 0., 1.]])

    K = np.array([[610.70306, 0., 319.78845],
                  [0., 610.40942, 242.05362],
                  [0., 0., 1.0]], np.float32)

    folder_name = f"/home/nhm/work/FFB6D/ffb6d/datasets/neuromeka/EDA/{ds_type}/{args.cls}"
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name, exist_ok=True)

    for i, cu_dt in tqdm.tqdm(
        enumerate(test_loader), leave=False, desc="val"
    ):
        K = cu_dt['K'].cpu().numpy()[0]
        show_kp_img = cu_dt['rgb'].cpu().numpy().astype("uint8")[0].transpose(1, 2, 0).copy()
        rotation_matrix = cu_dt["RTs"][0][0].cpu().numpy()
        # kp_3ds = cu_dt["kp_3ds"][0][0].cpu().numpy()
        # kp_3ds = np.append(kp_3ds, cu_dt["ctr_3ds"][0][0].cpu().unsqueeze(0).numpy(), axis=0)
        # # kp_3ds = np.dot(kp_3ds, rotation_matrix[:, :3].T) + rotation_matrix[:, 3]
        # kp_2ds = bs_utils.project_p3d(kp_3ds, 1.0, K=K)
        # # print("kp3d:", cu_dt["kp_3ds"][0][0])
        # # print("kp2d:", kp_2ds, "\n")
        # color = (0, 0, 255)  # bs_utils.get_label_color(cls_id.item())
        # show_kp_img = bs_utils.draw_p2ds(show_kp_img, kp_2ds, r=1, color=color)

        mesh_pts = bs_utils.get_pointxyz(obj_id, ds_type="neuromeka").copy()
        mesh_pts = np.dot(mesh_pts, rotation_matrix[:, :3].T) + rotation_matrix[:, 3]
        mesh_p2ds = bs_utils.project_p3d(mesh_pts, 1.0, K)
        # color = bs_utils.get_label_color(obj_id, n_obj=2, mode=2)
        color = (0, 255, 0)
        show_kp_img = bs_utils.draw_p2ds(show_kp_img, mesh_p2ds, color=color)

        file_name = os.path.join(folder_name, f'{i}.png')
        cv2.imwrite(file_name, show_kp_img[:,:,::-1])
