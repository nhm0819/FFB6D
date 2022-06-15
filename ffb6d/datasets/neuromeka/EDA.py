import csv
import glob
import os
import numpy as np

from ffb6d.datasets.neuromeka.preprocs import read_view, read_objects, calc_Tco



label = "bottle"
folder_list = sorted(glob.glob(f"/home/nhm/work/data/neuromeka/{label}/*"))

all_poses = {}
lens_f = []
sensor_dim = []
offset_px = []
lens_f_px = []

for i, folder in enumerate(folder_list):
    poses_files = sorted(glob.glob(folder+"/poses/*.csv"))
    poses = [read_view(filepath) for filepath in poses_files]

    for pose in poses:
        lens_f.append(pose['lens_f'])
        sensor_dim.append(pose['sensor_dim'])
        offset_px.append(pose['offset_px'])
        lens_f_px.append(pose['lens_f_px'])

    pose_dict = {
        f"lens_f_{i}" : np.array(lens_f),
        f"sensor_dim_{i}": np.array(sensor_dim),
        f"offset_px_{i}": np.array(offset_px),
        f"lens_f_px_{i}": np.array(lens_f_px),
    }

    all_poses = {**all_poses, **pose_dict}

np_lens_f_px = np.array(lens_f_px)
np_lens_f = np.array(lens_f)



label = "bottle"
folder_list = sorted(glob.glob(f"/home/nhm/work/data/neuromeka/{label}/*"))

all_configs = {}
cls = []
scale = []
pos = []
eu_zyx_rev = []
eu_zyx = []
for i, folder in enumerate(folder_list):
    config_files = sorted(glob.glob(folder+"/config/*.csv"))
    configs = [read_objects(filepath) for filepath in config_files]

    for config in configs:
        cls.append(config[0]['cls'])
        scale.append(config[0]['scale'])
        pos.append(config[0]['pos'])
        eu_zyx_rev.append(config[0]['eu_zyx_rev'])
        eu_zyx.append(config[0]['eu_zyx'])

    config_dict = {
        f"cls_{i}" : np.array(cls),
        f"scale_{i}": np.array(scale),
        f"pos_{i}": np.array(pos),
        f"eu_zyx_rev_{i}": np.array(eu_zyx_rev),
        f"eu_zyx_{i}": np.array(eu_zyx),
    }

    all_configs = {**all_configs, **config_dict}

np_scale = np.array(scale)
np_pos = np.array(pos)
###################################3
import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, sin, tan

zNear = float(0.1)
zFar = float(100.0)
DEPTH_SCALE_HQ = 50000

Tx180 = np.identity(4, 'float32')
Tx180[1, 1] = -1
Tx180[2, 2] = -1


######################################################3
Tcos = []
for i in range(100):
    obj = configs[i][0]
    view = poses[i]
    Tcos.append(np.matmul(Tx180, calc_Tco(obj, view)))


### train list
import glob
import os
from sklearn.model_selection import train_test_split

label_list = ["bottle", "car", "doorstop"]

for label in label_list:
    image_list = glob.glob(os.path.join("ffb6d", f"datasets/neuromeka/{label}/*/*.png"))
    image_list = ['/'.join(image_list[i].split("/")[3:])[:-4] for i in range(len(image_list))]
    len(image_list)

    train, test = train_test_split(image_list, test_size=0.5)

    train_path = f"ffb6d/datasets/neuromeka/{label}/{label}_train_list.txt"
    with open(train_path, 'w+') as lf:
        lf.write('\n'.join(train))

    test_path = f"ffb6d/datasets/neuromeka/{label}/{label}_test_list.txt"
    with open(test_path, 'w+') as lf:
        lf.write('\n'.join(test))


#########################
import pickle as pkl
import matplotlib.pyplot as plt
cls_list = ['bottle', 'car', 'doorstop']

dataset = {}
for cls in cls_list:
    file = open(f"/home/nhm/work/FFB6D/ffb6d/datasets/neuromeka/fuse/{cls}/1.pkl", 'rb')
    data = pkl.load(file)
    dataset[cls] = data

plt.imshow(dataset['car']['rgb'])
