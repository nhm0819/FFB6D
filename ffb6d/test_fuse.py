import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
from PIL import Image

file_list = glob.glob("/home/nhm/work/FFB6D-master/ffb6d/datasets/linemod/test/data/01/rgb/*.png")
file_list.sort()

file_name = file_list[0]

with Image.open(file_name) as ri:
    rgb = np.array(ri)[:, :, :3]
with Image.open(file_name.replace("rgb", "depth")) as di:
    dpt = np.array(di)
with Image.open(file_name.replace("rgb", "mask")) as li:
    labels = np.array(li)
    msk = (labels > 0).astype("uint8")
    rev = (labels == 0).astype("uint8")

rgb_render = rgb * msk
dpt_render = dpt * msk[..., 0]

# plt.imshow(msk)

len_img = len(file_list)
for idx, file_name in enumerate(file_list):

    with Image.open(file_name) as ri:
        rgb = np.array(ri)[:, :, :3]
    with Image.open(file_name.replace("rgb", "depth")) as di:
        dpt = np.array(di)
    with Image.open(file_name.replace("rgb", "mask")) as li:
        labels = np.array(li)
        msk = (labels > 0).astype("uint8")
        rev = (labels == 0).astype("uint8")

    rgb_render = rgb * msk
    dpt_render = dpt * msk[...,0]

    globals()[f"rgb_{idx}"] = rgb
    globals()[f"dpt_{idx}"] = dpt
    globals()[f"msk_{idx}"] = msk
    globals()[f"rev_{idx}"] = rev
    globals()[f"rgb_render_{idx}"] = rgb_render
    globals()[f"dpt_render_{idx}"] = dpt_render



new_msk = msk_0 + msk_1 + msk_2 + msk_3
new_msk = new_msk*255
new_rgb = rgb_0 * rev_0 * rev_1 * rev_2 * rev_3
new_rgb = new_rgb + rgb_render_0 + rgb_render_1 + rgb_render_2 + rgb_render_3
new_dpt = dpt_0 * rev_0[...,0] * rev_1[...,0] * rev_2[...,0] * rev_3[...,0]
new_dpt = new_dpt + dpt_render_0 + dpt_render_1 + dpt_render_2 + dpt_render_3

# plt.imshow(new_dpt*255)

save_msk = Image.fromarray(new_msk)
save_msk.save("/home/nhm/work/FFB6D-master/ffb6d/datasets/linemod/test/data/01/fuse/msk.png", "png")
save_rgb = Image.fromarray(new_rgb)
save_rgb.save("/home/nhm/work/FFB6D-master/ffb6d/datasets/linemod/test/data/01/fuse/rgb.png", "png")
save_dpt = Image.fromarray(new_dpt)
save_dpt.save("/home/nhm/work/FFB6D-master/ffb6d/datasets/linemod/test/data/01/fuse/dpt.png", "png")

