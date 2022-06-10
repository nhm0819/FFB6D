import os
import time
import cv2
import pickle
import yaml
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import ImageFile, Image
from plyfile import PlyData
from concurrent.futures import ProcessPoolExecutor
from argparse import ArgumentParser
try:
    from neupeak.utils.webcv2 import imshow, waitKey
except:
    from cv2 import imshow, waitKey

from ffb6d.datasets.neuromeka.preprocs import read_objects, read_view, calc_Tco, combine_bytes_arr


parser = ArgumentParser()
parser.add_argument(
    "--cls", type=str, default="bottle",
    help="Target object from {bottle, car, doorstop} (default bottle)"
)
parser.add_argument(
    '--fuse_num', type=int, default=10000,
    help="Number of images you want to generate."
)
parser.add_argument(
    '--DEBUG', action="store_true",
    help="To show the generated images or not."
)
args = parser.parse_args()
DEBUG = args.DEBUG

Intrinsic_matrix = {
    'linemod': np.array([[572.4114, 0., 325.2611],
                          [0., 573.57043, 242.04899],
                          [0., 0., 1.]]),
    'blender': np.array([[700.,    0.,  320.],
                         [0.,  700.,  240.],
                         [0.,    0.,    1.]]),
    'neuromeka': np.array([[610.70306, 0., 319.78845],
                           [0., 610.40942, 242.05362],
                           [0., 0., 1.0]], np.float32)
}
neuromeka_obj_dict={
    'bottle':1,
    'car':2,
    'doorstop':3
}
root = '../ffb6d/datasets/neuromeka'
cls_root_ptn = os.path.join(root, '%s')


def ensure_dir(pth):
    if not os.path.exists(pth):
        os.system("mkdir -p {}".format(pth))


def read_lines(pth):
    with open(pth, 'r') as f:
        return [
            line.strip() for line in f.readlines()
        ]


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, pkl_path):
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)


def collect_train_info(cls_name):
    cls_id = neuromeka_obj_dict[cls_name]
    cls_root = cls_root_ptn % cls_name
    tr_pth = os.path.join(
        cls_root, f"{cls_name}_train_list.txt"
    )
    train_fns = read_lines(tr_pth)

    return train_fns


def collect_neuromeka_set_info(
        neuromeka_dir, neuromeka_cls_name, cache_dir='../ffb6d/datasets/neuromeka/cache'
):
    database = []
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    if os.path.exists(
            os.path.join(cache_dir,'{}_info.pkl').format(neuromeka_cls_name)
    ):
        return read_pickle(
            os.path.join(cache_dir,'{}_info.pkl').format(neuromeka_cls_name)
        )

    train_fns = collect_train_info(neuromeka_cls_name)
    cls_id = neuromeka_obj_dict[neuromeka_cls_name]
    cls_root = cls_root_ptn % neuromeka_cls_name

    print('begin generate database {}'.format(neuromeka_cls_name))

    for item in train_fns:
        cls_type, folder_name, file_name = item.split("/")
        dpt_ptn = os.path.join(cls_root, folder_name, "depthmaps", file_name + f"_00_{cls_id:02d}.png")
        msk_ptn = os.path.join(cls_root, folder_name, "masks", file_name + f"_00_{cls_id:02d}.png")
        rgb_ptn = os.path.join(cls_root, folder_name, file_name + ".png")

        data={}
        data['rgb_pth'] = rgb_ptn
        data['dpt_pth'] = dpt_ptn
        data['msk_pth'] = msk_ptn

        file_pose = os.path.join(os.path.join(cls_root, folder_name, "poses", file_name + ".csv"))
        file_config = os.path.join(os.path.join(cls_root, folder_name, "config", file_name + ".csv"))

        Tx180 = np.identity(4, 'float32')
        Tx180[1, 1] = -1
        Tx180[2, 2] = -1

        obj = read_objects(file_config)[0]
        view = read_view(file_pose)
        Tco = np.matmul(Tx180, calc_Tco(obj, view))
        RT = Tco[:3]

        data['RT'] = RT
        database.append(data)

    print(
        'successfully generate database {} len {}'.format(
            neuromeka_cls_name, len(database)
        )
    )
    save_pickle(
        database, os.path.join(cache_dir,'{}_info.pkl').format(neuromeka_cls_name)
    )
    return database


def randomly_read_background(background_dir,cache_dir):
    if os.path.exists(os.path.join(cache_dir,'background_info.pkl')):
        fns = read_pickle(os.path.join(cache_dir,'background_info.pkl'))
    else:
        fns = glob(os.path.join(background_dir,'*.jpg')) + \
            glob(os.path.join(background_dir,'*.png'))
        save_pickle(fns, os.path.join(cache_dir,'background_info.pkl'))

    return cv2.imread(fns[np.random.randint(0,len(fns))])[:, :, ::-1]


def fuse_regions(rgbs, masks, depths, begins, cls_ids, background, th, tw, cls):
    fuse_order = np.arange(len(rgbs))
    np.random.shuffle(fuse_order)
    fuse_img = background
    fuse_img = cv2.resize(fuse_img,(tw,th),interpolation=cv2.INTER_LINEAR)
    fuse_mask = np.zeros([fuse_img.shape[0],fuse_img.shape[1]],np.int32)
    INF = pow(2,16)
    fuse_depth = np.ones([fuse_img.shape[0], fuse_img.shape[1]], np.uint32) * INF
    t_cls_id = neuromeka_obj_dict[cls]
    if len(background.shape) < 3:
        return None, None, None, None
    for idx in fuse_order:
        if len(rgbs[idx].shape) < 3:
            continue
        cls_id = cls_ids[idx]
        rh,rw = masks[idx].shape
        if cls_id == t_cls_id:
            bh, bw = begins[idx][0], begins[idx][1]
        else:
            bh = np.random.randint(0,fuse_img.shape[0]-rh)
            bw = np.random.randint(0,fuse_img.shape[1]-rw)

        silhouette = masks[idx]>0
        out_silhouette = np.logical_not(silhouette)
        fuse_depth_patch = fuse_depth[bh:bh+rh, bw:bw+rw].copy()
        cover = (depths[idx] < fuse_depth_patch) * silhouette
        not_cover = np.logical_not(cover)

        fuse_mask[bh:bh+rh,bw:bw+rw] *= not_cover.astype(fuse_mask.dtype)
        cover_msk = masks[idx] * cover.astype(masks[idx].dtype)
        fuse_mask[bh:bh+rh,bw:bw+rw] += cover_msk

        fuse_img[bh:bh+rh,bw:bw+rw] *= not_cover.astype(fuse_img.dtype)[:,:,None]
        cover_rgb = rgbs[idx] * cover.astype(rgbs[idx].dtype)[:,:,None]
        fuse_img[bh:bh+rh,bw:bw+rw] += cover_rgb

        fuse_depth[bh:bh+rh, bw:bw+rw] *= not_cover.astype(fuse_depth.dtype)
        cover_dpt = depths[idx] * cover.astype(depths[idx].dtype)
        fuse_depth[bh:bh+rh, bw:bw+rw] += cover_dpt.astype(fuse_depth.dtype)

        begins[idx][0] = -begins[idx][0]+bh
        begins[idx][1] = -begins[idx][1]+bw

    dp_bg = (fuse_depth == INF)
    dp_bg_filter = np.logical_not(dp_bg)
    fuse_depth *= dp_bg_filter.astype(fuse_depth.dtype)

    return fuse_img, fuse_mask, fuse_depth, begins


def randomly_sample_foreground(image_db, neuromeka_dir):
    idx = np.random.randint(0,len(image_db))
    rgb_pth = image_db[idx]['rgb_pth']
    dpt_pth = image_db[idx]['dpt_pth']
    msk_pth = image_db[idx]['msk_pth']
    with Image.open(dpt_pth) as di:
        dpt_buff = np.array(di)
        # depth = np.array(di).astype(np.int16)
    depth = combine_bytes_arr(dpt_buff[:, :, 1], dpt_buff[:, :, 2])
    with Image.open(msk_pth) as li:
        mask = np.array(li).astype(np.int16)
    with Image.open(rgb_pth) as ri:
        rgb = np.array(ri)[:, :, :3].astype(np.uint8)

    # mask = np.sum(mask,2)>0
    mask = mask > 0
    mask = np.asarray(mask,np.int32)

    hs, ws = np.nonzero(mask)
    hmin, hmax = np.min(hs),np.max(hs)
    wmin, wmax = np.min(ws),np.max(ws)

    mask = mask[hmin:hmax,wmin:wmax]
    rgb = rgb[hmin:hmax,wmin:wmax]
    depth = depth[hmin:hmax, wmin:wmax]

    rgb *= mask.astype(np.uint8)[:,:,None]
    depth *= mask.astype(np.uint16)[:,:]
    begin = [hmin,wmin]
    pose = image_db[idx]['RT']

    return rgb, mask, depth, begin, pose


def save_fuse_data(
    output_dir, idx, fuse_img, fuse_mask, fuse_depth, fuse_begins, t_pose, cls
):
    cls_id = neuromeka_obj_dict[cls]
    if (fuse_mask == cls_id).sum() < 20:
        return None
    os.makedirs(output_dir, exist_ok=True)
    DEPTH_SCALE_HQ = 50000.0
    fuse_mask = fuse_mask.astype(np.uint8)
    data = {}
    data['rgb'] = fuse_img
    data['mask'] = fuse_mask
    data['depth'] = fuse_depth.astype(np.float32) # / DEPTH_SCALE_HQ
    data['K'] = Intrinsic_matrix['linemod']
    data['RT'] = t_pose
    data['cls_typ'] = cls
    data['rnd_typ'] = 'fuse'
    data['begins'] = fuse_begins
    if DEBUG:
        imshow("rgb", fuse_img[:, :, ::-1])
        imshow("depth", (fuse_depth / fuse_depth.max() * 255).astype('uint8'))
        imshow("label", (fuse_mask / fuse_mask.max() * 255).astype("uint8"))
        waitKey(0)
    sv_pth = os.path.join(output_dir, "{}.pkl".format(idx))
    pickle.dump(data, open(sv_pth, 'wb'))
    sv_pth = os.path.abspath(sv_pth)
    return sv_pth


def prepare_dataset_single(
    output_dir, idx, neuromeka_dir, background_dir, cache_dir, seed, cls
):
    time_begin = time.time()
    np.random.seed(seed)
    rgbs, masks, depths, begins, poses, cls_ids = [], [], [], [], [], []
    image_dbs={}
    for cls_name in neuromeka_obj_dict.keys():
        cls_id = neuromeka_obj_dict[cls_name]
        image_dbs[cls_id] = collect_neuromeka_set_info(
            neuromeka_dir, cls_name, cache_dir
        )

    for cls_name in neuromeka_obj_dict.keys():
        cls_id = neuromeka_obj_dict[cls_name]
        rgb, mask, depth, begin, pose = randomly_sample_foreground(
            image_dbs[cls_id], neuromeka_dir
        )
        if cls_name == cls:
            t_pose = pose
        mask *= cls_id
        rgbs.append(rgb)
        masks.append(mask)
        depths.append(depth)
        begins.append(begin)
        poses.append(pose)
        cls_ids.append(cls_id)

    background = randomly_read_background(background_dir, cache_dir)

    fuse_img, fuse_mask, fuse_depth, fuse_begins= fuse_regions(
        rgbs, masks, depths, begins, cls_ids, background, 480, 640, cls
    )

    if fuse_img is not None:
        sv_pth = save_fuse_data(
            output_dir, idx, fuse_img, fuse_mask, fuse_depth, fuse_begins,
            t_pose, cls
        )
        return sv_pth


def prepare_dataset_parallel(
        output_dir, neuromeka_dir, fuse_num, background_dir, cache_dir,
        worker_num=16, cls="bottle"
):
    exector = ProcessPoolExecutor(max_workers=worker_num)
    futures = []

    for cls_name in neuromeka_obj_dict.keys():
        collect_neuromeka_set_info(neuromeka_dir, cls_name, cache_dir)
    randomly_read_background(background_dir, cache_dir)

    for idx in np.arange(fuse_num):
        seed = np.random.randint(500000)
        futures.append(exector.submit(
            prepare_dataset_single, output_dir, idx, neuromeka_dir,
            background_dir, cache_dir, seed, cls
        ))

    pth_lst = []
    for f in tqdm(futures):
        res = f.result()
        if res is not None:
            pth_lst.append(res)
    f_lst_pth = os.path.join(output_dir, "file_list.txt")
    with open(f_lst_pth, "w") as f:
        for item in pth_lst:
            print(item, file=f)


if __name__=="__main__":
    for cls in neuromeka_obj_dict.keys():
        # cls = args.cls
        neuromeka_dir = '../ffb6d/datasets/neuromeka'
        output_dir = os.path.join(neuromeka_dir, "fuse",  cls)
        ensure_dir(output_dir)
        background_dir = '../ffb6d/datasets/SUN2012pascalformat/JPEGImages'
        cache_dir = '../ffb6d/datasets/neuromeka/cache'
        fuse_num = args.fuse_num
        worker_num = 16
        prepare_dataset_parallel(
            output_dir, neuromeka_dir, fuse_num, background_dir, cache_dir,
            worker_num, cls=cls
        )
