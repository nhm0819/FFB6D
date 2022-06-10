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


def rescale_depthmap(depthmap):
    zNear = float(0.1)
    zFar = float(100.0)
    return ((zNear * zFar) / (zNear - zFar) / depthmap) + (zFar / (zFar - zNear))


def scaleback_depthmap(depthmap_res):
    zNear = float(0.1)
    zFar = float(100.0)
    return (zNear * zFar) / (zNear - zFar) / (depthmap_res - zFar / (zFar - zNear))


def read_view(filepath):
    with open(filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        view = {}
        for row in csv_reader:
            view['focus'] = np.array([float(row[0]), float(row[1]), float(row[2])])
            view['dist'] = float(row[3])
            view['euler'] = np.array([float(row[4]), float(row[5]), float(row[6])])
            view['lens_f'] = np.array([float(row[7]), float(row[8])])
            view['sensor_dim'] = np.array([float(row[9]), float(row[10])])
            view['offset_px'] = np.array([float(row[11]), float(row[12])])
            view['resolution'] = np.array([float(row[13]), float(row[14])])
            view['lens_f_px'] = view['lens_f'] / view['sensor_dim'] * view['resolution']
    return view


def read_objects(filepath):
    with open(filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        objects = []
        for row in csv_reader:
            obj = {}
            obj['cls'] = int(row[0])
            obj['scale'] = np.array([float(row[1]), float(row[2]), float(row[3])])
            obj['pos'] = np.array([float(row[4]), float(row[5]), float(row[6])])
            obj['eu_zyx_rev'] = np.array([float(row[7]), float(row[8]), float(row[9])])
            obj['eu_zyx'] = np.array([float(row[9]), float(row[8]), float(row[7])])
            objects += [obj]
    return objects


def calc_Tbo(obj):
    Tbo = np.identity(4)
    Tbo[0:3, 0:3] = Rot_zyx(*np.deg2rad(obj['eu_zyx']))
    Tbo[0:3, 3] = obj['pos']
    return Tbo


def calc_Tbc(view):
    Tbf = np.identity(4)
    Tbf[0:3, 3] = view['focus']
    Tff = np.identity(4)
    Tff[0:3, 0:3] = Rot_zxz(*np.deg2rad(view['euler']))
    Tfc = np.identity(4)
    Tfc[2, 3] = view['dist']
    Tbc = np.matmul(np.matmul(Tbf, Tff), Tfc)
    return Tbc


def calc_Tco(obj, view):

    Tx180 = np.identity(4, 'float32')
    Tx180[1, 1] = -1
    Tx180[2, 2] = -1

    Tbo = calc_Tbo(obj)
    Tbc = calc_Tbc(view)
    Tco = np.matmul(np.linalg.inv(Tbc), Tbo)
    if Tco[2, 3] > 0:
        Tco = np.matmul(Tx180, Tco)
    return Tco


def Rot_axis(axis, q):
    '''
    make rotation matrix along axis
    '''
    if axis == 1:
        R = np.asarray([[1, 0, 0], [0, cos(q), -sin(q)], [0, sin(q), cos(q)]])
    if axis == 2:
        R = np.asarray([[cos(q), 0, sin(q)], [0, 1, 0], [-sin(q), 0, cos(q)]])
    if axis == 3:
        R = np.asarray([[cos(q), -sin(q), 0], [sin(q), cos(q), 0], [0, 0, 1]])
    return R


def Rot_zyx(zr, yr, xr):
    '''
    zyx rotatio matrix - caution: axis order: z,y,x
    '''
    R = np.matmul(np.matmul(Rot_axis(3, zr), Rot_axis(2, yr)), Rot_axis(1, xr))
    return R


def Rot_zxz(zr1, xr2, zr3):
    '''
    zxz rotatio matrix - caution: axis order: z,x,z
    '''
    R = np.matmul(np.matmul(Rot_axis(3, zr1), Rot_axis(1, xr2)), Rot_axis(3, zr3))
    return R


def divide_bytes_arr(uint_arr):
    num = uint_arr.astype(np.uint)
    return np.floor(num % 0x10000 / 0x100).astype(np.uint) % 0x100, num % 0x100


def combine_bytes_arr(up_byte_arr, lo_byte_arr):
    return up_byte_arr * 0x100 + lo_byte_arr


def get_axis_points(Tco, cam_k, axis_length):
    vec0 = np.array([[0], [0], [0], [1]])
    vecx = np.array([[axis_length], [0], [0], [1]])
    vecy = np.array([[0], [axis_length], [0], [1]])
    vecz = np.array([[0], [0], [axis_length], [1]])

    Pc0 = np.matmul(Tco, vec0)[0:3, 0:1]
    Pcx = np.matmul(Tco, vecx)[0:3, 0:1]
    Pcy = np.matmul(Tco, vecy)[0:3, 0:1]
    Pcz = np.matmul(Tco, vecz)[0:3, 0:1]

    Pxn0 = np.matmul(cam_k, Pc0).flatten()
    Pxnx = np.matmul(cam_k, Pcx).flatten()
    Pxny = np.matmul(cam_k, Pcy).flatten()
    Pxnz = np.matmul(cam_k, Pcz).flatten()

    px0 = (Pxn0[0:2] / Pxn0[2]).astype('int64')
    pxx = (Pxnx[0:2] / Pxnx[2]).astype('int64')
    pxy = (Pxny[0:2] / Pxny[2]).astype('int64')
    pxz = (Pxnz[0:2] / Pxnz[2]).astype('int64')
    return px0, pxx, pxy, pxz


def draw_axis(img, Tco, cam_k, line_ratio=1 / 100, axis_length=0.05):
    'temporarily implemented as axis draw'
    imgout = img.copy()
    maxval = np.max(img).item()
    minval = np.min(img).item()

    px0, pxx, pxy, pxz = get_axis_points(Tco, cam_k, axis_length)

    pxt = px0.copy()
    pxt[0] += 10
    pxt[1] -= 10
    pxt = tuple(pxt.tolist())
    px0 = tuple(px0.tolist())
    pxx = tuple(pxx.tolist())
    pxy = tuple(pxy.tolist())
    pxz = tuple(pxz.tolist())

    thickness = max(1, round(imgout.shape[1] * line_ratio))
    cv2.line(imgout, px0, pxx, (maxval, minval, minval), thickness)
    cv2.line(imgout, px0, pxy, (minval, maxval, minval), thickness)
    cv2.line(imgout, px0, pxz, (minval, minval, maxval), thickness)

    return imgout