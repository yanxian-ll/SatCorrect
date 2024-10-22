# This is a simple baseline semantic MVS algorithm using IC-Net and DenseMapNet
# to check that the US3D metadata is correct and to demonstrate
# epipolar rectification, RPC projections, UTM conversions, and triangulation.
# This example also shows how to read image metadata from the IMD files to
# help guide image pair selection.


# Code for estimating the Fundamental matrix was adapted from the following OpenCV python tutorial:
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html


# Code for SGBM stereo matching was adapted from Timotheos Samartzidis's blog:
# http://timosam.com/python_opencv_depthimage
# which elaborates on the following OpenCV tutorial:
# https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/samples/disparity_filtering.cpp
# The weighted least squares filtering method used is based on the following paper:
# Min, Dongbo and Choi, Sunghwan and Lu, Jiangbo and Ham, Bumsub and Sohn, Kwanghoon and Do, Minh N,
# "Fast global image smoothing based on weighted least squares," IEEE Transactions on Image Processing, 2014.


# Epipolar rectification code was adapted from the following demo by Julien Rebetez:
# https://github.com/julienr/cvscripts/blob/master/rectification/rectification_demo.py
# Copyright (2012) Julien Rebetez <julien@fhtagn.net>. All rights reserved.
# Redistribution and use in source and binary forms, with or without modification, 
# are permitted provided that the following conditions are met:
#
#   1. Redistributions of source code must retain the above copyright notice, this list 
#      of conditions and the following disclaimer.
#   2. Redistributions in binary form must reproduce the above copyright notice, this list of 
#      conditions and the following disclaimer in the documentation and/or other materials 
#      provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE FREEBSD PROJECT ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, 
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE FREEBSD PROJECT OR CONTRIBUTORS BE LIABLE 
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT 
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN 
# IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import tifffile
import cv2
import numpy as np
from osgeo import gdal
import os
import math
from utm import *
from PIL import Image
import json
from plyfile import PlyData, PlyElement
import multiprocessing
from scipy.interpolate import RegularGridInterpolator

from flow_utils.raft_main import setup_raft
from flow_utils.raft_main import compute_optical_flow


# read IMD file with sensor azimuth and elevation values
def read_imd(name):
    file = open(name, 'r')
    lines = file.readlines()
    for j in range(len(lines)):
        pos = lines[j].find('meanSatAz')
        if pos != -1:
            last = lines[j].find(';') - 1
            az = float(lines[j][pos + 11:last])
        pos = lines[j].find('meanSatEl')
        if pos != -1:
            last = lines[j].find(';') - 1
            el = float(lines[j][pos + 11:last])
    return az, el


# compute convergence angle given two sets of azimuths and elevations
def convergence_angle(az1, az2, el1, el2):
    cosd = math.sin(el1) * math.sin(el2) + math.cos(el1) * math.cos(el2) * math.cos(az1 - az2)
    d = math.degrees(math.acos(cosd))
    return d


# get NITF metadata that we embedded in the GeoTIFF header
def get_image_metadata(img_name):
    dataset = gdal.Open(img_name, gdal.GA_ReadOnly)
    metadata = dataset.GetMetadata()
    rpc_data = dataset.GetMetadata('RPC')
    date_time = metadata['NITF_IDATIM']
    year = int(date_time[0:4])
    month = int(date_time[4:6])
    day = int(date_time[6:8])
    return metadata, month, day


def _triangulatePoint(x1, y1, x2, y2, P1, P2):
    A = np.array([
            [x1 * P1[2, 0] - P1[0, 0], x1 * P1[2, 1] - P1[0, 1], x1 * P1[2, 2] - P1[0, 2], x1*P1[2,3]-P1[0,3]],
            [y1 * P1[2, 0] - P1[1, 0], y1 * P1[2, 1] - P1[1, 1], y1 * P1[2, 2] - P1[1, 2], y1*P1[2,3]-P1[1,3]],
            [x2 * P2[2, 0] - P2[0, 0], x2 * P2[2, 1] - P2[0, 1], x2 * P2[2, 2] - P2[0, 2], x2*P2[2,3]-P2[0,3]],
            [y2 * P2[2, 0] - P2[1, 0], y2 * P2[2, 1] - P2[1, 1], y2 * P2[2, 2] - P2[1, 2], y2*P2[2,3]-P2[1,3]]
        ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3]/X[-1]
    

def optical_flow_to_xyz(img1_path, img2_path, cam1_path, cam2_path, model, device='cpu', error_threshold=1.0):
    # read images
    img1 = np.array(Image.open(img1_path)).astype(np.uint8)
    img2 = np.array(Image.open(img2_path)).astype(np.uint8)
    colors = img1.reshape((img1.shape[0]*img1.shape[1], 3))

    # read cameras
    cam1 = json.load(open(cam1_path))
    cam2 = json.load(open(cam2_path))
    W2C1, W2C2 = np.array(cam1['W2C']).reshape((4,4)), np.array(cam2['W2C']).reshape((4,4))
    K1, K2 = np.array(cam1['K']).reshape((4,4)), np.array(cam2['K']).reshape((4,4))
    P1 = K1[:3, :3] @ W2C1[:3,:4]
    P2 = K2[:3,:3] @ W2C2[:3,:4]

    # compute flow
    forward_flow, backward_flow = compute_optical_flow(img1, img2, model, device)

    # TODO:forward-backward consistency check


    # flow to xyz
    l_rows, l_cols = forward_flow.shape[:2]
    r_rows, r_cols = backward_flow.shape[:2]
    left_rows, left_cols = np.mgrid[0:l_rows, 0:l_cols]
    right_rows = left_rows + forward_flow[:,:,1]
    right_cols = left_cols + forward_flow[:,:,0]

    left_rows = left_rows.ravel()
    left_cols = left_cols.ravel()
    right_rows = right_rows.ravel()
    right_cols = right_cols.ravel()

    mask = (right_rows>=0) & (right_rows<r_rows) & (right_cols>=0) & (right_cols<r_cols)
    left_rows = left_rows[mask]
    left_cols = left_cols[mask]
    right_rows = right_rows[mask]
    right_cols = right_cols[mask]
    colors = colors[mask]

    u_inter = RegularGridInterpolator(points=(np.linspace(0,r_rows-1, r_rows), np.linspace(0,r_cols-1, r_cols)), 
                                      values=backward_flow[:,:,0], bounds_error=False)
    v_inter = RegularGridInterpolator(points=(np.linspace(0,r_rows-1, r_rows), np.linspace(0,r_cols-1, r_cols)),
                                      values=backward_flow[:,:,1], bounds_error=False)
    u = u_inter(np.vstack([right_rows, right_cols]).T) + right_cols
    v = v_inter(np.vstack([right_rows, right_cols]).T) + right_rows

    # ## visualization test
    # import matplotlib.pyplot as plt
    # if img1.shape[0] < img2.shape[0]:
    #     img1 = np.pad(img1, ((0, img2.shape[0]-img1.shape[0]), (0, 0), (0, 0)))
    # if img1.shape[1] < img2.shape[1]:
    #     img1 = np.pad(img1, ((0, 0), (0, img2.shape[1]-img1.shape[1]), (0, 0)))
    # img2 = np.pad(img2, ((0, img1.shape[0]-img2.shape[0]), (0, img1.shape[1]-img2.shape[1]), (0, 0)))
    # img = np.concatenate((img1, img2), axis=1) / 255.0
    # W = img1.shape[1]

    # plt.imshow(img)
    # cmap = plt.get_cmap('jet')
    # num_matches = 30
    # idx = np.random.randint(0, len(left_rows), num_matches)
    # for ii, i in enumerate(idx):
    #     x0 = u[i]
    #     y0 = v[i]
    #     x1 = right_cols[i]
    #     y1 = right_rows[i]
    #     plt.plot([x0, x1 + W], [y0, y1], '-+', color=cmap(ii / (num_matches - 1)), scalex=False, scaley=False)
    # plt.show()
    
    error = np.sqrt(np.square(u - left_cols) + np.square(v - left_rows))
    mask = (error<error_threshold)
    print(f"num valid points: {mask.sum()}")

    points1 = np.vstack((left_cols[mask], left_rows[mask]))
    points2 = np.vstack((right_cols[mask], right_rows[mask]))
    ## TODO:opencv crushed
    # xyz = cv2.triangulatePoints(P1.copy(), P2.copy(), points1.copy(), points2.copy())
    # xyz /= xyz[3]
    # xyz = np.transpose(xyz)[:, :3] # (N,3)

    # pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    # results = [pool.apply_async(_triangulatePoint, args=(x1,y1,x2,y2,P1,P2)) for x1,y1,x2,y2 in zip(points1[0], points1[1], points2[0], points2[1])]
    # points_3d = [result.get() for result in results]
    # xyz = np.vstack(points_3d)

    points_3d = []
    for x1,y1,x2,y2 in zip(points1[0], points1[1], points2[0], points2[1]):
        points_3d.append(_triangulatePoint(x1,y1,x2,y2,P1,P2))
    xyz = np.vstack(points_3d)

    return xyz, colors[mask]


# main program to demonstrate a baseline MVS algorithm
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="data/JAX_214/rot_correct/images")
    parser.add_argument('--camera_path', type=str, default="data/JAX_214/rot_correct/cameras")
    parser.add_argument('--meta_path', type=str, default="data/JAX_214/preprocess/metas")
    parser.add_argument('--max_pairs', type=int, default=1)
    # for raft
    parser.add_argument('--raft_ckpt_model', type=str, default="flow_utils/checkpoints/raft-things.pth")
    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--threshold', type=float, default=1.0)
    parser.add_argument('--output_path', type=str, default="./test_output")

    args = parser.parse_args()
    
    image_files = [os.path.join(args.image_path, f) for f in sorted(os.listdir(args.image_path))]
    camera_files = [os.path.join(args.camera_path, f) for f in sorted(os.listdir(args.camera_path))]
    meta_files = [os.path.join(args.meta_path, f) for f in sorted(os.listdir(args.meta_path))]
    assert len(image_files) == len(camera_files) == len(meta_files)

    # load optical flow model
    model = setup_raft(args.raft_ckpt_model, device=args.device)

    # get azimuth and elevation angles for all images
    # also get semantic segmentation outputs for all images.
    azimuths = []
    elevations = []
    months = []
    for i in range(len(image_files)):
        # # get image index
        # pos = image_files[i].find('_RGB')
        # ndx = int(image_files[i][pos - 3:pos])
        # # get IMD metadata for this image
        # imd_name = os.path.join(args.meta_path, '{:02d}.IMD'.format(ndx))
        # az, el = read_imd(imd_name)
        # azimuths.append(az)
        # elevations.append(el)
        # meta, month, day = get_image_metadata(image_files[i])

        month = json.load(open(meta_files[i]))['capture_date'][1]
        months.append(month)

        # get list of pairs to process
        pairs = []
        convergence = []
        distances = []
    
    for i in range(len(image_files) - 1):
        for j in range(i + 1, len(image_files)):
            # pair = [image_files[i], image_files[j]]
            pair = [i, j]
            pairs.append(pair)
            # # get convergence angle
            # d = convergence_angle(azimuths[i], azimuths[j], elevations[i], elevations[j])
            # convergence.append(d)
            # get time distance in months
            dist = abs(months[i] - months[j])
            dist = min(dist, 12 - dist)
            distances.append(dist)

    distances = np.asarray(distances)
    indices = distances.argsort()
    pairs = [pairs[i] for i in indices if i < args.max_pairs]
    npairs = len(pairs)
    print('Number of pairs = ', npairs)

    # process all pairs and save XYZ files in output folder
    for i in range(npairs):
         # run a single optical flow
        xyzs, colors = optical_flow_to_xyz(image_files[pairs[i][0]], image_files[pairs[i][1]], 
                                          camera_files[pairs[i][0]], camera_files[pairs[i][1]], 
                                          model=model, device=args.device)
        
        dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, colors), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(os.path.join(args.output_path, f"{image_files[pairs[i][0]].split('/')[-1].split('.')[0]}.ply"))
