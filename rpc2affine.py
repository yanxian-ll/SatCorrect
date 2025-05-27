"""
RPC to Affine Camera Model Conversion

This module converts Rational Polynomial Camera (RPC) models to affine camera models
for satellite imagery. It handles:
- RPC parameter extraction from metadata
- Coordinate system transformations (lat/lon to UTM)
- Affine camera model estimation
- Sun position and view direction calculations

Key functions:
- _process_single_image: Main processing pipeline for single image
- _generate_samples: Generate sample points for projection matrix estimation
- _solve_projection_matrix: Solve camera projection matrix from correspondences
- _factorize_projection_matrix: Factorize projection matrix into K, R, t
"""

import argparse
import os
import tifffile
import multiprocessing
import numpy as np
import pyproj
from osgeo import gdal
import json
import imageio
import cv2
from scipy import linalg

from preprocess.rpc_model import RPCModel


def eastnorth_to_latlon(east, north, zone_number, hemisphere):
    if hemisphere == 'N':
        south = False
    else:
        south = True
    proj = pyproj.Proj(proj='utm', ellps='WGS84', zone=zone_number, south=south)
    lon, lat = proj(east, north, inverse=True)
    return lat, lon


def latlon_to_utm(lat, lon, zone_number=None):
    if zone_number is None:
        zone_number = int((lon.min() + 180) / 6) + 1
    utm_proj = pyproj.Proj(proj='utm', zone=zone_number, ellps='WGS84')
    easting, northing = utm_proj(lon, lat)
    return easting, northing


def read_tif_and_info(tiff_fpath, rpc_file):
    """
    Read TIFF image and extract metadata including RPC parameters.

    Args:
        tiff_fpath (str): Path to input TIFF file
        rpc_file (str): Path to JSON file containing RPC parameters

    Returns:
        tuple: (image_data, meta_dict) where:
            image_data: Numpy array of image data (H,W,3)
            meta_dict: Dictionary containing:
                - rpc: RPC parameters
                - height/width: Image dimensions
                - capture_date: Image capture timestamp
                - sun_elevation/azimuth: Sun position angles
    """
    dataset = gdal.Open(tiff_fpath, gdal.GA_ReadOnly)
    img = dataset.ReadAsArray()
    assert (len(img.shape) == 3 and img.shape[0] == 3)
    img = img.transpose((1, 2, 0))   # [c, h, w] --> [h, w, c]
    assert (img.dtype == np.uint8)

    metadata = dataset.GetMetadata()
    date_time = metadata['NITF_IDATIM']
    year = int(date_time[0:4])
    month = int(date_time[4:6])
    day = int(date_time[6:8])
    hour = int(date_time[8:10])
    minute = int(date_time[10:12])
    second = int(date_time[12:14])
    capture_date = [year, month, day, hour, minute, second]

    with open(rpc_file, 'r') as file:
        data = json.load(file)
        rpc_data = data['rpc']
        sun_elevation = float(data['sun_elevation'])
        sun_azimuth = float(data['sun_azimuth'])

    rpc_dict = {
        'lonOff': float(rpc_data['lon_offset']),
        'lonScale': float(rpc_data['lon_scale']),
        'latOff': float(rpc_data['lat_offset']),
        'latScale': float(rpc_data['lat_scale']),
        'altOff': float(rpc_data['alt_offset']),
        'altScale': float(rpc_data['alt_scale']),
        'rowOff': float(rpc_data['row_offset']),
        'rowScale': float(rpc_data['row_scale']),
        'colOff': float(rpc_data['col_offset']),
        'colScale': float(rpc_data['col_scale']),
        'rowNum': rpc_data['row_num'],
        'rowDen': rpc_data['row_den'],
        'colNum': rpc_data['col_num'],
        'colDen': rpc_data['col_den'],
    }

    # rpc_data = dataset.GetMetadata('RPC')
    # rpc_dict = {
    #     'lonOff': float(rpc_data['LONG_OFF']),
    #     'lonScale': float(rpc_data['LONG_SCALE']),
    #     'latOff': float(rpc_data['LAT_OFF']),
    #     'latScale': float(rpc_data['LAT_SCALE']),
    #     'altOff': float(rpc_data['HEIGHT_OFF']),
    #     'altScale': float(rpc_data['HEIGHT_SCALE']),
    #     'rowOff': float(rpc_data['LINE_OFF']),
    #     'rowScale': float(rpc_data['LINE_SCALE']),
    #     'colOff': float(rpc_data['SAMP_OFF']),
    #     'colScale': float(rpc_data['SAMP_SCALE']),
    #     'rowNum': np.asarray(rpc_data['LINE_NUM_COEFF'].split(), dtype=np.float64).tolist(),
    #     'rowDen': np.asarray(rpc_data['LINE_DEN_COEFF'].split(), dtype=np.float64).tolist(),
    #     'colNum': np.asarray(rpc_data['SAMP_NUM_COEFF'].split(), dtype=np.float64).tolist(),
    #     'colDen': np.asarray(rpc_data['SAMP_DEN_COEFF'].split(), dtype=np.float64).tolist()
    # }

    meta_dict = { 'rpc': rpc_dict,
                  'height': img.shape[0],
                  'width': img.shape[1], 
                  'capture_date': capture_date,
                  'sun_elevation': sun_elevation,
                  'sun_azimuth': sun_azimuth
    }
    return img, meta_dict


def _generate_samples(meta_dict, lat_minmax, lon_minmax, alt_minmax, 
                                 lat_N, lon_N, alt_N):
    '''
        meta_dict: see parse_tiff.py for information about meta_dict
        lat_minmax, lon_minmax, alt_minmax: (2,)
        lat_N, lon_N, alt_N: integers
    '''
    rpc_model = RPCModel(meta_dict)

    lon, lat, alt = np.meshgrid(np.linspace(lon_minmax[0], lon_minmax[1], lon_N),
                                np.linspace(lat_minmax[0], lat_minmax[1], lat_N),
                                np.linspace(alt_minmax[0], alt_minmax[1], alt_N))
    lon, lat, alt = lon.reshape((-1,)), lat.reshape((-1,)), alt.reshape((-1,))

    col, row = rpc_model.projection(lat, lon, alt)
    keep_mask = np.logical_and(col >= 0, row >= 0)
    keep_mask = np.logical_and(keep_mask, col < rpc_model.width)
    keep_mask = np.logical_and(keep_mask, row < rpc_model.height)
    return lat[keep_mask], lon[keep_mask], alt[keep_mask], col[keep_mask], row[keep_mask]


def _solve_projection_matrix(x, y, z, col, row, enable_debug=False):
    '''
        x, y, z, col, row: [N, ]; numpy array

        return:
            P: 3x4 projection matrix
    '''
    x, y, z, col, row = x.reshape((-1, 1)), y.reshape((-1, 1)), z.reshape((-1, 1)),\
                             col.reshape((-1, 1)), row.reshape((-1, 1))
    point_cnt = x.shape[0]
    all_ones = np.ones((point_cnt, 1))
    all_zeros = np.zeros((point_cnt, 4))

    A1 = np.hstack((x, y, z, all_ones,
                    all_zeros,
                    -col * x, -col * y, -col * z, -col * all_ones))
    A2 = np.hstack((all_zeros,
                    x, y, z, all_ones,
                    -row * x, -row * y, -row * z, -row * all_ones))
    A = np.vstack((A1, A2))
    u, s, vh = linalg.svd(A, full_matrices=False)
    P = np.real(vh[11, :]).reshape((3, 4))

    # if enable_debug:
    #     tmp = np.matmul(np.hstack((x, y, z, all_ones)), P.T)
    #     approx_col = tmp[:, 0:1] / tmp[:, 2:3]
    #     approx_row = tmp[:, 1:2] / tmp[:, 2:3]
    #     pixel_err = np.sqrt((approx_row - row) ** 2 + (approx_col - col) ** 2)
    #     ic('# points: {}, approx. error (pixels): {}'.format(point_cnt, np.median(pixel_err)))
    return P


def _factorize_projection_matrix(P):
    '''
        factorize a 3x4 projection matrix P to K, R, t
        P: [3, 4]; numpy array
        
        return:
            K: [3, 3]; numpy array
            R: [3, 3]; numpy array
            t: [3, ]; numpy array
    '''
    K, R = linalg.rq(P[:, :3])
    t = linalg.lstsq(K, P[:, 3:4])[0]

    # fix the intrinsic and rotation matrix
    #   intrinsic matrix's diagonal entries must be all positive
    #   rotation matrix's determinant must be 1; otherwise there's an reflection component

    neg_sign_cnt = int(K[0, 0] < 0) + int(K[1, 1] < 0) + int(K[2, 2] < 0)
    if neg_sign_cnt == 1 or neg_sign_cnt == 3:
        K = -K

    new_neg_sign_cnt = int(K[0, 0] < 0) + int(K[1, 1] < 0) + int(K[2, 2] < 0)
    assert (new_neg_sign_cnt == 0 or new_neg_sign_cnt == 2)

    fix = np.diag((1, 1, 1))
    if K[0, 0] < 0 and K[1, 1] < 0:
        fix = np.diag((-1, -1, 1))
    elif K[0, 0] < 0 and K[2, 2] < 0:
        fix = np.diag((-1, 1, -1))
    elif K[1, 1] < 0 and K[2, 2] < 0:
        fix = np.diag((1, -1, -1))
    K = np.matmul(K, fix)
    R = np.matmul(fix, R)
    t = np.matmul(fix, t).reshape((-1,))

    assert (linalg.det(R) > 0)
    K /= K[2, 2]
    return K, R, t


def sun_angles_to_enu(azimuth, elevation):
    azimuth_rad = np.radians(azimuth)
    elevation_rad = np.radians(elevation)
    E = np.cos(elevation_rad) * np.sin(azimuth_rad)
    N = np.cos(elevation_rad) * np.cos(azimuth_rad)
    U = np.sin(elevation_rad)
    dir = np.array([E, N, U])
    return dir / np.linalg.norm(dir)


def pix2ndc(pix, s):
    return (pix * 2.0 + 1.0) / s - 1.0

def ndc2pix(ndc, s):
    return ((ndc + 1.0) * s - 1.0) / 2.0


def sunview_2_w2c(sun_view, sun_position):
    z_axis = -sun_view
    if np.isclose(np.abs(z_axis[1]), 1.0):
        up = np.array([1, 0, 0])
    else:
        up = np.array([0, 1, 0])

    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    R = np.stack((x_axis, y_axis, z_axis), axis=1)
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, -1] = sun_position
    w2c = np.linalg.inv(c2w)
    return w2c


def _process_single_image(image_file, rpc_file, output_path, lat_minmax, lon_minmax, alt_minmax, translation, scales, 
    world_points=None  # for test
):
    """
    Process a single satellite image to convert RPC model to affine camera model.

    Args:
        image_file (str): Path to input image file
        rpc_file (str): Path to RPC parameters file
        output_path (str): Directory to save processed outputs
        lat_minmax (list): [min, max] latitude bounds
        lon_minmax (list): [min, max] longitude bounds  
        alt_minmax (list): [min, max] altitude bounds
        translation (np.array): ENU coordinate system origin
        scales (np.array): Scaling factors for normalization
        world_points (np.array, optional): Ground truth points for validation

    Outputs:
        - Processed image in output_path/images/
        - Camera parameters JSON in output_path/cameras/
    """
    img, meta_dict = read_tif_and_info(image_file, rpc_file)

    image_height, image_width = img.shape[:2]

    lat, lon, alt, col, row = _generate_samples(meta_dict, lat_minmax, lon_minmax, alt_minmax, lat_N=100, lon_N=100, alt_N=100)

    utm_e, utm_n = latlon_to_utm(lat, lon)
    sample_utm = np.vstack([utm_e, utm_n, alt]).T
    sample_world = (sample_utm - translation) / scales

    uvs_sample = np.vstack([col, row]).T
    ndc_sample = (uvs_sample * 2.0 + 1.0) / np.array([[image_width, image_height]]) - 1.0
    points_2d = np.concatenate([ndc_sample, np.zeros((ndc_sample.shape[0],1))], axis=1)
    _, affine_matrix, _ = cv2.estimateAffine3D(sample_world, points_2d)  #(3,4)
    full_affine = np.zeros((4,4))
    full_affine[:2, :4] = affine_matrix[:2, :4]
    full_affine[2, 2] = scales[0, -1]
    full_affine[2, 3] = translation[0, -1]
    full_affine[3, 3] = 1.0

    P = _solve_projection_matrix(sample_world[:,0], sample_world[:,1], sample_world[:,2], col, row)
    K, R, t = _factorize_projection_matrix(P)
    W2C = np.eye(4)
    W2C[:3, :3] = R
    W2C[:3, 3] = t

    cam_center = np.linalg.inv(W2C)[:3, -1]
    cam_dist = np.linalg.norm(cam_center)
    cam_view = cam_center / cam_dist


    #####################TEST
    # rpc projection
    utm_points = world_points * scales + translation
    lat, lon = eastnorth_to_latlon(utm_points[:,0], utm_points[:,1], 17, 'N')
    alt = utm_points[:, 2]
    rpc_model = RPCModel(meta_dict)
    u, v = rpc_model.projection(lat, lon, alt)
    # affine projection
    ndcalt = (full_affine @ np.concatenate([world_points, np.ones((world_points.shape[0],1))], axis=-1).T).T[:, :3]
    h, w = img.shape[:2]
    u1, v1 = ndc2pix(ndcalt[:,0], w), ndc2pix(ndcalt[:,1], h)
    # sat-sfm
    xyz_ic = (W2C @ np.concatenate([world_points, np.ones((world_points.shape[0],1))], axis=-1).T).T[:, :3]
    uvs = (K @ xyz_ic.T).T
    u2, v2 = uvs[:,0]/uvs[:,-1], uvs[:,1]/uvs[:,-1]
    print(image_file)
    print(np.abs(u1-u).mean(), np.abs(v1-v).mean(), np.sqrt((u1-u)*(u1-u) + (v1-v)*(v1-v)).mean())
    print(np.abs(u2-u).mean(), np.abs(v2-v).mean(), np.sqrt((u2-u)*(u2-u) + (v2-v)*(v2-v)).mean())

    # # vis
    # num = 50
    # idx = np.random.randint(0, len(u1), num)
    # u1, v1, u2, v2 = u[idx], v[idx], u1[idx], v1[idx]
    # vsi_img = np.concatenate((img, img), axis=1)
    # w = img.shape[1]
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(vsi_img)
    # cmap = plt.get_cmap('jet')
    # for i in range(num):
    #     x1 = u1[i]
    #     y1 = v1[i]
    #     x2 = u2[i]
    #     y2 = v2[i]
    #     plt.plot([x1, x2 + w], [y1, y2], '-+', color=cmap(i / (num - 1)), scalex=False, scaley=False)
    # plt.show()
    ###########################
   

    new_img = img
    image_name = image_file.split('/')[-1]
    imageio.imwrite(os.path.join(output_path, 'images', image_name.replace('tif', 'png')), new_img)

    sun_azimuth = meta_dict['sun_azimuth']
    sun_elevation = meta_dict['sun_elevation']

    sun_view = sun_angles_to_enu(sun_azimuth, sun_elevation)
    # 太阳越远，越准确
    sun_position = sun_view * cam_dist * 2.0  
    w2c = sunview_2_w2c(sun_view, sun_position)
    sun_affine_matrix = w2c[:2, :]
    full_sun_affine = np.zeros((4,4))
    full_sun_affine[:2, :4] = sun_affine_matrix
    full_sun_affine[2, 2] = scales[0, -1]
    full_sun_affine[2, 3] = translation[0, -1]
    full_sun_affine[3, 3] = 1.0

    # #########################TEST
    # xyz_ic = (w2c @ np.concatenate([world_points, np.ones((world_points.shape[0],1))], axis=-1).T).T[:, :3]
    # # save as ply
    # from plyfile import PlyElement, PlyData
    # elements = np.empty(xyz_ic.shape[0], dtype=[('x','f4'), ('y','f4'), ('z', 'f4')])
    # elements[:] = list(map(tuple, xyz_ic))
    # el = PlyElement.describe(elements, 'vertex')
    # PlyData([el]).write("./ply.ply")
    # ####################################

    cam_dict = {
        'affine': full_affine.flatten().tolist(),
        'sun_affine': full_sun_affine.flatten().tolist(),
        'K': K.flatten().tolist(),
        'W2C': W2C.flatten().tolist(),
        'img_size': [new_img.shape[1], new_img.shape[0]]
    }
    with open(os.path.join(output_path, 'cameras', image_name.replace('tif', 'json')), 'w') as fp:
        json.dump(cam_dict, fp, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_path', type=str, default="/home/csuzhang/disk/satedata/DFC19/JAX_068", help='Folder containing input data.')
    parser.add_argument('--relative_image_path', default='input')
    parser.add_argument('--relative_rpc_path', default='rpcs')
    parser.add_argument('--relative_gt_path', default='gts')
    parser.add_argument('--max_processes', type=int, default=-1)
    args = parser.parse_args()

    image_path = os.path.join(args.scene_path, args.relative_image_path)
    rpc_path = os.path.join(args.scene_path, args.relative_rpc_path)
    gt_path = os.path.join(args.scene_path, args.relative_gt_path)
    output_path = args.scene_path

    for f in os.listdir(gt_path):
        if f.endswith('_CLS.tif'):
            cls_file = os.path.join(gt_path, f)
        if f.endswith('_DSM.tif'):
            dsm_file = os.path.join(gt_path, f)
        if f.endswith('_DSM.txt'):
            bbx_file = os.path.join(gt_path, f)
            
    if args.max_processes <= 0:
        max_processes = multiprocessing.cpu_count()
    else:
        max_processes = args.max_processes

    # read tile bounds
    easting, northing, pixels, gsd = np.loadtxt(bbx_file)
    pixels = int(pixels)
    ul_utm_e = easting
    ul_utm_n = northing + (pixels - 1) * gsd
    site_width = pixels * gsd
    site_height = pixels * gsd

    # read dsm
    dsm = tifffile.imread(dsm_file)
    alt_min = float(np.nanmin(dsm))
    alt_max = float(np.nanmax(dsm))

    bbx_dict = {
        'easting': [ul_utm_e, ul_utm_e+site_width],
        'northing': [ul_utm_n-site_height, ul_utm_n],
        'alt': [alt_min-5, alt_max+5],
    }
    with open(os.path.join(output_path, 'bbox.json'), 'w') as fp:
        json.dump(bbx_dict, fp, indent=2)

    if 'JAX' in os.path.basename(dsm_file):
        zone_number = 17
        hemisphere = 'N'
    elif 'OMA' in os.path.basename(dsm_file):
        zone_number = 15
        hemisphere = 'N'

    ## 确定 ENU 坐标系
    ENU_easting = easting + (pixels - 1) / 2. * gsd
    ENU_northing = northing + (pixels - 1) / 2. * gsd
    ENU_alt = (alt_min + alt_max) / 2.0
    translation = np.array([ENU_easting, ENU_northing, ENU_alt]).reshape(1,3)

    # sx = site_width / 2.0
    # sy = site_height / 2.0
    # sz = (alt_max - alt_min) / 2.0
    sx = sy = sz = max(site_height, site_width)   # must have same scales
    scales = np.array([sx, sy, sz]).reshape(1,3)

    utm_e, utm_n = np.meshgrid(np.linspace(ul_utm_e, ul_utm_e+site_width, dsm.shape[1]),
                               np.linspace(ul_utm_n, ul_utm_n-site_height, dsm.shape[0]))
    utm_e = utm_e.reshape((-1))
    utm_n = utm_n.reshape((-1))
    dsm = dsm.reshape((-1))
    lat, lon = eastnorth_to_latlon(utm_e, utm_n, zone_number=zone_number, hemisphere=hemisphere)
    world_points = (np.vstack([utm_e, utm_n, dsm]).T - translation) / scales

    lat_minmax = [np.nanmin(lat), np.nanmax(lat)]
    lon_minmax = [np.nanmin(lon), np.nanmax(lon)]
    alt_minmax = [alt_min-5., alt_max+5.]

    os.makedirs(os.path.join(output_path, 'cameras'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)

    list_images = sorted([os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith('.tif')])
    list_rpcs =  sorted([os.path.join(rpc_path, f) for f in os.listdir(rpc_path) if f.endswith('.json')])

    for img_file, rpc_file in zip(list_images, list_rpcs):
        _process_single_image(img_file, rpc_file, output_path, lat_minmax, lon_minmax, alt_minmax, translation, scales, world_points)

    # assert len(list_images) == len(list_rpcs)
    # with multiprocessing.Pool(processes=max_processes) as pool:
    #     for img_file, rpc_file in zip(list_images, list_rpcs):
    #         pool.apply_async(_process_single_image, args=(img_file, rpc_file, output_path, lat_minmax, lon_minmax, alt_minmax, translation, scales))
    #     pool.close()
    #     pool.join()
