import argparse
import os
import numpy as np
import tifffile
import pyproj
import pymap3d
import cv2
import json
from plyfile import PlyData, PlyElement
from osgeo import gdal
import shutil
import imageio
import multiprocessing
import rasterio
import rpcm

from preprocess.approximate_rpc_locally import approximate_rpc_locally, approximate_rpc_locally_and_rectify_image


def eastnorth_to_latlon(east, north, zone_number, hemisphere):
    if hemisphere == 'N':
        south = False
    else:
        south = True
    proj = pyproj.Proj(proj='utm', ellps='WGS84', zone=zone_number, south=south)
    lon, lat = proj(east, north, inverse=True)
    return lat, lon


def latlonalt_to_enu(lat, lon, alt, lat0, lon0, alt0):
    e, n, u = pymap3d.geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
    return e, n, u


def read_tif_and_info(tiff_fpath):
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

    rpc_data = dataset.GetMetadata('RPC')
    rpc_dict = {
        'lonOff': float(rpc_data['LONG_OFF']),
        'lonScale': float(rpc_data['LONG_SCALE']),
        'latOff': float(rpc_data['LAT_OFF']),
        'latScale': float(rpc_data['LAT_SCALE']),
        'altOff': float(rpc_data['HEIGHT_OFF']),
        'altScale': float(rpc_data['HEIGHT_SCALE']),
        'rowOff': float(rpc_data['LINE_OFF']),
        'rowScale': float(rpc_data['LINE_SCALE']),
        'colOff': float(rpc_data['SAMP_OFF']),
        'colScale': float(rpc_data['SAMP_SCALE']),
        'rowNum': np.asarray(rpc_data['LINE_NUM_COEFF'].split(), dtype=np.float64).tolist(),
        'rowDen': np.asarray(rpc_data['LINE_DEN_COEFF'].split(), dtype=np.float64).tolist(),
        'colNum': np.asarray(rpc_data['SAMP_NUM_COEFF'].split(), dtype=np.float64).tolist(),
        'colDen': np.asarray(rpc_data['SAMP_DEN_COEFF'].split(), dtype=np.float64).tolist()
    }

    meta_dict = { 'rpc': rpc_dict,
                  'height': img.shape[0],
                  'width': img.shape[1], 
                  'capture_date': capture_date
    }
    return img, meta_dict


def _process_single_image(image_path, f, output_path, lat_minmax, lon_minmax, alt_minmax, ENU_lat, ENU_lon, ENU_alt):
    img, meta_dict = read_tif_and_info(os.path.join(image_path, f))

    with open(os.path.join(output_path, 'metas', f.replace('tif', 'json')), 'w') as fp:
        json.dump(meta_dict, fp, indent=2)

    K, W2C = approximate_rpc_locally(meta_dict, lat_minmax, lon_minmax, alt_minmax, ENU_lat, ENU_lon, ENU_alt)
    new_img = img
    imageio.imwrite(os.path.join(output_path, 'images', f.replace('tif', 'png')), new_img)

    # K, W2C, new_img = approximate_rpc_locally_and_rectify_image(img, meta_dict, lat_minmax, lon_minmax, alt_minmax, ENU_lat, ENU_lon, ENU_alt)
    # imageio.imwrite(os.path.join(output_path, 'images', f.replace('tif', 'png')), new_img)

    cam_dict = {
            'K': K.flatten().tolist(),
            'W2C': W2C.flatten().tolist(),
            'img_size': [new_img.shape[1], new_img.shape[0]]
        }
    with open(os.path.join(output_path, 'cameras', f.replace('tif', 'json')), 'w') as fp:
        json.dump(cam_dict, fp, indent=2)


def geojson_polygon(coords_array):
    """
    define a geojson polygon from a Nx2 numpy array with N 2d coordinates delimiting a boundary
    """
    from shapely.geometry import Polygon

    # first attempt to construct the polygon, assuming the input coords_array are ordered
    # the centroid is computed using shapely.geometry.Polygon.centroid
    # taking the mean is easier but does not handle different densities of points in the edges
    pp = coords_array.tolist()
    poly = Polygon(pp)
    x_c, y_c = np.array(poly.centroid.xy).ravel()

    # check that the polygon is valid, i.e. that non of its segments intersect
    # if the polygon is not valid, then coords_array was not ordered and we have to do it
    # a possible fix is to sort points by polar angle using the centroid (anti-clockwise order)
    if not poly.is_valid:
        pp.sort(key=lambda p: np.arctan2(p[0] - x_c, p[1] - y_c))

    # construct the geojson
    geojson_polygon = {"coordinates": [pp], "type": "Polygon"}
    geojson_polygon["center"] = [x_c, y_c]
    return geojson_polygon

def _crop_geotiff_lonlat_aoi(image_path, f, output_path, lonlat_aoi):
    geotiff_path = os.path.join(image_path, f)
    with rasterio.open(geotiff_path, 'r') as src:
        profile = src.profile
        tags = src.tags()
    crop, x, y = rpcm.utils.crop_aoi(geotiff_path, lonlat_aoi)
    rpc = rpcm.rpc_from_geotiff(geotiff_path)

    rpc.row_offset -= y
    rpc.col_offset -= x

    not_pan = len(crop.shape) > 2
    if not_pan:
        profile["height"] = crop.shape[1]
        profile["width"] = crop.shape[2]
    else:
        profile["height"] = crop.shape[0]
        profile["width"] = crop.shape[1]
        profile["count"] = 1

    out_geotiff_path = os.path.join(output_path, f)
    with rasterio.open(out_geotiff_path, 'w', **profile) as dst:
        if not_pan:
            dst.write(crop)
        else:
            dst.write(crop, 1)
        dst.update_tags(**tags)
        dst.update_tags(ns='RPC', **rpc.to_geotiff_dict())


def run_center_crop(image_path, output_path, easting_range, northing_range, zone_number, hemisphere, max_processes=-1, margin=25):
    if max_processes <= 0:
        max_processes = multiprocessing.cpu_count()
    else:
        max_processes = max_processes
    
    # gt dsm bbx
    utm_e_s, utm_e_e = easting_range
    utm_n_s, utm_n_e = northing_range
    
    utm_e_s, utm_e_e = utm_e_s-margin, utm_e_e+margin
    utm_n_s, utm_n_e = utm_n_s+margin, utm_n_e-margin

    easts = [utm_e_s, utm_e_s, utm_e_e, utm_e_e, utm_e_s]
    norths = [utm_n_s, utm_n_e, utm_n_e, utm_n_s, utm_n_s]
    lats, lons = eastnorth_to_latlon(easts, norths, zone_number, hemisphere=hemisphere)
    lonlat_bbx = geojson_polygon(np.vstack((lons, lats)).T)

    # crop
    os.makedirs(output_path, exist_ok=True)
    with multiprocessing.Pool(processes=max_processes) as pool:
        for f in os.listdir(image_path):
            pool.apply_async(_crop_geotiff_lonlat_aoi, args=(image_path, f, output_path, lonlat_bbx))
        pool.close()
        pool.join()


def preprocess_us3d(scene_path, output_path=None, center_crop=False, max_processes=-1):    
    assert os.path.exists(os.path.join(scene_path, 'input')), "input image path not exists"

    if max_processes <= 0:
        max_processes = multiprocessing.cpu_count()
    else:
        max_processes = max_processes

    bbox_file = [os.path.join(scene_path, f) for f in os.listdir(scene_path) if f.endswith('_DSM.txt')][0]
    dsm_file = [os.path.join(scene_path, f) for f in os.listdir(scene_path) if f.endswith('_DSM.tif')][0]
    dsm_mask_file = [os.path.join(scene_path, f) for f in os.listdir(scene_path) if f.endswith('_CLS.tif')][0]
    assert (bbox_file is not None) and (dsm_file is not None)

    if output_path is None:
        output_path = os.path.join(scene_path, 'preprocess')
    os.makedirs(output_path, exist_ok=True)

    # read tile bounds
    easting, northing, pixels, gsd = np.loadtxt(bbox_file)
    pixels = int(pixels)

    ul_utm_e = easting
    ul_utm_n = northing + (pixels - 1) * gsd

    site_width = pixels * gsd
    site_height = pixels * gsd

    # read dsm
    dsm = tifffile.imread(dsm_file)
    alt_min = float(np.nanmin(dsm))
    alt_max = float(np.nanmax(dsm))

    if 'JAX' in os.path.basename(dsm_file):
        zone_number = 17
        hemisphere = 'N'
    elif 'OMA' in os.path.basename(dsm_file):
        zone_number = 15
        hemisphere = 'N'

    ## 确定 ENU 坐标系
    ENU_easting = easting + (pixels - 1) / 2. * gsd
    ENU_northing = northing + (pixels - 1) / 2. * gsd
    ENU_lat, ENU_lon = eastnorth_to_latlon(ENU_easting, ENU_northing, zone_number, hemisphere=hemisphere)
    ENU_alt = alt_min - 10.0

    ## transform dsm to point-cloud in ENU-coordinate
    # dsm = cv2.resize(dsm, (2*dsm.shape[1], 2*dsm.shape[0]), interpolation=cv2.INTER_NEAREST)   # densify lifted point cloud
    utm_e, utm_n = np.meshgrid(np.linspace(ul_utm_e, ul_utm_e+site_width, dsm.shape[1]),
                               np.linspace(ul_utm_n, ul_utm_n-site_height, dsm.shape[0]))
    utm_e = utm_e.reshape((-1))
    utm_n = utm_n.reshape((-1))
    dsm = dsm.reshape((-1))

    ## center crop
    if center_crop:
        run_center_crop(image_path=os.path.join(scene_path, 'input'),
                        output_path=os.path.join(output_path, 'crops'),
                        easting_range=(ul_utm_e, ul_utm_e+site_width),
                        northing_range=(ul_utm_n, ul_utm_n-site_height),
                        zone_number=zone_number, hemisphere=hemisphere, 
                        max_processes=max_processes)
        image_path = os.path.join(output_path, 'crops')
    else:
        image_path = os.path.join(scene_path, 'input')

    lat, lon = eastnorth_to_latlon(utm_e, utm_n, zone_number=zone_number, hemisphere=hemisphere)

    enu_e, enu_n, enu_u = latlonalt_to_enu(lat, lon, dsm, ENU_lat, ENU_lon, ENU_alt)

    enu_bbx = {
        'e_minmax': [np.nanmin(enu_e), np.nanmax(enu_e)],
        'n_minmax': [np.nanmin(enu_n), np.nanmax(enu_n)],
        'u_minmax': [np.nanmin(enu_u)-5., np.nanmax(enu_u)+5.] }
    with open(os.path.join(output_path, 'enu_bbx.json'), 'w') as fp:
        json.dump(enu_bbx, fp, indent=2)

    lat_minmax = [np.nanmin(lat), np.nanmax(lat)]
    lon_minmax = [np.nanmin(lon), np.nanmax(lon)]
    alt_minmax = [alt_min-5., alt_max+5.]
    latlonalt_bbx = {
        'lat_minmax': lat_minmax,
        'lon_minmax': lon_minmax,
        'alt_minmax': alt_minmax,
    }
    with open(os.path.join(output_path, 'latlonalt_bbx.json'), 'w') as fp:
        json.dump(latlonalt_bbx, fp, indent=2)
    
    elements = np.empty(enu_e.shape[0], dtype=[('x','f4'), ('y','f4'), ('z', 'f4')])
    attributes = np.vstack((enu_e, enu_n, enu_u)).T
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(os.path.join(output_path, "point_cloud_DSM.ply"))

    with rasterio.open(dsm_file, "r") as f:
        profile = f.profile
    
    with rasterio.open(os.path.join(output_path, "enu_DSM.tif"),  "w", **profile) as f:
        enu_dsm = enu_u.reshape(pixels, pixels)
        f.write(enu_dsm, 1)
    
    shutil.copy2(dsm_mask_file, os.path.join(output_path, "enu_CLS.tif"))
    
    with open(os.path.join(output_path, "enu_DSM.txt"), 'w') as f:
        min_e = np.min(enu_e)
        min_n = np.min(enu_n)
        f.write(f"{min_e}\n{min_n}\n{pixels}\n{gsd}")

    ## init
    os.makedirs(os.path.join(output_path, 'metas'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'cameras'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)

    with multiprocessing.Pool(processes=max_processes) as pool:
        for f in os.listdir(image_path):
            pool.apply_async(_process_single_image, args=(image_path, f, output_path, lat_minmax, lon_minmax, alt_minmax, ENU_lat, ENU_lon, ENU_alt))
        pool.close()
        pool.join()

    all_cam_dict = {}
    for f in os.listdir(os.path.join(output_path, 'cameras')):
        cam_dict = json.load(open(os.path.join(output_path, 'cameras', f)))
        all_cam_dict[f.split('.')[0]] = cam_dict
    
    with open(os.path.join(output_path, 'cam_dict.json'), 'w') as fp:
        json.dump(all_cam_dict, fp, indent=2)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_path', '-s', type=str, default="data/JAX_214")
    parser.add_argument('--max_processes', type=int, default=-1)
    args = parser.parse_args()

    preprocess_us3d(args.scene_path)