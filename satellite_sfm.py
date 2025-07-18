"""
Satellite Structure from Motion (SfM) Pipeline

This module implements a complete SfM pipeline for satellite imagery, including:
- Initial feature extraction and matching
- Camera parameter estimation
- Multi-stage bundle adjustment
- Geometric correction (skew, rotation)
- Visualization tools

Key components:
- run_sfm(): Main SfM pipeline
- visualize_correspondence(): Visualization of feature matches
- Geometric correction functions
"""

import os
import json
import argparse
import numpy as np
import logging
from collections import Iterable
import shutil
import multiprocessing

from utils.preprocess_us3d import preprocess_us3d
from utils.colmap_sfm_utils import init_posed_sfm, extract_camera_dict
from utils.skew_correct import run_skew_correct
from utils.rotation_correct import run_rotation_correct
from utils.focal_correct import run_focal_correct

from utils.colmap_read_write_model import read_model, write_model, Camera, Image, Point3D, qvec2rotmat, rotmat2qvec

def run_sfm(scene_path, 
            reproj_err_threshold=[32.0, 2.0],
            mapper_ba_refine_principal_point=0,
            mapper_ba_refine_focal_length=0,
            global_ba_refine_principal_point=1,
            global_ba_refine_focal_length=0,
            global_ba_refine_extrinsics=0):
    """
    Run the complete SfM pipeline for satellite images.

    Args:
        scene_path (str): Path to scene directory containing:
            - images/: Input images
            - cam_dict.json: Camera parameters
            - sparse/0: Initial reconstruction
            - database.db: Feature database
        
        reproj_err_threshold (list): Reprojection error thresholds for different stages
        mapper_ba_refine_principal_point (int): Refine principal point in mapper BA
        mapper_ba_refine_focal_length (int): Refine focal length in mapper BA 
        global_ba_refine_principal_point (int): Refine principal point in global BA
        global_ba_refine_focal_length (int): Refine focal length in global BA
        global_ba_refine_extrinsics (int): Refine extrinsics in global BA

    Pipeline steps:
    1. Feature extraction using COLMAP
    2. Feature matching
    3. Initial triangulation
    4. Bundle adjustment
    5. Camera parameter refinement
    """
    remote_scene_path = '/workspace'
    COLMAP_BIN = f"docker run -it --rm --gpus all -u $(id -u):$(id -g) \
                        -w /workspace \
                        -v {os.path.abspath(scene_path)}:{remote_scene_path} \
                        colmapforvissat"

    ## feature extraction
    feat_extracton_cmd = f"{COLMAP_BIN} feature_extractor \
        --database_path {os.path.join(remote_scene_path, 'database.db')} \
        --image_path {os.path.join(remote_scene_path, 'images')} \
        --ImageReader.camera_model PERSPECTIVE \
        --SiftExtraction.max_image_size 10000  \
        --SiftExtraction.estimate_affine_shape 0 \
        --SiftExtraction.domain_size_pooling 1 \
        --SiftExtraction.max_num_features 20000 \
        --SiftExtraction.num_threads -1 \
        --SiftExtraction.use_gpu 1 \
        --SiftExtraction.gpu_index -1"
    print(feat_extracton_cmd)
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)
    
    ## feature matching
    feat_matching_cmd = f"{COLMAP_BIN} exhaustive_matcher \
        --database_path {os.path.join(remote_scene_path, 'database.db')} \
        --SiftMatching.guided_matching 1 \
        --SiftMatching.num_threads -1 \
        --SiftMatching.max_error 3 \
        --SiftMatching.max_num_matches 50000 \
        --SiftMatching.use_gpu 1 \
        --SiftMatching.gpu_index -1"
    print(feat_matching_cmd)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)
    
    os.makedirs(os.path.join(scene_path, 'sparse/base'), exist_ok=True)

    if not isinstance(reproj_err_threshold, Iterable):
        reproj_err_threshold = [reproj_err_threshold]
    
    for err in reproj_err_threshold:
        ## transfrom cam_dict as colmap form
        init_posed_sfm(
            db_file=os.path.join(scene_path, 'database.db'),
            cam_dict_file=os.path.join(scene_path, 'cam_dict.json'),
            out_dir=os.path.join(scene_path, 'sparse/base')
        )
    
        ## triangulate
        point_triangulator_cmd = f"{COLMAP_BIN} point_triangulator \
            --database_path {os.path.join(remote_scene_path, 'database.db')} \
            --image_path {os.path.join(remote_scene_path, 'images')} \
            --input_path {os.path.join(remote_scene_path, 'sparse/base')} \
            --output_path {os.path.join(remote_scene_path, 'sparse/base')} \
            --Mapper.filter_min_tri_angle 4.99 \
            --Mapper.init_max_forward_motion 1e20 \
            --Mapper.tri_min_angle 5.00 \
            --Mapper.tri_merge_max_reproj_error {err} \
            --Mapper.tri_complete_max_reproj_error {err} \
            --Mapper.filter_max_reproj_error {err} \
            --Mapper.extract_colors 1 \
            --Mapper.ba_refine_principal_point {mapper_ba_refine_principal_point} \
            --Mapper.ba_refine_focal_length {mapper_ba_refine_focal_length} \
            --Mapper.ba_refine_extra_params 0 \
            --Mapper.max_extra_param 1e20 \
            --Mapper.ba_local_num_images 6 \
            --Mapper.ba_local_max_num_iterations 100 \
            --Mapper.ba_global_images_ratio 1.0000001 \
            --Mapper.ba_global_max_num_iterations 100 \
            --Mapper.tri_ignore_two_view_tracks 0"
        print(point_triangulator_cmd)
        exit_code = os.system(point_triangulator_cmd)
        if exit_code != 0:
            logging.error(f"Triangulation failed with code {exit_code}. Exiting.")
            exit(exit_code)
    
        ## global bundle adjustment
        # one meter is roughly three pixels, we should square it
        global_ba_cmd = f"{COLMAP_BIN} bundle_adjuster \
            --input_path {os.path.join(remote_scene_path, 'sparse/base')} \
            --output_path {os.path.join(remote_scene_path, 'sparse/base')} \
            --BundleAdjustment.max_num_iterations 5000 \
            --BundleAdjustment.refine_focal_length {global_ba_refine_focal_length} \
            --BundleAdjustment.refine_principal_point {global_ba_refine_principal_point} \
            --BundleAdjustment.refine_extra_params 0 \
            --BundleAdjustment.refine_extrinsics {global_ba_refine_extrinsics} \
            --BundleAdjustment.function_tolerance 0 \
            --BundleAdjustment.gradient_tolerance 0 \
            --BundleAdjustment.parameter_tolerance 1e-10 \
            --BundleAdjustment.constrain_points 1 \
            --BundleAdjustment.constrain_points_loss_weight 0.01"
        print(global_ba_cmd)
        exit_code = os.system(global_ba_cmd)
        if exit_code != 0:
            logging.error(f"Global bundle failed with code {exit_code}. Exiting.")
            exit(exit_code)

        ## update camera dict
        cam_dict_adjusted = extract_camera_dict(os.path.join(scene_path, 'sparse/base'))
        with open(os.path.join(scene_path, 'cam_dict.json'), 'w') as fp:
            json.dump(cam_dict_adjusted, fp, indent=2)


def visualize_correspondence(images, cameras, points):
    """
    Visualize feature correspondences between two randomly selected images.

    Args:
        images (dict): Dictionary of Image objects from COLMAP model
        cameras (dict): Dictionary of Camera objects from COLMAP model
        points (dict): Dictionary of Point3D objects from COLMAP model

    Displays:
        Matplotlib plot showing:
        - Two images side by side
        - Random sample of feature matches between them
        - Lines connecting corresponding features
    """
    images = [v for k,v in images.items()]
    points = np.vstack([p.xyz for i,p in points.items()])
    
    first_idx = np.random.randint(0, len(images))
    img1 = images.pop(first_idx)
    R1 = qvec2rotmat(img1.qvec)
    t1 = img1.tvec.reshape((3,1))
    cam1 = cameras[img1.camera_id]
    K1 = np.array([
            [cam1.params[0], 0, cam1.params[2]],
            [0, cam1.params[1], cam1.params[3]],
            [0, 0, 1]])

    second_idx = np.random.randint(0, len(images))
    img2 = images[second_idx]
    R2 = qvec2rotmat(img2.qvec)
    t2 = img2.tvec.reshape((3,1))
    cam2 = cameras[img2.camera_id]
    K2 = np.array([
            [cam2.params[0], 0, cam2.params[2]],
            [0, cam2.params[1], cam2.params[3]],
            [0, 0, 1]])
        
    import imageio.v2 as imageio
    img1 = imageio.imread(os.path.join(args.scene_path, 'images', img1.name))[:,:,:3]
    img2 = imageio.imread(os.path.join(args.scene_path, 'images', img2.name))[:,:,:3]

    X_cam = (R1 @ points.T + t1)  #(3,N)
    uv = K1 @ X_cam
    u1, v1 = uv[0,:]/uv[2,:], uv[1,:]/uv[2,:]

    X_cam = (R2 @ points.T + t2)
    uv = K2 @ X_cam
    u2, v2 = uv[0,:]/uv[2,:], uv[1,:]/uv[2,:]

    num = 50
    idx = np.random.randint(0, len(u1), num)
    u1, v1, u2, v2 = u1[idx], v1[idx], u2[idx], v2[idx]

    if img1.shape[0] < img2.shape[0]:
        img1 = np.pad(img1, ((0, img2.shape[0]-img1.shape[0]), (0, 0), (0, 0)))
    if img1.shape[1] < img2.shape[1]:
        img1 = np.pad(img1, ((0, 0), (0, img2.shape[1]-img1.shape[1]), (0, 0)))

    img2 = np.pad(img2, ((0, img1.shape[0]-img2.shape[0]), (0, img1.shape[1]-img2.shape[1]), (0, 0)))
    img = np.concatenate((img1, img2), axis=1)
    w = img1.shape[1]

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(img)
    cmap = plt.get_cmap('jet')
    for i in range(num):
        x1 = u1[i]
        y1 = v1[i]
        x2 = u2[i]
        y2 = v2[i]
        plt.plot([x1, x2 + w], [y1, y2], '-+', color=cmap(i / (num - 1)), scalex=False, scaley=False)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_path', type=str, default="data/OMA_042", help='Folder containing input data.')
    parser.add_argument('--center_crop', action='store_false')
    parser.add_argument('--skew_correct', action='store_false')
    parser.add_argument('--rot_correct', action='store_false')
    parser.add_argument('--max_processes', type=int, default=-1)
    args = parser.parse_args()
    
    # if True:
    #     from utils.colmap_read_write_model import read_model, qvec2rotmat
    #     cameras, images, points = read_model(os.path.join(args.scene_path, 'sparse/base'))
    #     visualize_correspondence(images, cameras, points)
        

    if args.max_processes <= 0:
        max_processes = multiprocessing.cpu_count()
    else:
        max_processes = args.max_processes

    for f in os.listdir(args.scene_path):
        if f=='input' or f.endswith('_CLS.tif') or f.endswith('_DSM.tif') or f.endswith('_DSM.txt'):
            continue
        if os.path.isfile(os.path.join(args.scene_path, f)):
            os.remove(os.path.join(args.scene_path, f))
        if os.path.isdir(os.path.join(args.scene_path, f)):
            shutil.rmtree(os.path.join(args.scene_path, f))

    ## preprocess US3D dataset
    preprocess_us3d(args.scene_path, os.path.join(args.scene_path, 'preprocess'), center_crop=args.center_crop, max_processes=max_processes)

    ## First time SFM
    run_sfm(os.path.join(args.scene_path, 'preprocess'), 
            reproj_err_threshold=[32.0, 2.0], 
            mapper_ba_refine_principal_point=1,
            mapper_ba_refine_focal_length=0,
            global_ba_refine_principal_point=1,
            global_ba_refine_focal_length=0,
            global_ba_refine_extrinsics=0)
    
    all_cam_dict = json.load(open(os.path.join(args.scene_path, 'preprocess', 'cam_dict.json')))
    for f, cam in all_cam_dict.items():
        with open(os.path.join(args.scene_path, 'preprocess/cameras', f"{f}.json"), 'w') as fp:
            json.dump(cam, fp, indent=2)

    cam_path = os.path.join(args.scene_path, 'preprocess/cameras')
    img_path = os.path.join(args.scene_path, 'preprocess/images')

    ## skew correction
    if args.skew_correct or args.rot_correct:
        print("Start Skew Correction")
        skew_cam_path = os.path.join(args.scene_path, 'skew_correct/cameras')
        skew_img_path = os.path.join(args.scene_path, 'skew_correct/images')
        run_skew_correct(img_path, cam_path, skew_img_path, skew_cam_path, max_processes=max_processes)
        cam_path = skew_cam_path
        img_path = skew_img_path
    
    # if args.focal_correct:
    #     print("Start focal Correct")
    #     focal_cam_path = os.path.join(args.scene_path, 'focal_correct/cameras')
    #     focal_img_path = os.path.join(args.scene_path, 'focal_correct/images')
    #     run_focal_correct(img_path, cam_path, focal_img_path, focal_cam_path, max_processes=max_processes)
    #     cam_path = focal_cam_path
    #     img_path = focal_img_path
    
    ## geometric correction
    if args.rot_correct:
        print("Start Rotation Transformation and Correction")
        rot_cam_path = os.path.join(args.scene_path, 'rot_correct/cameras')
        rot_img_path = os.path.join(args.scene_path, 'rot_correct/images')
        run_rotation_correct(img_path, cam_path, rot_img_path, rot_cam_path, max_processes=max_processes)
        cam_path = rot_cam_path
        img_path = rot_img_path

    # Update Cameras and images
    all_cam_dict = {}
    for f in os.listdir(cam_path):
        cam_dict = json.load(open(os.path.join(cam_path, f)))
        all_cam_dict[f.split('.')[0]] = cam_dict
    with open(os.path.join(args.scene_path, 'cam_dict.json'), 'w') as fp:
        json.dump(all_cam_dict, fp, indent=2)
    
    # copy images
    image_path = os.path.join(args.scene_path, 'images')
    if os.path.exists(image_path): shutil.rmtree(image_path)
    os.makedirs(image_path, exist_ok=True)
    for img in os.listdir(img_path):
        orig_path = os.path.join(img_path, img)
        new_path = os.path.join(image_path, img)
        shutil.copy(orig_path, new_path)

    # Run Sceond-time SFM
    run_sfm(args.scene_path, 
            reproj_err_threshold=[2.0, 1.0],
            mapper_ba_refine_principal_point=0,
            mapper_ba_refine_focal_length=1,
            global_ba_refine_principal_point=0,
            global_ba_refine_focal_length=1,
            global_ba_refine_extrinsics=0)

    # read models
    cameras, images, points3D = read_model(os.path.join(args.scene_path, 'sparse/base'))
    sorted_image = dict(sorted(images.items(), key=lambda x:x[1].name))
    new_cameras, new_images, new_points3D = {}, {}, {}
    list_zm, list_zM = [], []

    # if args.scale_scene:
    #     enu_bbx = json.load(open(os.path.join(args.scene_path, 'preprocess/enu_bbx.json')))
    #     scene_scale = np.sqrt(
    #         (np.max(enu_bbx['e_minmax']) - np.min(enu_bbx['e_minmax'])) ** 2 +  \
    #         (np.max(enu_bbx['n_minmax']) - np.min(enu_bbx['n_minmax'])) ** 2 +  \
    #         (np.max(enu_bbx['u_minmax']) - np.min(enu_bbx['u_minmax'])) ** 2
    #     )
    # else:
    #     scene_scale = 1.0
    scene_scale = 1.0


    # update point3d
    points = np.vstack([p.xyz for i,p in points3D.items()]) / scene_scale
    for idx, p in points3D.items():
        new_points3D[idx] = Point3D(
            id=p.id, xyz=p.xyz / scene_scale, rgb=p.rgb, error=p.error,
            image_ids=p.image_ids, point2D_idxs=p.point2D_idxs
        )
    
    for i, (idx, img) in enumerate(sorted_image.items()):
        cam = cameras[img.camera_id]
        fx, fy, cx, cy = cam.params[:4]
        # update camera intrinsic
        new_cameras[cam.id] = Camera(
            id=cam.id, model="PINHOLE", 
            width=cam.width, height=cam.height,
            params=np.array([fx,fy,cx,cy])
        )
        
        # scale scene
        R = qvec2rotmat(img.qvec)
        t = img.tvec.reshape((3,1)) / scene_scale

        new_images[idx] = Image(
            id=img.id, qvec=img.qvec, tvec=img.tvec / scene_scale,
            camera_id=img.camera_id, name=img.name,
            xys=img.xys, point3D_ids=img.point3D_ids
        )

        # compute znear and zfar
        z = (R @ points.T + t)[-1,:]
        list_zm.append(np.min(z))
        list_zM.append(np.max(z))

    print(f"scene_scale: {scene_scale}, znear: {min(list_zm)}, zfar: {max(list_zM)}")

    sparse_path = os.path.join(args.scene_path, 'sparse/0')
    if os.path.exists(sparse_path): shutil.rmtree(sparse_path)
    os.makedirs(sparse_path, exist_ok=True)
    write_model(new_cameras, new_images, new_points3D, sparse_path, ".bin")


    if True:
        from utils.colmap_read_write_model import read_model, qvec2rotmat
        cameras, images, points = read_model(os.path.join(args.scene_path, 'sparse/base'))
        visualize_correspondence(images, cameras, points)
