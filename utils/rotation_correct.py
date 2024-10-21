import json
import numpy as np
import math
import cv2
import imageio.v2 as imageio
import multiprocessing
import os

def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1


    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error
                count += 1

    return xyzs, rgbs, errors

def rotation_correct(cam_file, img_file, out_cam_file, out_img_file, center_crop=True):
    img_src = imageio.imread(img_file)
    cam_dict = json.load(open(cam_file))
    if img_src.shape[-1] == 4:
        mask = img_src[:,:,-1]
        img_src = img_src[:,:,:3]
    elif img_src.shape[-1] == 3:
        mask = np.ones((img_src.shape[0], img_src.shape[1]), dtype=np.uint8) * 255

    K = np.array(cam_dict['K']).reshape((4,4))
    W2C = np.array(cam_dict['W2C']).reshape((4,4))
    img_size = np.array(cam_dict['img_size'])

    if 'mask_obb' in cam_dict:
        mask_obb = np.array(cam_dict['mask_obb']).reshape((2,4))
    else:
        center_crop = False
        print(f"key 'mask_abb' not in the camera dict, so can't center crop, set center_crop=False")

    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    w, h = img_size[0], img_size[1]

    R = W2C[:3, :3]
    t = W2C[:3, 3:4]

    d = np.array([(w/2-cx)/fx, (h/2-cy)/fy, 1])
    d = d / np.linalg.norm(d)

    z_axis = d
    if np.isclose(np.abs(z_axis[1]), 1.0):
        up = np.array([1, 0, 0])
    else:
        up = np.array([0, 1, 0])

    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    R_delta = np.stack((x_axis, y_axis, z_axis), axis=0)

    ## update R, t
    R_prime = R_delta @ R
    t_prime = R_delta @ t

    # K_prime = np.array([[fx, 0, w/2],
    #                     [0, fy, h/2],
    #                     [0,  0,  1]])

    K_prime = np.array([[fx, 0, w/2],
                        [0, fy, h/2],
                        [0,  0,  1]])
    
    new_K = np.eye(4)
    new_K[:3, :3] = K_prime
    new_W2C = np.eye(4)
    new_W2C[:3, :3] = R_prime
    new_W2C[:3, 3:4] = t_prime

    cam_dict = {}
    cam_dict['img_size'] = [int(w), int(h)]
    cam_dict['K'] = new_K.flatten().tolist()
    cam_dict['W2C'] = new_W2C.flatten().tolist()

    H = K_prime @ R_delta @ np.linalg.inv(K[:3,:3])
    warped_mask = cv2.warpPerspective(mask, H, (int(w), int(h)))
    warped_image = cv2.warpPerspective(img_src, H, (int(w), int(h)))
    
    if center_crop:
        bbx = np.dot(H, np.concatenate([mask_obb, np.array([[1,1,1,1]])], axis=0))
        bbx = bbx[:2,:] / bbx[2:3,:]
        top = max(bbx[1,0], bbx[1,1])
        buttom = min(bbx[1,2], bbx[1,3])
        row_margin = max(int(top+0.5), int(h-buttom+0.5))

        left = max(bbx[0,0], bbx[0,3])
        right = min(bbx[0,1],bbx[0,2])
        col_margin = max(int(left+0.5), int(w-right+0.5))

        warped_image = warped_image[row_margin:w-row_margin, col_margin:h-col_margin, :]
        cam_dict['img_size'] = [int(warped_image.shape[1]), int(warped_image.shape[0])]
        new_K[0, 2] -= col_margin
        new_K[1, 2] -= row_margin
        cam_dict['K'] = new_K.flatten().tolist()
    else:
        warped_image = np.concatenate([warped_image, warped_mask[:,:,None]], axis=-1)
    # save
    with open(out_cam_file, 'w') as fp:
        json.dump(cam_dict, fp, indent=2, sort_keys=True)
    imageio.imwrite(out_img_file, warped_image)


def run_rotation_correct(input_img_path, input_cam_path, out_img_path, out_cam_path):
    os.makedirs(out_cam_path, exist_ok=True)
    os.makedirs(out_img_path, exist_ok=True)

    list_cam = [os.path.join(input_cam_path, f) for f in sorted(os.listdir(input_cam_path))]
    list_img = [os.path.join(input_img_path, f) for f in sorted(os.listdir(input_img_path))]
    assert len(list_cam) == len(list_img)

    list_out_cam = [os.path.join(out_cam_path, f) for f in sorted([c.split('/')[-1] for c in list_cam])]
    list_out_img = [os.path.join(out_img_path, f) for f in sorted([c.split('/')[-1] for c in list_img])]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for cam, img, out_cam, out_img in zip(list_cam, list_img, list_out_cam, list_out_img):
            pool.apply_async(rotation_correct, args=(cam, img, out_cam, out_img))
        pool.close()
        pool.join()


if __name__ == "__main__":
    import argparse
    import os
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser("Test")
    parser.add_argument("--cam_file", default="data/JAX_214/skew_correct/cameras/JAX_214_001_RGB.json")
    parser.add_argument("--img_file", default="data/JAX_214/skew_correct/images/JAX_214_001_RGB.png")
    parser.add_argument("--points_file", default="data/JAX_214/preprocess/colmap_triangulate_postba/points3D.txt")
    parser.add_argument("--output_path", default="./test_output")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    out_cam_file = os.path.join(args.output_path, args.cam_file.split('/')[-1])
    out_img_file = os.path.join(args.output_path, args.img_file.split('/')[-1])
    rotation_correct(args.cam_file, args.img_file, out_cam_file, out_img_file, True)

    points, _, _  = read_points3D_text(args.points_file)
    img1 = imageio.imread(args.img_file)[:,:,:3]
    cam1 = json.load(open(args.cam_file))
    img2 = imageio.imread(out_img_file)[:,:,:3]
    cam2 = json.load(open(out_cam_file))
    W2C1, W2C2 = np.array(cam1['W2C']).reshape((4,4)), np.array(cam2['W2C']).reshape((4,4))
    R1, t1, R2, t2 = W2C1[:3,:3], W2C1[:3, 3:4], W2C2[:3,:3], W2C2[:3, 3:4] 
    K1, K2 = np.array(cam1['K']).reshape((4,4))[:3,:3], np.array(cam2['K']).reshape((4,4))[:3,:3]
    
    X_cam = (R1 @ points.T + t1)  #(3,N)
    uv = K1 @ X_cam
    u1, v1 = uv[0,:]/uv[2,:], uv[1,:]/uv[2,:]

    X_cam = (R2 @ points.T + t2)
    uv = K2 @ X_cam
    u2, v2 = uv[0,:]/uv[2,:], uv[1,:]/uv[2,:]

    
    num = 30
    idx = np.random.randint(0, len(u1), num)
    u1, v1, u2, v2 = u1[idx], v1[idx], u2[idx], v2[idx]

    img2 = np.pad(img2, ((0, img1.shape[0]-img2.shape[0]), (0, img1.shape[1]-img2.shape[1]), (0, 0)))

    img = np.concatenate((img1, img2), axis=1)
    w = img1.shape[1]

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
    