import numpy as np
import json
import imageio.v2 as imageio
import cv2
import multiprocessing
import os


def warp_affine(img_src, affine_matrix, mask_obb):
    '''
    img_src: [H, W, 3] numpy array
    affine_matrix: [2, 3] numpy array
    mask_obb: [3, 4]s numpy array
    '''
    height, width = img_src.shape[:2]

    # compute bounding box
    bbx = np.dot(affine_matrix, np.concatenate([mask_obb, np.array([[1,1,1,1]])]))
    col_min = np.min(bbx[0, :])
    col_max = np.max(bbx[0, :])
    row_min = np.min(bbx[1, :])
    row_max = np.max(bbx[1, :])

    w = int(np.round(col_max - col_min + 1))
    h = int(np.round(row_max - row_min + 1))

    # add offset to the affine_matrix
    affine_matrix[0, 2] -= col_min
    affine_matrix[1, 2] -= row_min

    off_set = (-col_min, -row_min)

    # warp image
    img_dst = cv2.warpAffine(img_src, affine_matrix, (w, h))

    mask = np.ones((img_src.shape[0], img_src.shape[1]), dtype=np.uint8) * 255
    warped_mask = cv2.warpAffine(mask, affine_matrix, (w, h))
    img_dst = np.concatenate([img_dst, warped_mask[:,:,None]], axis=-1)

    # compute mask-corner-uv
    new_mask_bbx = np.dot(affine_matrix, np.concatenate([mask_obb, np.array([[1,1,1,1]])]))

    assert (h == img_dst.shape[0] and w == img_dst.shape[1])
    return img_dst, off_set, new_mask_bbx


def skew_correct(cam_file, img_file, out_cam_file, out_img_file):
    # read camera info
    cam_dict = json.load(open(cam_file))
    K = np.array(cam_dict['K']).reshape((4, 4))
    fx, s, cx = K[0, 0], K[0, 1], K[0, 2]
    fy, cy = K[1, 1], K[1, 2]

    # compute homography and update s, cx
    norm_skew = s / fy
    cx = cx - s * cy / fy
    s = 0.
    
    img_src = imageio.imread(img_file)
    orig_h, orig_w = img_src.shape[:2]
    if img_src.shape[-1] == 4:
        mask = img_src[:,:,-1]
        img_src = img_src[:,:,:3]
    elif img_src.shape[-1] == 3:
        mask = np.ones((img_src.shape[0], img_src.shape[1]), dtype=np.uint8) * 255
    
    if 'mask_obb' in cam_dict:
        mask_obb = np.array(cam_dict['mask_obb']).reshape((2,4))
    else:
        mask_obb = np.array([[0, orig_w, orig_w, 0],
                            [0, 0, orig_h, orig_h]])

    # warp image
    affine_matrix = np.array([[1., -norm_skew, 0.],
                              [0., 1., 0.]])
    img_dst, off_set, new_mask_obb = warp_affine(img_src, affine_matrix, mask_obb)
    cx += off_set[0]
    cy += off_set[1]

    # if keep_img_size:
    #     if img_dst.shape[0] > orig_h:
    #         img_dst = img_dst[:orig_h, :, :]
    #     elif img_dst.shape[0] < orig_h:
    #         img_dst = np.pad(img_dst, ((0, orig_h-img_dst.shape[0]), (0, 0), (0, 0)))

    #     if img_dst.shape[1] > orig_w:
    #         img_dst = img_dst[:, :orig_w, :]
    #     elif img_dst.shape[1] < orig_w:
    #         img_dst = np.pad(img_dst, ((0, 0), (0, orig_w-img_dst.shape[1]), (0, 0)))

    new_h, new_w = img_dst.shape[:2]
    cam_dict['img_size'] = [new_w, new_h]
    K[0, 1] = 0
    K[0, 2] = cx
    K[1, 2] = cy
    cam_dict['K'] = K.flatten().tolist()
    cam_dict['mask_obb'] = new_mask_obb.flatten().tolist()

    with open(out_cam_file, 'w') as fp:
        json.dump(cam_dict, fp, indent=2, sort_keys=True)
    
    imageio.imwrite(out_img_file, img_dst)


def run_skew_correct(input_img_path, input_cam_path, out_img_path, out_cam_path):
    os.makedirs(out_cam_path, exist_ok=True)
    os.makedirs(out_img_path, exist_ok=True)

    list_cam = [os.path.join(input_cam_path, f) for f in sorted(os.listdir(input_cam_path))]
    list_img = [os.path.join(input_img_path, f) for f in sorted(os.listdir(input_img_path))]
    assert len(list_cam) == len(list_img)

    list_out_cam = [os.path.join(out_cam_path, f) for f in sorted([c.split('/')[-1] for c in list_cam])]
    list_out_img = [os.path.join(out_img_path, f) for f in sorted([c.split('/')[-1] for c in list_img])]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for cam, img, out_cam, out_img in zip(list_cam, list_img, list_out_cam, list_out_img):
            pool.apply_async(skew_correct, args=(cam, img, out_cam, out_img))
        pool.close()
        pool.join()


if __name__ == "__main__":
    import argparse
    import os
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser("Test")
    parser.add_argument("--cam_file", default="data/JAX_214/preprocess/cameras/JAX_214_001_RGB.json")
    parser.add_argument("--img_file", default="data/JAX_214/preprocess/images/JAX_214_001_RGB.png")
    parser.add_argument("--points_file", default="data/JAX_214/sparse/0/points3D.txt")
    parser.add_argument("--output_path", default="./test_output")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    out_cam_file = os.path.join(args.output_path, args.cam_file.split('/')[-1])
    out_img_file = os.path.join(args.output_path, args.img_file.split('/')[-1])
    skew_correct(args.cam_file, args.img_file, out_cam_file, out_img_file)

    from colmap_read_write_model import read_points3D_text_as_numpy

    points, _, _  = read_points3D_text_as_numpy(args.points_file)
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
    if img1.shape[0] < img2.shape[0]:
        img1 = np.pad(img1, ((0, img2.shape[0]-img1.shape[0]), (0, 0), (0, 0)))
    if img1.shape[1] < img2.shape[1]:
        img1 = np.pad(img1, ((0, 0), (0, img2.shape[1]-img1.shape[1]), (0, 0)))
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