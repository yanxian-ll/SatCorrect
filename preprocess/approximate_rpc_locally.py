import numpy as np
from scipy import linalg
import cv2

from preprocess.factorize_projection_matrix import factorize_projection_matrix
from preprocess.rpc_model import RPCModel
from preprocess.coordinate_system import latlonalt_to_enu


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


def approximate_rpc_locally(meta_dict, lat_minmax, lon_minmax, alt_minmax, 
                                       observer_lat, observer_lon, observer_alt,
                                       lat_N=100, lon_N=100, alt_N=50):
    '''
        meta_dict: see parse_tiff.py for information about meta_dict
        lat_minmax, lon_minmax, alt_minmax: (2,)
        observer_lat, observer_lon, observer_alt: float
        lat_N, lon_N, alt_N: integers
    '''
    lat, lon, alt, col, row = _generate_samples(meta_dict, lat_minmax, lon_minmax, alt_minmax, 
                                                           lat_N, lon_N, alt_N)

    assert (observer_alt < np.min(alt_minmax))

    e, n, u = latlonalt_to_enu(lat, lon, alt, observer_lat, observer_lon, observer_alt)

    P = _solve_projection_matrix(e, n, u, col, row)
    K, R, t = factorize_projection_matrix(P)

    K_4by4 = np.eye(4)
    K_4by4[:3, :3] = K
    W2C = np.eye(4)
    W2C[:3, :3] = R
    W2C[:3, 3] = t

    return K_4by4, W2C


def svd_factorization(src_pts, dst_pts):
    # SVD
    point_cnt = src_pts.shape[0]
    all_ones = np.ones((point_cnt,1))
    all_zeros = np.zeros((point_cnt, 6))
    x1, y1 = src_pts[:, 0].reshape(-1, 1), src_pts[:, 1].reshape(-1, 1)
    x2, y2 = dst_pts[:, 0].reshape(-1, 1), dst_pts[:, 1].reshape(-1, 1)
    x1, y1 = np.float32(x1), np.float32(y1)
    x2, y2 = np.float32(x2), np.float32(y2)

    B = np.hstack((all_ones, x1, y1, x1 * y1, x1 * x1, y1 * y1))
    T = np.dot(B.T,B)
    R = np.linalg.matrix_rank(T)
    A1 = np.hstack((all_ones, x1, y1, x1 * y1, x1 * x1, y1 * y1,
                    all_zeros))
    A2 = np.hstack((all_zeros,
                   all_ones, x1, y1, x1 * y1, x1 * x1, y1 * y1))
    # A2 = np.hstack((all_zeros,
    #                 all_ones, y1 * y1, x1 * x1, x1 * y1, y1, x1))
    A = np.vstack((A1, A2))
    U, S, V = linalg.svd(A, full_matrices=False)

    b = np.vstack((x2, y2))
    D = np.diag(1 / S)
    H = V.T @ D @ U.T @ b
    H = H.reshape(2,6)

    return H


def approximate_rpc_locally_and_rectify_image(
        image, meta_dict, 
        lat_minmax, lon_minmax, alt_minmax, 
        observer_lat, observer_lon, observer_alt,
        lat_N=100, lon_N=100, alt_N=100
):
    """
    following the code: https://github.com/2022hong/REPMSatelliteStereo
    """
    lat, lon, alt, col, row = _generate_samples(meta_dict, lat_minmax, lon_minmax, alt_minmax, 
                                                           lat_N, lon_N, alt_N)

    assert (observer_alt < np.min(alt_minmax))

    e, n, u = latlonalt_to_enu(lat, lon, alt, observer_lat, observer_lon, observer_alt)

    P = _solve_projection_matrix(e, n, u, col, row)
    K, R, t = factorize_projection_matrix(P)

    K_4by4 = np.eye(4)
    K_4by4[:3, :3] = K
    W2C = np.eye(4)
    W2C[:3, :3] = R
    W2C[:3, 3] = t

    # before rectify
    b = np.ones((e.shape[0],), dtype=float)
    uv_perspective = K @ W2C[:3,:4] @ np.vstack([e,n,u,b])  #(3,N)
    uv_perspective = (uv_perspective[:2,:] / uv_perspective[2:3, :]).T  #(N,2)
    err_col = np.abs(col - uv_perspective[:,0])
    err_row = np.abs(row - uv_perspective[:,1])
    project_err = np.mean(np.sqrt(err_col**2 + err_row**2))
    
    ## rectify
    uv_rpc = np.vstack([col, row]).T  #(N,2)
    H = svd_factorization(uv_perspective, uv_rpc)
    x, y = uv_perspective[:,0], uv_perspective[:,1]
    uv_rectified = (H @ np.vstack([b, x, y, x*y, x*x, y*y])).T  #(N,2)
    new_err_col = np.abs(col - uv_rectified[:,0])
    new_err_row = np.abs(row - uv_rectified[:,1])
    new_project_err = np.mean(np.sqrt(new_err_col**2 + new_err_row**2))
    print(f"before rectify project error: {project_err}, after rectify project error: {new_project_err}")

    ## remap image
    height, width = image.shape[:2]
    x, y = np.meshgrid(np.linspace(0, width-1, width), 
                       np.linspace(0, height-1, height))
    x, y = x.reshape(-1), y.reshape(-1)
    b = np.ones((x.shape[0]), dtype=float)
    image_rectified = (H @ np.vstack([b, x, y, x*y, x*x, y*y])).T
    new_x = image_rectified[:,0].reshape(height, width).astype(np.float32)
    new_y = image_rectified[:,1].reshape(height, width).astype(np.float32)
    new_image = cv2.remap(image, new_x, new_y, cv2.INTER_LINEAR)
    return K_4by4, W2C, new_image



if __name__  == '__main__':
    pass
