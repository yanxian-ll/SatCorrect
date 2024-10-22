import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from flow_utils.raft.utils.flow_viz import flow_to_image
from flow_utils.raft.raft import RAFT


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    plt.imshow(img_flo / 255.0)
    plt.show()


def plot_matches(left, right, flow, num_matches=50):
    left = left[0].permute(1,2,0).cpu().numpy()
    right = right[0].permute(1,2,0).cpu().numpy()
    flow = flow[0].permute(1,2,0).cpu().numpy()

    H, W = flow.shape[:2]

    # plot matches
    u0 = np.random.randint(0, W, num_matches)
    v0 = np.random.randint(0, H, num_matches)

    img = np.concatenate((left, right), axis=1) / 255.0
    # img = np.concatenate((left, right), axis=0) / 255.0

    plt.imshow(img)
    cmap = plt.get_cmap('jet')
    for i in range(num_matches):
        x0 = u0[i]
        y0 = v0[i]
        x1 = x0 + flow[y0, x0, 0]
        y1 = y0 + flow[y0, x0, 1]
        plt.plot([x0, x1 + W], [y0, y1], '-+', color=cmap(i / (num_matches - 1)), scalex=False, scaley=False)
        # pl.plot([x0, x1], [y0, y1 + H], '-+', color=cmap(i / (num_matches - 1)), scalex=False, scaley=False)
    plt.show()


def plot_error(gt, pred):
    pred = pred[0].permute(1,2,0).cpu().numpy()
    error = np.abs(gt - pred)
    u = error[:,:,0]
    v = error[:,:,1]

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(u, cmap='hot')
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.imshow(v,  cmap='hot')
    plt.colorbar()

    plt.tight_layout()
    plt.show()


def readFlow(fn):
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            return np.resize(data, (int(h), int(w), 2))

TAG_CHAR = np.array([202021.25], np.float32)
def writeFlow(filename, uv):
    nBands = 2
    u = uv[:, :, 0]
    v = uv[:, :, 1]
    assert (u.shape == v.shape)
    height, width = u.shape
    with open(filename, 'wb') as f:
        # write the header
        f.write(TAG_CHAR)
        # TAG_CHAR.tofile(f)
        np.array(width).astype(np.int32).tofile(f)
        np.array(height).astype(np.int32).tofile(f)
        # arrange into matrix form
        tmp = np.zeros((height, width * nBands))
        tmp[:, np.arange(width) * 2] = u
        tmp[:, np.arange(width) * 2 + 1] = v
        tmp.astype(np.float32).tofile(f)


def setup_raft(ckpt_model, small=False, mixed_precision=False, alternate_corr=False, device='cpu'):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.small = small
    args.mixed_precision = mixed_precision
    args.alternate_corr = alternate_corr

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(ckpt_model, weights_only=True))
    
    model = model.module
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def compute_optical_flow(img1, img2, model='flow_utils/checkpoints', device='cpu'):
    """
    img numpy.array (H,W,3)
    return flow numpy.array (H,W)
    """
    org_h1, org_w1 = img1.shape[:2]
    org_h2, org_w2 = img2.shape[:2]

    # pad
    if img1.shape[0] < img2.shape[0]:
        img1 = np.pad(img1, ((0, img2.shape[0]-img1.shape[0]), (0, 0), (0, 0)))
    if img1.shape[1] < img2.shape[1]:
        img1 = np.pad(img1, ((0, 0), (0, img2.shape[1]-img1.shape[1]), (0, 0)))
    img2 = np.pad(img2, ((0, img1.shape[0]-img2.shape[0]), (0, img1.shape[1]-img2.shape[1]), (0, 0)))
    
    h, w = img1.shape[:2]
    pad_h = (h//8 + 1)*8 - h
    pad_w = (w//8 + 1)*8 - w
    img1 = np.pad(img1, ((0,pad_h),(0,pad_w),(0,0)))
    img2 = np.pad(img2, ((0,pad_h),(0,pad_w),(0,0)))
    assert img1.shape[0]%8==0 and img1.shape[1]%8==0
    assert img2.shape[0]%8==0 and img2.shape[1]%8==0

    # to tensor (1,3,H,W)
    img1 = torch.from_numpy(img1).permute(2, 0, 1).float()[None].to(device)
    img2 = torch.from_numpy(img2).permute(2, 0, 1).float()[None].to(device)

    _, forward_flow = model(img1, img2, iters=20, test_mode=True)
    _, backward_flow = model(img2, img1, iters=20, test_mode=True)

    # viz(img1, forward_flow)
    # viz(img2, backward_flow)
    # plot_matches(img1, img2, forward_flow)
    # plot_matches(img2, img1, backward_flow)

    # crop flow
    forward_flow = forward_flow[0].permute(1,2,0).detach().cpu().numpy()[:org_h1, :org_w1, :]
    backward_flow = backward_flow[0].permute(1,2,0).detach().cpu().numpy()[:org_h2, :org_w2, :]
    return forward_flow, backward_flow
