import argparse
import os
import cv2
import dvd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from dvd.configs import midas_pretrain_path
from dvd.third_party.MiDaS import MidasNet
from dvd.third_party.RAFT.core.raft import RAFT
from skimage.transform import resize as imresize

# Gaps between frames for flow computation
default_gaps = [1, 2, 3, 4, 5, 6, 7, 8]

H_midas = 192
W_midas = 384

# --------------------------------------------------------------------
# Preprocess depth
# --------------------------------------------------------------------
#


def _load_depth(out_dir, fid):
    return np.load(os.path.join(out_dir, "depth", f"frame_{fid:05d}.npz"))


def _pytorch_camera_to_dvd(camera, s, W, H):
    """
    Convert PyTorch3D cameras to DVD cameras.

    See `README.md` for details.

    - `s`: the PyTorch camera aspect ration (W_torch/H_torch)
    - `W`: the DVD image width
    - `H`: the DVD image height

    Arguments:
    - camera: PyTorch3D cameras
    - image_size: [width, height]
    - scale: multiply camera translation by this number
    - normalize_first_frame: make the first camera [I,0]

    Returns:
        K, R, T

    """
    N = len(camera)
    L = np.diag([-1, -1, 1])
    R__ = np.zeros((N, 3, 3))
    T__ = np.zeros((N, 3))
    K__ = np.zeros((N, 3, 3))

    for t in range(N):
        # PyTorch camera intrinsics
        f = camera[t].focal_length.cpu().numpy()
        assert s >= 1
        assert f[0] == f[1], "Non-square pixels detected, must be a bug"

        f = f[0]

        c = camera[t].principal_point.cpu().numpy()
        K = np.array(
            [
                [f, 0, 0],
                [0, f, 0],
                [c[0], c[1], 1],
            ]
        )
        # Translate PyTorch NDC to COLMAP image coords
        A = np.array(
            [
                [-W / (2 * s), 0, 0],
                [0, -H / 2, 0],
                [(W - 1) / 2, (H - 1) / 2, 1],
            ]
        )
        K__[t] = L @ K @ A

        # Extrinsics conversion
        R = camera[t].R.cpu().numpy()
        T = camera[t].T.cpu().numpy()
        R__[t] = L @ R.T
        T__[t] = -T @ R.T

    return K__, R__, T__


def _process_depth(dataloader, out_dir, rescale_depth_using_masked_region, resume):
    depth_dir = os.path.join(out_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)
    model = None

    fids = []
    image_rgb = []
    depth_gt = []
    depth_pred = []
    mask = []
    K = []
    T = []
    R = []

    print("Generating depth")
    index_path = os.path.join(out_dir, "index.npz")
    resume=False
    if resume and os.path.exists(index_path):
        return

    for batch in tqdm(dataloader):
        fids.extend(batch["frame_number"].tolist())

        H, W = batch["image_rgb"].shape[2:]
        # H_midas = 256
        # step = 32
        # W_midas = (((round(W * (H_midas / H)) - 1) // step) + 1) * step

        # Target size for processing by DVD
        # Note that this resamples pixels so that they are not square anymore
        max_W = 384
        multiple = 64
        if W > max_W:
            sc = max_W / W
            W_dvd = max_W
        else:
            W_dvd = W
        H_dvd = int(np.round((H * sc) / multiple) * multiple)

        if model is None:
            model = MidasNet(
                midas_pretrain_path,
                non_negative=True,
                resize=(H_midas, W_midas),
                normalize_input=True,
            )
            model.eval()

        with torch.no_grad():
            depth_pred.append(model(batch["image_rgb"]).detach().cpu().numpy())

        image_rgb.append(batch["image_rgb"].detach().cpu().numpy())
        depth_gt.append(batch["depth_map"].detach().cpu().numpy())
        mask.append(batch["depth_mask"].detach().cpu().numpy())

        # Convert cameras
        cam = _pytorch_camera_to_dvd(batch["camera"], W / H, W=W_dvd, H=H_dvd)
        K.append(cam[0])
        R.append(cam[1])
        T.append(cam[2])

    image_rgb = np.concatenate(image_rgb)
    depth_gt = np.concatenate(depth_gt)
    depth_pred = np.concatenate(depth_pred)
    mask = np.concatenate(mask)

    K = np.concatenate(K)
    R = np.concatenate(R)
    T = np.concatenate(T)

    print("Rescaling depth")
    scales = []
    for i in tqdm(range(len(depth_pred))):
        if rescale_depth_using_masked_region:
            m = (mask[i].ravel() > 0)
        else:
            m = slice(None, None)
        this_scale = np.median(depth_pred[i].ravel()[m] / depth_gt[i].ravel()[m])
        scales.append(this_scale)
    scale = np.mean(scales)

    assert not np.isnan(scale)

    def resize(x):
        x = np.transpose(x, (1, 2, 0))
        x = imresize(x, (H_dvd, W_dvd), preserve_range=True).astype(np.float32)
        return np.transpose(x, (2, 0, 1))

    # Save individual files
    print("Saving depth")
    for t in tqdm(range(len(image_rgb))):
        fid = fids[t]
        path = os.path.join(depth_dir, f"frame_{fid:05d}.npz")
        np.savez(
            path,
            fid=fid,
            image_rgb=resize(image_rgb[t]),
            depth_pred=resize(depth_pred[t]),
            depth_gt=resize(depth_gt[t] * scale),
            mask=resize(mask[t]),
            K=K[t],
            R=R[t],
            T=T[t] * scale,
        )

    print(f"Saving list of frames to {out_dir}")
    np.savez(index_path, fids=fids, scale=scale, height=H, width=W)


# --------------------------------------------------------------------
# Preprocess flow
# --------------------------------------------------------------------


def _load_flow(out_dir, fid1, fid2):
    return np.load(os.path.join(out_dir, "flow", f"flow_{fid1:05d}_{fid2:05d}.npz"))


def _resize_flow(flow, size):
    resized_width, resized_height = size
    H, W = flow.shape[:2]
    scale = np.array((resized_width / float(W), resized_height / float(H))).reshape(
        1, 1, -1
    )
    resized = cv2.resize(
        flow, dsize=(resized_width, resized_height), interpolation=cv2.INTER_CUBIC
    )
    resized *= scale
    return resized


def _get_oob_mask(flow_1_2):
    H, W, _ = flow_1_2.shape
    hh, ww = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float())
    coord = torch.zeros([H, W, 2])
    coord[..., 0] = ww
    coord[..., 1] = hh
    target_range = coord + flow_1_2
    m1 = (target_range[..., 0] < 0) + (target_range[..., 0] > W - 1)
    m2 = (target_range[..., 1] < 0) + (target_range[..., 1] > H - 1)
    return (m1 + m2).float().numpy()


def _backward_flow_warp(im2, flow_1_2):
    H, W, _ = im2.shape
    hh, ww = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float())
    coord = torch.zeros([1, H, W, 2])
    coord[0, ..., 0] = ww
    coord[0, ..., 1] = hh
    sample_grids = coord + flow_1_2[None, ...]
    sample_grids[..., 0] /= (W - 1) / 2
    sample_grids[..., 1] /= (H - 1) / 2
    sample_grids -= 1
    im = torch.from_numpy(im2).float().permute(2, 0, 1)[None, ...]
    out = F.grid_sample(im, sample_grids, align_corners=True)
    o = out[0, ...].permute(1, 2, 0).numpy()
    return o


def _load_RAFT():
    default_raft_path = os.path.join(
        os.path.dirname(dvd.third_party.RAFT.core.raft.__file__),
        "..",
        "models",
        "raft-sintel.pth",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="restore checkpoint")
    parser.add_argument("--path", help="dataset for evaluation")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    parser.add_argument(
        "--alternate_corr",
        action="store_true",
        help="use efficient correlation implementation",
    )
    args = parser.parse_args(["--model", default_raft_path, "--path", "./"])
    net = torch.nn.DataParallel(RAFT(args).cuda())
    net.load_state_dict(torch.load(args.model))
    return net


def _get_flow(net, out_dir, fid1, fid2):
    sample1 = _load_depth(out_dir, fid1)
    sample2 = _load_depth(out_dir, fid2)

    image1_rgb = torch.from_numpy(sample1["image_rgb"])
    image2_rgb = torch.from_numpy(sample2["image_rgb"])

    H, W = image1_rgb.shape[1:]

    def resize(x):
        x = np.transpose(x.detach().cpu().numpy(), (1, 2, 0))
        x = imresize(x, [288, 512], anti_aliasing=True) * 255
        x = np.transpose(x, (2, 0, 1))
        return torch.FloatTensor(x)  # .to(device)

    resized1 = resize(image1_rgb)
    resized2 = resize(image2_rgb)

    def raft(image1, image2):
        with torch.no_grad():
            flow_low, flow_up = net(
                image1=image1.unsqueeze(0),
                image2=image2.unsqueeze(0),
                iters=20,
                test_mode=True,
            )
            flow = flow_up.squeeze().permute(1, 2, 0).cpu().numpy()
            return _resize_flow(flow, [W, H])

    flow_1_2 = raft(resized1, resized2)
    flow_2_1 = raft(resized2, resized1)

    warp_flow_1_2 = _backward_flow_warp(
        flow_1_2, flow_2_1
    )  # using latter to sample former
    err_1 = np.linalg.norm(warp_flow_1_2 + flow_2_1, axis=-1)
    mask_1 = np.where(err_1 > 1, 1, 0)
    oob_mask_1 = _get_oob_mask(flow_2_1)
    mask_1 = np.clip(mask_1 + oob_mask_1, a_min=0, a_max=1)
    warp_flow_2_1 = _backward_flow_warp(flow_2_1, flow_1_2)
    err_2 = np.linalg.norm(warp_flow_2_1 + flow_1_2, axis=-1)
    mask_2 = np.where(err_2 > 1, 1, 0)
    oob_mask_2 = _get_oob_mask(flow_1_2)
    mask_2 = np.clip(mask_2 + oob_mask_2, a_min=0, a_max=1)
    return {
        "flow_1_2": flow_1_2.astype(np.float32),
        "flow_2_1": flow_2_1.astype(np.float32),
        "mask_1": (1 - mask_1).astype(np.uint8),
        "mask_2": (1 - mask_2).astype(np.uint8),
        "frame_id_1": fid1,
        "frame_id_2": fid2,
    }


def _process_flow(out_dir, gaps, resume):
    flow_dir = os.path.join(out_dir, "flow")
    os.makedirs(flow_dir, exist_ok=True)

    fids = np.load(
        os.path.join(out_dir, "index.npz"),
    )["fids"]

    net = _load_RAFT()

    for gap in gaps:
        print(f"Generating flows for gap {gap}")
        N = len(fids)
        for k in tqdm(range(N - gap)):
            fid1 = fids[k]
            fid2 = fids[k + gap]
            path = os.path.join(flow_dir, f"flow_{fid1:05d}_{fid2:05d}.npz")
            if resume and os.path.exists(path):
                continue
            data = _get_flow(net, out_dir, fid1, fid2)
            np.savez(
                path,
                **data,
            )


# --------------------------------------------------------------------
# Preprocess batches
# --------------------------------------------------------------------


def _make_pair(fid, gap, out_dir):
    fid1 = fid
    fid2 = fid1 + gap
    depth1 = _load_depth(out_dir, fid1)
    depth2 = _load_depth(out_dir, fid2)
    flow = _load_flow(out_dir, fid1, fid2)

    def tt(x):
        x = torch.tensor(x)
        if x.dtype == torch.float64:
            x = x.float()
        return x

    sample = {
        "R_1": tt(depth1["R"][None, None, None, ...]),
        "R_1_T": tt(depth1["R"].T[None, None, None, ...]),
        "t_1": tt(depth1["T"][None, None, None, None, ...]),
        #
        "R_2": tt(depth2["R"][None, None, None, ...]),
        "R_2_T": tt(depth2["R"].T[None, None, None, ...]),
        "t_2": tt(depth2["T"][None, None, None, None, ...]),
        #
        "K": tt(depth1["K"][None, None, None, ...]),
        "K_inv": tt(np.linalg.inv(depth1["K"])[None, None, None, ...]),
        "flow_1_2": tt(flow["flow_1_2"][None, ...]),
        "flow_2_1": tt(flow["flow_2_1"][None, ...]),
        "mask_1": tt(flow["mask_1"][None, ..., None, None]),
        "mask_2": tt(flow["mask_2"][None, ..., None, None]),
        "motion_seg_1": tt(depth1["mask"][..., None, None]),
        "depth_1": tt(depth1["depth_gt"][None, ...]),
        "depth_pred_1": tt(depth1["depth_pred"][None, ...]),
        "fid_1": tt([fid1]),
        "fid_2": tt([fid2]),
        "img_1": tt(depth1["image_rgb"][None, ...]),
        "img_2": tt(depth2["image_rgb"][None, ...]),
    }
    return sample


def _collate(samples):
    batch = {}
    for k in samples[0].keys():
        batch[k] = torch.cat([s[k] for s in samples])
    return batch


def _process_batches(out_dir, gaps, resume):
    fids = np.load(
        os.path.join(out_dir, "index.npz"),
    )["fids"]

    batch_dir = os.path.join(out_dir, "batch")
    os.makedirs(batch_dir, exist_ok=True)
    N = len(fids)

    batch_size = 1

    for i, gap in enumerate(gaps):
        print(f"Generating batches for gap {i} of {len(gaps)} (gap {gap})")
        for t in tqdm(range(N - batch_size - gap + 1)):
            fid = fids[t]
            path = os.path.join(batch_dir, f"batch_{gap:02d}_{fid:05d}.pt")
            if resume and os.path.exists(path):
                continue
            seq = range(fid, fid + batch_size)
            samples = [_make_pair(fid, gap, out_dir) for fid in seq]
            batch = _collate(samples)
            torch.save(batch, path)


# --------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------


def preprocess(
    dataloader,
    out_dir,
    gaps=default_gaps,
    rescale_depth_using_masked_region=False,
    resume=False,
):
    """
    `dataloader` must provide frame dicts with fields:

    - `image_rgb`: (N, 3, H, W) input image
    - `depth_pred` (N, 1, H, W) predicted depth map
    - `camera`: PyTorch3D camera
    - ``:

    """
    _process_depth(
        dataloader,
        out_dir,
        rescale_depth_using_masked_region=rescale_depth_using_masked_region,
        resume=resume,
    )
    _process_flow(out_dir, gaps=gaps, resume=resume)
    _process_batches(out_dir, gaps=gaps, resume=resume)
    return None
