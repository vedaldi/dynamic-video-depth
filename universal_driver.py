import os
import torch.multiprocessing as mp
from .scripts.preprocess.universal import preprocess
from . import train
import numpy as np
from skimage.transform import resize as imresize

gaps = [1, 2, 4, 6, 8]


def load_index(out_dir):
    index_path = os.path.join(out_dir, "preproc", "index.npz")
    return np.load(index_path)


def load_frame(out_dir, fid, index=None, dry_run=False):
    if index is None:
        index = load_index(out_dir)

    depth_dir = os.path.join(
        out_dir,
        "testscene_flow_motion_field_universal_sequence_default",
        "epoch0020_test",
    )

    H = index["height"]
    W = index["width"]
    scale = index["scale"]

    i = np.nonzero(index["fids"] == fid)[0].item()
    batch_path = os.path.join(depth_dir, f"batch{i:04d}.npz")
    if not dry_run:
        batch = np.load(batch_path)
        x = batch["depth"].squeeze(0)
        x = np.transpose(x, (1, 2, 0))
        x = imresize(x, (H, W), preserve_range=True).astype(np.float32)
        x = np.transpose(x, (2, 0, 1)) / scale
    else:
        if not os.path.exists(batch_path):
            raise FileNotFoundError()
        x = True

    return x


def load(out_dir, dry_run=False):
    try:
        depth = []
        index = load_index(out_dir)
        for i, fid in enumerate(index["fids"]):
            depth.append(load_frame(out_dir, fid, index=index, dry_run=dry_run))
        if not dry_run:
            depth = np.concatenate(depth)
        return {"depth": depth, "fids": index["fids"]}

    except FileNotFoundError:
        if dry_run:
            return None
        raise


def run(
    dataloader,
    out_dir,
    resume=False,
    rescale_depth_using_masked_region=False,
    use_motion_mask=False,
):
    preproc_dir = os.path.join(out_dir, "preproc")
    checkpoint_dir = os.path.join(out_dir, "checkpoints", "0")
    test_script_path = os.path.join(
        os.path.dirname(__file__), "experiments/universal/test_cmd.txt"
    )

    preprocess(
        dataloader,
        preproc_dir,
        gaps=gaps,
        rescale_depth_using_masked_region=rescale_depth_using_masked_region,
        resume=resume,
    )

    # fmt: off
    args = [
        "--net", "scene_flow_motion_field",
        "--dataset", "universal_sequence",
        "--data_dir", preproc_dir,
        "--log_time",
        "--epoch_batches", "2000",
        "--epoch", "20",
        "--lr", "1e-6",
        "--html_logger",
        "--vali_batches", "150",
        "--batch_size", "1",
        "--optim", "adam",
        "--vis_batches_vali", "4",
        "--vis_every_vali", "1",
        "--vis_every_train", "1",
        "--vis_batches_train", "5",
        "--vis_at_start",
        "--tensorboard",
        "--gpu", "0",
        "--save_net", "1",
        "--workers", "4",
        "--one_way",
        "--loss_type", "l1",
        "--l1_mul", "0",
        "--acc_mul", "1",
        "--disp_mul", "1",
        "--warm_sf", "5",
        "--scene_lr_mul", "1000",
        "--repeat", "1",
        "--flow_mul", "1",
        "--sf_mag_div", "100",
        "--time_dependent",
        "--gaps", ','.join([str(gap) for gap in gaps]),
        "--midas",
        "--use_disp",
        "--full_logdir", checkpoint_dir,
        "--test_template", test_script_path,
        "--resume", "-1" if resume else "0",  # resume from the last epoch
        "--force_overwrite",
    ]
    # fmt: on

    if use_motion_mask:
        args.append("--use_motion_seg")

    # mp.set_start_method('spawn', force=True)
    train.main(args=args)
