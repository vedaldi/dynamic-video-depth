import os
import torch.multiprocessing as mp
from .scripts.preprocess.universal import preprocess
from . import train
import numpy as np
from skimage.transform import resize as imresize

gaps = [1, 2, 4, 6, 8]


def load(out_dir, dry_run=False):
    depth_dir = os.path.join(
        out_dir,
        "testscene_flow_motion_field_universal_sequence_default",
        "epoch0020_test",
    )

    index_path = os.path.join(out_dir, "preproc", "index.npz")
    if dry_run and not os.path.exists(index_path):
        return None
    index = np.load(index_path)
    fids = index["fids"]
    H = index["height"]
    W = index["width"]
    scale = index["scale"]

    def resize(x):
        x = np.transpose(x, (1, 2, 0))
        x = imresize(x, (H, W), preserve_range=True).astype(np.float32)
        return np.transpose(x, (2, 0, 1)) / scale

    depth = []
    for i, fid in enumerate(fids):
        batch_path = os.path.join(depth_dir, f"batch{i:04d}.npz")
        if not os.path.exists(batch_path):
            return None
        if not dry_run:
            batch = np.load(batch_path)
            x = batch["depth"].squeeze(0)
            x = resize(x)
            depth.append(x)
        else:
            depth.append(True)

    return np.concatenate(depth)


def run(dataloader, out_dir, resume=False):
    preproc_dir = os.path.join(out_dir, "preproc")
    checkpoint_dir = os.path.join(out_dir, "checkpoints", "0")
    test_script_path = os.path.join(
        os.path.dirname(__file__), "experiments/universal/test_cmd.txt"
    )

    preprocess(dataloader, preproc_dir, gaps=gaps, resume=resume)

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
        # "--use_motion_seg",  # av
        "--full_logdir", checkpoint_dir,
        "--test_template", test_script_path,
        "--resume", "-1" if resume else "0",  # resume from the last epoch
        "--force_overwrite",  #
        # "--pt_no_overwrite", # when resuming, do not overwrite the new options with the saved ones
    ]
    # fmt: on

    # mp.set_start_method('spawn', force=True)
    train.main(args=args)
