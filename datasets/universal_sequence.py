from implicitron.tools.configurable import configurable
import numpy as np
from .base_dataset import Dataset as base_dataset
import os
import torch


class Dataset(base_dataset):
    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "--cache", action="store_true", help="cache the data into ram"
        )
        parser.add_argument(
            "--subsample", action="store_true", help="subsample the video in time"
        )
        parser.add_argument(
            "--data_dir", default=".", type=str, help="the preprocessed data to load"
        )
        parser.add_argument(
            "--overfit", action="store_true", help="overfit and see if things works"
        )
        parser.add_argument(
            "--gaps",
            type=lambda y: [int(x) for x in y.split(",")],
            default=[1, 2, 3, 4],
            help="gaps for sequences",
        )
        parser.add_argument(
            "--repeat", type=int, default=1, help="number of repetitions"
        )
        parser.add_argument("--select", action="store_true", help="pred")
        return parser, set()

    def __init__(self, opt, mode="train", model=None):
        super().__init__(opt, mode, model)
        self.mode = mode
        self.data_dir = opt.data_dir
        assert mode in ("train", "vali")

        self.gaps = opt.gaps
        self.fids = np.load(
            os.path.join(self.data_dir, "index.npz"),
        )["fids"]

        if mode == "train":
            # pairs of frames with flow
            self.file_list = [
                f"batch_{gap:02d}_{fid:05d}.pt"
                for gap in self.gaps
                for fid in self.fids[:-gap]
            ]
        else:
            # just individual frames
            self.file_list = [f"frame_{fid:05d}.npz" for fid in self.fids]

    def __len__(self):
        if self.mode != "train":
            return len(self.file_list)
        else:
            return len(self.file_list) * self.opt.repeat

    def __getitem__(self, idx):

        # Make sure iterating over the dataset actually stops
        if idx > len(self):
            raise IndexError()

        if self.opt.overfit:
            idx = idx % self.opt.capat  # ??
        else:
            idx = idx % len(self.file_list)

        N = len(self.fids)
        H, W = None, None

        def get_time(fid):
            return (self.fids == fid).nonzero()[0].item()

        def bcast(t):
            return torch.tensor([t / N]).reshape((-1, 1, 1, 1)).expand(1, 1, H, W)

        if self.mode == "train":
            sample = torch.load(
                os.path.join(self.data_dir, "batch", self.file_list[idx])
            )
            H, W = sample["depth_pred_1"].shape[-2:]

            fid1 = sample["fid_1"].item()
            fid2 = sample["fid_2"].item()
            t1 = get_time(fid1)
            t2 = get_time(fid2)
            sample["time_step"] = 1 / N
            sample["time_stamp_1"] = bcast(t1)
            sample["time_stamp_2"] = bcast(t2)
            sample["frame_id_1"] = sample["fid_1"]
            sample["frame_id_2"] = sample["fid_2"]

        else:
            sample_np = np.load(
                os.path.join(self.data_dir, "depth", self.file_list[idx])
            )
            H, W = sample_np["depth_pred"].shape[-2:]

            c2w = np.eye(4)
            c2w[:3, :3] = sample_np["R"]
            c2w[:3, 3] = sample_np["T"]
            fid = sample_np["fid"].item()
            t = get_time(fid)
            sample = {
                "time_stamp_1": bcast(t).squeeze(0),
                "img": sample_np["image_rgb"],
                "frame_id_1": np.asarray(fid),
                "time_step": np.asarray(1 / N),
                "depth_pred": sample_np["depth_pred"],#[None, ...],
                "depth_mvs": sample_np["depth_gt"],#[None, ...],  # TODO: gt
                "K": sample_np["K"][None, None, ...],
                "K_inv": np.linalg.inv(sample_np["K"])[None, None, ...],
                "R_1": sample_np["R"][None, None, ...],
                "R_1_T": sample_np["R"].T[None, None, ...],
                "t_1": sample_np["T"][None, None, None, ...],
                "cam_c2w": c2w,
            }

        self.convert_to_float32(sample)
        # for k in sample.keys():
        #     print(k, sample[k].shape, "*")
        sample["pair_path"] = self.file_list[idx]
        return sample

        # N = len(self) + 1
        # num_frames = len(self.image_rgb)
        # fid1 = idx
        # fid2 = idx + 1
        # H, W = self.image_rgb.shape[2:]
        # sample_loaded = {
        #     "img_1": self.image_rgb[idx : idx + 1][None, :],
        #     "img_2": self.image_rgb[idx + 1][None, :],
        #     "depth_pred_1": self.depth_pred[idx][None, :],
        #     "depth_pred_2": self.depth_pred[idx + 1][None, :],
        #     "flow_1_2": self.flow_1_2[idx][None, :],
        #     "flow_2_1": self.flow_2_1[idx + 1][None, :],
        #     "mask_1": self.mask[idx][None, :],
        #     "mask_2": self.mask[idx + 1][None, :],
        #     "R_1": np.eye(3),
        #     "R_1_T": np.eye(3),
        #     "t_1": np.zeros(3),
        #     "R_2": np.eye(3),
        #     "R_2_T": np.eye(3),
        #     "t_2": np.zeros(3),
        #     "K": np.eye(3),
        #     "K_inv": np.eye(3),
        #     "time_step": 1 / num_frames,
        #     "time_stamp_1": np.broadcast_to(
        #         np.asarray(fid1).reshape((-1, 1, 1, 1)), (1, 1, H, W)
        #     )
        #     / num_frames,
        #     "time_stamp_2": np.broadcast_to(
        #         np.asarray(fid2).reshape((-1, 1, 1, 1)), (1, 1, H, W)
        #     )
        #     / num_frames,
        #     "frame_id_1": np.asarray(fid1),
        #     "frame_id_2": np.asarray(fid2),
        #     "pair_path": f"{fid1}_{fid2}",
        # }
        # self.convert_to_float32(sample_loaded)
        # return sample_loaded
