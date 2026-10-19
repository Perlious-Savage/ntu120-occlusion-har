"""Single-modality datasets for the Milestone 2 baselines (clean data, no corruption)."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.ctrgcn.feeder_tools import random_rot, valid_crop_resize

from .rgb_loader import read_video_frames, to_clip_tensor
from .temporal import uniform_temporal_indices


class RGBClips(torch.utils.data.Dataset):
    """R3D-18 input: (3, T, S, S) in [0, 1] (the model normalises).
    Train: one random frame inside each of T equal segments + random S x S crop.
    Eval: deterministic uniform frames (same helper as the multimodal loader) + center crop."""

    def __init__(self, split_csv, data_root='data', train=False, num_frames=16, size=112, resize_hw=(128, 171)):
        self.df = pd.read_csv(split_csv)
        self.root, self.train = Path(data_root), train
        self.num_frames, self.size, self.resize_hw = num_frames, size, tuple(resize_hw)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        frames = read_video_frames(self.root / row.rgb_path, self.resize_hw)
        n = len(frames)
        if self.train:
            edges = np.linspace(0, n, self.num_frames + 1)
            idx = np.minimum(np.floor(edges[:-1] + np.random.rand(self.num_frames) * np.diff(edges)), n - 1).astype(int)
            top = np.random.randint(0, frames.shape[1] - self.size + 1)
            left = np.random.randint(0, frames.shape[2] - self.size + 1)
            clip = frames[idx, top:top + self.size, left:left + self.size]
        else:
            clip = frames[uniform_temporal_indices(n, self.num_frames)]
        return to_clip_tensor(clip, self.size), int(row.label), row.sample_id


class SkeletonClips(torch.utils.data.Dataset):
    """CTR-GCN input: (3, window, 25, M) from the packed float16 cache (scripts/check_skeletons.py).

    Preprocessing as in the official CTR-GCN pipeline: coordinates are translated so that joint 2
    (spine middle) of the first tracked body in its first tracked frame is the origin; then the
    official feeder functions crop a random 50-100% of the valid frames (train) or the central
    95% (eval), resize to `window` frames, and apply a random rotation (train only)."""

    def __init__(self, split_csv, data_root='data', train=False, window=64, p_interval=None, rotate=None):
        self.df = pd.read_csv(split_csv)
        self.root, self.train, self.window = Path(data_root), train, window
        self.p_interval = p_interval or ([0.5, 1] if train else [0.95])
        self.rotate = train if rotate is None else rotate
        index = pd.read_csv(self.root / 'processed/skeleton_index.csv').set_index('sample_id')
        self.index = index.loc[self.df.sample_id]  # raises if a split sample is missing from the cache
        self.max_bodies = int(index.max_bodies.iloc[0])
        self.cache = None  # opened lazily so every DataLoader worker gets its own memmap

    def __len__(self):
        return len(self.df)

    def load(self, i):
        if self.cache is None:
            self.cache = np.memmap(self.root / 'processed/skeleton_xyz_f16.bin', dtype=np.float16, mode='r')
        r = self.index.iloc[i]
        n = 3 * r.num_frames * 25 * self.max_bodies
        x = np.asarray(self.cache[r.offset:r.offset + n], dtype=np.float32).reshape(3, r.num_frames, 25, self.max_bodies)
        tracked = x.any(axis=0, keepdims=True)                    # (1, T, 25, M) joint present
        first = np.flatnonzero(tracked[0, :, :, 0].any(axis=1))   # frames where body 0 is tracked
        if len(first):
            x = np.where(tracked, x - x[:, first[0], 1, 0][:, None, None, None], 0).astype(np.float32)
        valid = x.any(axis=(0, 2, 3))
        return x[:, valid]  # drop frames without anybody, as the official preprocessing does

    def __getitem__(self, i):
        x = self.load(i)
        x = valid_crop_resize(x, x.shape[1], self.p_interval, self.window)
        if self.rotate:
            x = random_rot(x)
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        return x, int(self.df.label.iloc[i]), self.df.sample_id.iloc[i]


def build_dataset(cfg, split_csv, train):
    inp = cfg['input']
    if cfg['model']['name'] == 'r3d18':
        return RGBClips(split_csv, cfg['data_root'], train, inp['num_frames'], inp['size'], inp['resize_hw'])
    return SkeletonClips(split_csv, cfg['data_root'], train, inp['window'])
