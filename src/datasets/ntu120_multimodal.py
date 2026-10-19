"""Synchronized RGB + skeleton dataset for NTU RGB+D 120."""
from pathlib import Path

import pandas as pd
import torch

from .rgb_loader import read_video_frames, to_clip_tensor
from .skeleton_loader import load_skeleton
from .temporal import uniform_temporal_indices


class NTU120Multimodal(torch.utils.data.Dataset):
    """Each item: rgb (3, T, S, S), skeleton (3, T, 25, M), label 0-119, IDs, and the frame indices
    used in each modality (same normalised positions, see uniform_temporal_indices)."""

    def __init__(self, split_csv, data_root='data', num_frames=16, rgb_size=112,
                 rgb_resize_hw=(128, 171), max_bodies=2, rgb_dir=None):
        self.df = pd.read_csv(split_csv)
        self.root = Path(data_root)
        if rgb_dir:  # shrunk copies (same sample ID, same frame count) instead of raw 1080p
            self.df['rgb_path'] = [f'{rgb_dir}/{s}_rgb.mp4' for s in self.df.sample_id]
        self.num_frames, self.rgb_size = num_frames, rgb_size
        self.rgb_resize_hw, self.max_bodies = tuple(rgb_resize_hw), max_bodies

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        frames = read_video_frames(self.root / row.rgb_path, self.rgb_resize_hw)
        skel = load_skeleton(self.root / row.skeleton_path, self.max_bodies)
        rgb_idx = uniform_temporal_indices(len(frames), self.num_frames)
        sk_idx = uniform_temporal_indices(skel['num_frames'], self.num_frames)
        return {
            'rgb': to_clip_tensor(frames[rgb_idx], self.rgb_size),
            'skeleton': torch.from_numpy(skel['xyz'][:, sk_idx]),
            'label': int(row.label),
            'sample_id': row.sample_id,
            'setup_id': int(row.setup_id),
            'subject_id': int(row.subject_id),
            'action_id': int(row.action_id),
            'rgb_frame_idx': torch.from_numpy(rgb_idx),
            'skeleton_frame_idx': torch.from_numpy(sk_idx),
            'rgb_num_frames': len(frames),
            'skeleton_num_frames': skel['num_frames'],
        }
