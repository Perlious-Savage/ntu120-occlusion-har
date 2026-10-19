"""Visual RGB-skeleton synchronization check on N diverse samples (Milestone 1, step 12).

For each sample the dataset class picks synchronized RGB / skeleton frame indices. Four of those
positions are shown: top row = RGB frame with the skeleton's 2D colour-camera projection drawn on
it, bottom row = the 3D skeleton (x, y). If the overlay sits on the person, the modalities agree.

Uses the same RGB files as training (config rgb_dir), so it checks exactly what the models see.
Usage: python scripts/visualize_sync.py --split data/splits/xsub_train.csv [--n 20] [--out outputs/sync_check]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.ntu120_multimodal import NTU120Multimodal  # noqa: E402
from src.datasets.rgb_loader import read_video_frames  # noqa: E402
from src.datasets.skeleton_loader import load_skeleton  # noqa: E402
from src.utils.ntu import NTU_BONES  # noqa: E402

DISPLAY_HW = (256, 456)  # native size of the shrunk videos (raw 1080p videos are downscaled to it)
SHOW = [0, 5, 10, 15]    # which of the 16 synchronized positions to display
BODY_COLORS = ['lime', 'magenta']


def pick_diverse(df, n, seed):
    """Greedy pick: each sample must add a new action and, preferably, a new subject/setup/camera."""
    pool = df.sample(min(len(df), 3000), random_state=seed)
    seen = {k: set() for k in ['action_id', 'subject_id', 'setup_id', 'camera_id']}
    chosen = []
    while len(chosen) < n and len(pool):
        novelty = sum(~pool[k].isin(seen[k]) * (4 if k == 'action_id' else 1) for k in seen)
        best = novelty.idxmax()
        chosen.append(best)
        for k in seen:
            seen[k].add(pool.at[best, k])
        pool = pool.drop(best)
    return chosen


def draw_skeleton(ax, pts, color):
    """pts: (2, 25) for one body; all-zero joints (untracked) are skipped."""
    if not pts.any():
        return
    for a, b in NTU_BONES:
        ax.plot(pts[0, [a, b]], pts[1, [a, b]], color=color, lw=1.5)
    ax.scatter(pts[0], pts[1], s=6, c=color)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/data.yaml')
    ap.add_argument('--split', default='data/splits/xsub_train.csv')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--out', default='outputs/sync_check')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = NTU120Multimodal(args.split, cfg['data_root'], cfg['num_frames'], cfg['rgb_size'],
                          cfg['rgb_resize_hw'], cfg['max_bodies'])
    # skeleton colour coords are pixels of the 1920x1080 colour camera
    scale = np.array([DISPLAY_HW[1] / 1920, DISPLAY_HW[0] / 1080])[:, None]
    for k, i in enumerate(pick_diverse(ds.df, args.n, cfg['seed']), 1):
        item = ds[i]  # the exact indices the training pipeline uses
        row = ds.df.iloc[i]
        frames = read_video_frames(ds.root / row.rgb_path, DISPLAY_HW)
        skel = load_skeleton(ds.root / row.skeleton_path, cfg['max_bodies'])
        assert len(frames) == item['rgb_num_frames']

        fig, axes = plt.subplots(2, len(SHOW), figsize=(4.2 * len(SHOW), 6.4))
        for col, p in enumerate(SHOW):
            ri, si = int(item['rgb_frame_idx'][p]), int(item['skeleton_frame_idx'][p])
            top, bottom = axes[0, col], axes[1, col]
            top.imshow(frames[ri])
            for m in range(cfg['max_bodies']):
                draw_skeleton(top, skel['color'][:, si, :, m] * scale, BODY_COLORS[m])
                draw_skeleton(bottom, skel['xyz'][:2, si, :, m], BODY_COLORS[m])
            top.set_title(f'RGB frame {ri}/{item["rgb_num_frames"] - 1}  |  skel frame {si}/{item["skeleton_num_frames"] - 1}',
                          fontsize=8)
            top.axis('off')
            bottom.set_aspect('equal')
            bottom.set_title(f'3D skeleton (x, y), frame {si}', fontsize=8)
        fig.suptitle(f'{row.sample_id}   action {row.action_id} (label {row.label})   subject {row.subject_id}   '
                     f'setup {row.setup_id}   camera {row.camera_id}', fontsize=11)
        fig.tight_layout()
        path = out / f'{k:02d}_{row.sample_id}.png'
        fig.savefig(path, dpi=80)
        plt.close(fig)
        print(f'{path}  rgb_frames={item["rgb_num_frames"]} skel_frames={item["skeleton_num_frames"]}', flush=True)


if __name__ == '__main__':
    main()
