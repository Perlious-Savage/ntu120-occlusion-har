"""Visual validation of the six corruptions (Milestone 3, step 3.5): Clean, S1, S3, S5 on N actions.

Per sample, three figures in outputs/corruption_examples/:
  *_rgb.png       occlusion and blur on the middle frame of the exact model input (112 x 112)
  *_temporal.png  temporal masking as a strip of all 16 input frames
  *_skeleton.png  joint dropout, body-part dropout and noise on the middle frame of the CTR-GCN input
Usage: python scripts/visualize_corruptions.py [--split data/splits/xsub_test.csv] [--n 10]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.corruptions.base import sample_seed  # noqa: E402
from src.corruptions.registry import get  # noqa: E402
from src.corruptions.rgb_occlusion import boxes_in_clip  # noqa: E402
from src.datasets.baseline_datasets import SkeletonClips  # noqa: E402
from src.datasets.rgb_loader import read_selected_frames  # noqa: E402
from src.datasets.temporal import uniform_temporal_indices  # noqa: E402
from src.utils.ntu import NTU_BONES  # noqa: E402

SHOW = [0, 1, 3, 5]


def model_clip(path, num_frames, size, resize_hw):
    """The exact uint8 R3D-18 eval input (16, 112, 112, 3) plus the crop offsets used."""
    _, frames = read_selected_frames(path, lambda n: uniform_temporal_indices(n, num_frames), resize_hw)
    top, left = (resize_hw[0] - size) // 2, (resize_hw[1] - size) // 2
    return np.ascontiguousarray(frames[:, top:top + size, left:left + size]), top, left


def limits(x, t, pad=0.15):
    """Axis limits of the clean pose, so every corrupted panel of a row uses the same scale."""
    p = x[:2, t][:, x[:, t].any(axis=0)]
    lo, hi = p.min(axis=1), p.max(axis=1)
    m = pad * (hi - lo).max()
    return (lo[0] - m, hi[0] + m), (lo[1] - m, hi[1] + m)


def draw_skeleton(ax, x, t, lims):
    for m in range(x.shape[-1]):
        if m == 1 and np.array_equal(x[..., 0], x[..., 1]):
            break                                      # single actor copied to slot 2: draw once
        p = x[:2, t, :, m]
        ok = x[:, t, :, m].any(axis=0)
        for a, b in NTU_BONES:
            if ok[a] and ok[b]:
                ax.plot(p[0, [a, b]], p[1, [a, b]], color='tab:green' if m == 0 else 'tab:purple', lw=1.5)
        ax.scatter(p[0, ok], p[1, ok], s=6, c='k')
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1])
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/data.yaml')
    ap.add_argument('--split', default='data/splits/xsub_test.csv')
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--out', default='outputs/corruption_examples')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root, out = Path(cfg['data_root']), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.split)
    boxes = pd.read_csv(root / 'metadata/person_boxes.csv').set_index('sample_id')
    picks = df.sample(frac=1, random_state=cfg['seed']).groupby('action_id').head(1).head(args.n)
    skel_ds = SkeletonClips(args.split, root, train=False)
    pos = {s: i for i, s in enumerate(skel_ds.df.sample_id)}

    for k, (_, row) in enumerate(picks.iterrows(), 1):
        sid, title = row.sample_id, f'{row.sample_id}  action {row.action_id}'
        clip, top, left = model_clip(root / row.rgb_path, cfg['num_frames'], cfg['rgb_size'], cfg['rgb_resize_hw'])
        person = boxes_in_clip(boxes.loc[sid], cfg['rgb_resize_hw'], cfg['rgb_size'])
        mid = len(clip) // 2

        fig, axes = plt.subplots(2, 4, figsize=(10, 5.4))
        for r, name in enumerate(['rgb_occlusion', 'rgb_blur']):
            for c, s in enumerate(SHOW):
                ax = axes[r, c]
                ax.imshow(get(name).apply(clip, s, sample_seed(sid, name, s), boxes=person)[mid])
                ax.set_title(f'{name}  {"clean" if s == 0 else f"S{s}"}', fontsize=8); ax.axis('off')
        fig.suptitle(title); fig.tight_layout(); fig.savefig(out / f'{k:02d}_{sid}_rgb.png', dpi=90); plt.close(fig)

        fig, axes = plt.subplots(len(SHOW), 1, figsize=(12, 4.2))
        for r, s in enumerate(SHOW):
            strip = get('rgb_temporal_mask').apply(clip, s, sample_seed(sid, 'rgb_temporal_mask', s))
            axes[r].imshow(np.concatenate(list(strip[:, ::2, ::2]), axis=1))
            axes[r].set_ylabel('clean' if s == 0 else f'S{s}', rotation=0, labelpad=18, fontsize=8)
            axes[r].set_xticks([]); axes[r].set_yticks([])
        axes[0].set_title(f'rgb_temporal_mask: all 16 input frames   ({title})', fontsize=9)
        fig.tight_layout(); fig.savefig(out / f'{k:02d}_{sid}_temporal.png', dpi=90); plt.close(fig)

        x = skel_ds[pos[sid]][0].numpy()
        tmid = x.shape[1] // 2
        fig, axes = plt.subplots(3, 4, figsize=(9, 8))
        for r, name in enumerate(['skeleton_joint_dropout', 'skeleton_part_dropout', 'skeleton_noise']):
            for c, s in enumerate(SHOW):
                draw_skeleton(axes[r, c], get(name).apply(x, s, sample_seed(sid, name, s)), tmid, limits(x, tmid))
                axes[r, c].set_title(f'{name.replace("skeleton_", "")}  {"clean" if s == 0 else f"S{s}"}', fontsize=8)
        fig.suptitle(f'{title}  (CTR-GCN input, frame {tmid} of {x.shape[1]})'); fig.tight_layout()
        fig.savefig(out / f'{k:02d}_{sid}_skeleton.png', dpi=90); plt.close(fig)
        print(f'{k:02d} {sid}', flush=True)


if __name__ == '__main__':
    main()
