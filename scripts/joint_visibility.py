"""Kinect joint visibility for the natural-occlusion split and the coupled occlusion (Milestone 3/4).

Writes data/metadata/joint_tracking.csv: per sample, the share of the actors' joint observations that
Kinect marks "inferred" (trackingState 1: the joint was not seen by the camera, i.e. occluded; 2 = tracked).
With --cache it also writes data/processed/skeleton_color_f16.bin: the joints' colour-camera pixel
positions (2, T, 25, M) float16 per sample, in the same order as skeleton_xyz_f16.bin (offset = 2/3 of
the skeleton_index.csv offset). Needs the raw .skeleton files.

Usage: python scripts/joint_visibility.py [--cache] [--workers N]
"""
import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.skeleton_loader import load_skeleton  # noqa: E402
from src.utils.ntu import read_metadata  # noqa: E402

ROOT, CACHE = None, False


def init(root, cache):
    global ROOT, CACHE
    ROOT, CACHE = root, cache


def work(item):
    sid, rel, n_expected = item
    s = load_skeleton(ROOT / rel)
    assert s['num_frames'] == n_expected, f'{sid}: {s["num_frames"]} frames, cache has {n_expected}'
    st = s['state'][..., :max(s['num_actors'], 1)]                # the copied single actor is not counted twice
    obs = int((st > 0).sum())
    row = {'sample_id': sid, 'joint_observations': obs,
           'inferred_frac': round(float((st == 1).sum()) / max(obs, 1), 4)}
    if not CACHE:
        return row, b''
    blob = s['color'].astype(np.float16).tobytes()
    return row, blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/data.yaml')
    ap.add_argument('--cache', action='store_true')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    args = ap.parse_args()
    root = Path(yaml.safe_load(open(args.config))['data_root'])
    index = pd.read_csv(root / 'processed/skeleton_index.csv')
    paths = read_metadata(root).set_index('sample_id').skeleton_path
    items = list(zip(index.sample_id, paths.loc[index.sample_id], index.num_frames))
    rows, offset = [], 0
    cache = open(root / 'processed/skeleton_color_f16.bin', 'wb') if args.cache else None
    with Pool(args.workers, init, (root, args.cache)) as pool:
        for i, (row, blob) in enumerate(pool.imap(work, items, chunksize=64)):
            rows.append(row)
            if cache:
                assert 3 * offset == 2 * index.offset.iloc[i], f'{row["sample_id"]}: offset mismatch'
                cache.write(blob)
                offset += len(blob) // 2
            if (i + 1) % 10000 == 0:
                print(f'  {i + 1}/{len(items)}', flush=True)
    if cache:
        cache.close()
    df = pd.DataFrame(rows)
    df.to_csv(root / 'metadata/joint_tracking.csv', index=False)
    q = df.inferred_frac.quantile([0.5, 0.9, 0.99])
    print(f'{len(df)} samples; inferred share of joint observations: overall '
          f'{(df.inferred_frac * df.joint_observations).sum() / df.joint_observations.sum():.3f}, '
          f'per-sample median {q[0.5]:.3f}, p90 {q[0.9]:.3f}, p99 {q[0.99]:.3f}')


if __name__ == '__main__':
    main()
