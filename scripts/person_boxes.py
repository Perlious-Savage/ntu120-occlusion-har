"""Per-actor bounding boxes in the 1920x1080 colour image, from the skeleton's colour-camera joint
projections (2nd-98th percentile over that actor's tracked joints and frames). Columns x0..y1 are
actor 1, x0_2..y1_2 actor 2 (empty for single-actor samples).
Used to place RGB occlusions on the person. Writes data/metadata/person_boxes.csv.

Usage: python scripts/person_boxes.py [--config configs/data.yaml] [--workers N]
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

ROOT = None


def init(root):
    global ROOT
    ROOT = root


def box(item):
    sid, rel = item
    row = {'sample_id': sid}
    try:
        s = load_skeleton(ROOT / rel, duplicate_single=False)
    except Exception:
        return row
    for m in range(s['num_actors']):
        pts = s['color'][..., m].reshape(2, -1)
        pts = pts[:, (pts != 0).any(axis=0)]              # tracked joints only
        if pts.shape[1]:
            lo, hi = np.percentile(pts, 2, axis=1), np.percentile(pts, 98, axis=1)
            sfx = '' if m == 0 else '_2'
            row.update({'x0' + sfx: lo[0], 'y0' + sfx: lo[1], 'x1' + sfx: hi[0], 'y1' + sfx: hi[1]})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/data.yaml')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    args = ap.parse_args()
    root = Path(yaml.safe_load(open(args.config))['data_root'])
    meta = read_metadata(root)
    meta = meta[meta.skeleton_exists]
    with Pool(args.workers, init, (root,)) as pool:
        rows = pool.map(box, list(zip(meta.sample_id, meta.skeleton_path)), chunksize=64)
    df = pd.DataFrame(rows, columns=['sample_id', 'x0', 'y0', 'x1', 'y1', 'x0_2', 'y0_2', 'x1_2', 'y1_2']).round(1)
    df.to_csv(root / 'metadata/person_boxes.csv', index=False)
    w, h = df.x1 - df.x0, df.y1 - df.y0
    print(f'{len(df)} samples, {df.x0.isna().sum()} without tracked joints, {df.x0_2.notna().sum()} with 2 actors | median size {w.median():.0f} x {h.median():.0f} px '
          f'(of 1920 x 1080)')


if __name__ == '__main__':
    main()
