"""Exact RGB frame count per video by decoding every frame (container headers can be wrong).

Writes data/metadata/rgb_frame_counts.csv with sample_id, rgb_num_frames, header_num_frames.
Usage: python scripts/rgb_frame_counts.py [--config configs/data.yaml] [--workers N]
"""
import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.ntu import read_metadata  # noqa: E402


def count(args):
    sid, path = args
    cap = cv2.VideoCapture(str(path))
    header = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return {'sample_id': sid, 'rgb_num_frames': n, 'header_num_frames': header}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/data.yaml')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    args = ap.parse_args()
    root = Path(yaml.safe_load(open(args.config))['data_root'])
    meta = read_metadata(root)
    meta = meta[meta.rgb_exists]
    with Pool(args.workers) as pool:
        rows = pool.map(count, [(s, root / p) for s, p in zip(meta.sample_id, meta.rgb_path)], chunksize=16)
    df = pd.DataFrame(rows)
    df.to_csv(root / 'metadata/rgb_frame_counts.csv', index=False)
    bad = df[df.rgb_num_frames == 0]
    print(f'{len(df)} videos; frames min {df.rgb_num_frames.min()} max {df.rgb_num_frames.max()}; '
          f'header count wrong for {(df.rgb_num_frames != df.header_num_frames).sum()}; undecodable {len(bad)}')
    if len(bad):
        print('  undecodable:', bad.sample_id.tolist()[:10])


if __name__ == '__main__':
    main()
