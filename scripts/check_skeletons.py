"""Load every skeleton in metadata.csv, check integrity, and pack all of them into one binary cache.

Writes:
  data/metadata/skeleton_integrity_report.csv  one row per skeleton file: lengths, issues, fatal flag
  data/processed/skeleton_xyz_f16.bin          every parsed sample's xyz (3, T, 25, M) as float16, back to back
  data/processed/skeleton_index.csv            sample_id -> offset/num_frames into the .bin

Usage: python scripts/check_skeletons.py [--config configs/data.yaml] [--workers N]
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
from src.datasets.skeleton_loader import check_skeleton, is_fatal  # noqa: E402
from src.utils.ntu import read_metadata  # noqa: E402

ROOT, MAX_BODIES = None, 2


def init(root, max_bodies):
    global ROOT, MAX_BODIES
    ROOT, MAX_BODIES = root, max_bodies


def work(item):
    sid, rel_path = item
    s, issues = check_skeleton(ROOT / rel_path, MAX_BODIES)
    row = {'sample_id': sid, 'num_frames': s and s['num_frames'], 'num_bodies_total': s and s['num_bodies_total'],
           'max_bodies_per_frame': s and s['max_bodies_per_frame'], 'issues': ';'.join(issues),
           'fatal': is_fatal(issues)}
    # float16 keeps ~2 mm precision for coordinates within a few metres and halves the cache size
    return row, (s['xyz'].astype(np.float16).tobytes() if s else b'')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/data.yaml')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root = Path(cfg['data_root'])

    meta = read_metadata(root)
    meta = meta[meta.skeleton_exists]
    items = list(zip(meta.sample_id, meta.skeleton_path))
    (root / 'processed').mkdir(exist_ok=True)

    rows, index, offset = [], [], 0
    with Pool(args.workers, init, (root, cfg['max_bodies'])) as pool, \
            open(root / 'processed/skeleton_xyz_f16.bin', 'wb') as cache:
        for i, (row, blob) in enumerate(pool.imap(work, items, chunksize=64), 1):
            rows.append(row)
            if blob:
                cache.write(blob)
                index.append({'sample_id': row['sample_id'], 'offset': offset, 'num_frames': row['num_frames']})
                offset += len(blob) // 2  # float16 elements
            if i % 10000 == 0:
                print(f'  {i}/{len(items)}', flush=True)

    report = pd.DataFrame(rows).merge(meta[['sample_id', 'skeleton_official_missing']], on='sample_id')
    report.to_csv(root / 'metadata/skeleton_integrity_report.csv', index=False)
    pd.DataFrame(index).assign(max_bodies=cfg['max_bodies']).to_csv(root / 'processed/skeleton_index.csv', index=False)

    flagged = report[report.issues != '']
    fatal, official = report.fatal, report.skeleton_official_missing
    print(f'checked {len(report)} skeletons; {len(flagged)} with any issue, {fatal.sum()} fatal')
    for kind in ['parse_error', 'empty_sequence', 'no_bodies', 'nan', 'inf', 'all_zero',
                 'too_many_bodies', 'frames_without_body']:
        print(f'  {kind:20} {report.issues.str.contains(kind, regex=False).sum()}')
    print(f'official NTU missing list: {official.sum()} present here; '
          f'{(official & fatal).sum()} of them flagged fatal by us, '
          f'{(fatal & ~official).sum()} fatal samples NOT in the official list')
    print(f'cache: {offset * 2 / 2**30:.2f} GB, {len(index)} samples')


if __name__ == '__main__':
    main()
