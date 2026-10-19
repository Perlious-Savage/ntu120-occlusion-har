"""One-time RGB downscale: raw 1080p .avi -> small H.264 .mp4 with every frame kept.

Decoding 1080p MPEG-4 costs ~2 s per video, which makes training from raw files impossible.
Each output keeps the original sample ID in its filename, and its frame count is checked against
the raw video's decoded frame count, so temporal synchronization is unchanged.

Writes <out>/<sample_id>_rgb.mp4 and <out>/frame_counts_<tag>.csv
(sample_id, rgb_num_frames = decoded raw frame count, header_num_frames = raw container header).
Usage: python scripts/shrink_rgb.py [--src data/raw/rgb] [--out data/processed/rgb_256p] [--height 256] [--tag all]
"""
import argparse
import os
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import pandas as pd

RAW_SUFFIX = '_rgb.avi'


def frame_count(path):
    """(decoded frame count, container header count)."""
    cap = cv2.VideoCapture(str(path))
    header = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n, header


def shrink(job):
    src, dst, height = job
    sid = src.name[:-len(RAW_SUFFIX)]
    expected, header = frame_count(src)
    row = {'sample_id': sid, 'rgb_num_frames': expected, 'header_num_frames': header, 'error': ''}
    if expected == 0:
        row['error'] = 'raw video undecodable'
        return row
    if not (dst.exists() and frame_count(dst)[0] == expected):  # resumable: skip verified outputs
        tmp = dst.with_suffix('.tmp.mp4')
        r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-threads', '2', '-i', str(src),
                            '-vf', f'scale=-2:{height}:flags=area', '-fps_mode', 'passthrough',
                            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p',
                            '-an', str(tmp)], capture_output=True, text=True)
        if r.returncode != 0:
            row['error'] = 'ffmpeg: ' + r.stderr.strip()[-200:]
            return row
        tmp.replace(dst)
    n = frame_count(dst)[0]
    if n != expected:
        row['error'] = f'shrunk video has {n} frames, raw has {expected}'
    return row


def shrink_dir(src, out, height=256, workers=None, tag='all'):
    """Shrink every *_rgb.avi in src into out; returns the frame-count table (raises if any failed)."""
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(p, out / p.name.replace(RAW_SUFFIX, '_rgb.mp4'), height) for p in sorted(src.glob(f'*{RAW_SUFFIX}'))]
    with Pool(workers or os.cpu_count()) as pool:
        rows = []
        for i, row in enumerate(pool.imap_unordered(shrink, jobs, chunksize=4), 1):
            rows.append(row)
            if row['error']:
                print(f'  FAILED {row["sample_id"]}: {row["error"]}', flush=True)
            if i % 1000 == 0:
                print(f'  {i}/{len(jobs)}', flush=True)
    df = pd.DataFrame(rows).sort_values('sample_id')
    failed = df[df.error != '']
    df.drop(columns='error').to_csv(out / f'frame_counts_{tag}.csv', index=False)
    print(f'{len(df) - len(failed)}/{len(df)} videos shrunk, frame counts verified '
          f'(header count wrong for {(df.rgb_num_frames != df.header_num_frames).sum()})')
    if len(failed):
        raise RuntimeError(f'{len(failed)} videos failed, e.g. {failed.sample_id.tolist()[:5]}')
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/raw/rgb')
    ap.add_argument('--out', default='data/processed/rgb_256p')
    ap.add_argument('--height', type=int, default=256)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--tag', default='all')
    args = ap.parse_args()
    try:
        shrink_dir(Path(args.src), Path(args.out), args.height, args.workers, args.tag)
    except RuntimeError as e:
        sys.exit(str(e))


if __name__ == '__main__':
    main()
