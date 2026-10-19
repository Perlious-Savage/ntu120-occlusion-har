"""One-time RGB downscale: raw 1080p .avi -> small H.264 .mp4 with every frame kept.

Decoding 1080p MPEG-4 costs ~2 s per video, which makes training from raw files impossible.
The output keeps the original sample ID in the filename and the exact frame count (verified
against data/metadata/rgb_frame_counts.csv), so temporal synchronization is unchanged.

Writes data/processed/rgb_<H>p/<sample_id>_rgb.mp4
Usage: python scripts/shrink_rgb.py [--config configs/data.yaml] [--height 256] [--workers N]
"""
import argparse
import os
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.ntu import read_metadata  # noqa: E402


def frame_count(path):
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def shrink(job):
    sid, src, dst, height, expected = job
    if not (dst.exists() and frame_count(dst) == expected):  # skip videos already done (resumable)
        tmp = dst.with_suffix('.tmp.mp4')
        r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-threads', '2', '-i', str(src),
                            '-vf', f'scale=-2:{height}:flags=area', '-fps_mode', 'passthrough',
                            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p',
                            '-an', str(tmp)], capture_output=True, text=True)
        if r.returncode != 0:
            return sid, -1, r.stderr.strip()[-200:]
        tmp.replace(dst)
    n = frame_count(dst)
    return sid, n, '' if n == expected else f'frame count {n} != original {expected}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/data.yaml')
    ap.add_argument('--height', type=int, default=256)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    args = ap.parse_args()
    root = Path(yaml.safe_load(open(args.config))['data_root'])
    out = root / f'processed/rgb_{args.height}p'
    out.mkdir(parents=True, exist_ok=True)

    meta = read_metadata(root)
    counts = pd.read_csv(root / 'metadata/rgb_frame_counts.csv').set_index('sample_id').rgb_num_frames
    meta = meta[meta.rgb_exists]
    jobs = [(s, root / p, out / f'{s}_rgb.mp4', args.height, counts[s]) for s, p in zip(meta.sample_id, meta.rgb_path)]

    failed = []
    with Pool(args.workers) as pool:
        for i, (sid, n, err) in enumerate(pool.imap_unordered(shrink, jobs, chunksize=4), 1):
            if err:
                failed.append((sid, err))
                print(f'  FAILED {sid}: {err}', flush=True)
            if i % 1000 == 0:
                print(f'  {i}/{len(jobs)}', flush=True)
    size = sum(f.stat().st_size for f in out.glob('*.mp4'))
    print(f'{len(jobs) - len(failed)}/{len(jobs)} videos ok, frame counts verified; '
          f'{size / 2**30:.2f} GB in {out}')
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
