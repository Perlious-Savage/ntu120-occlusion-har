"""Shrink every NTU RGB zip one at a time (a machine with too little disk for all 269 GB, e.g. Colab).

Per zip: copy to local disk, extract the videos, shrink_rgb, pack the shrunk videos and their frame
counts into one .tar on the store (e.g. Google Drive), delete the extracted videos. The next zip is
copied and extracted in the background while the current one is being shrunk, so the CPUs never wait
on I/O. Zips whose .tar already exists on the store are skipped, so the run resumes after a disconnect.

--shard k/n processes every n-th zip starting at k, so n machines can split the work; a faster
machine can take several shares, e.g. --shard 0,1/5.
--restore unpacks all .tar files from the store into data/processed/rgb_256p (start of a session).

Usage: python scripts/process_rgb_zips.py --zips <dir with nturgbd_rgb_s0XX.zip> --store <dir>
           [--shard 0/1] [--restore]
"""
import argparse
import os
import shutil
import socket
import sys
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.shrink_rgb import shrink_dir  # noqa: E402

OUT, TMP = Path('data/processed/rgb_256p'), Path('data/tmp')  # TMP is wiped; never point it at data/raw


def tag_of(zip_path):
    return zip_path.stem.split('_')[-1]  # s001


def restore(store):
    OUT.mkdir(parents=True, exist_ok=True)
    tars = sorted(store.glob('rgb_256p_s*.tar'))
    for t in tars:
        with tarfile.open(t) as tf:
            tf.extractall(OUT, filter='data')
    print(f'restored {len(tars)} setups, {len(list(OUT.glob("*.mp4")))} videos into {OUT}')


def claim(zip_on_store, store, stale_minutes=45):
    """Shared work queue across machines: a zip is taken by writing <tar name>.claim (holder + time) on the store.
    Skips zips that are finished or freshly claimed by another machine; reclaims our own or stale claims."""
    tag = tag_of(zip_on_store)
    if (store / f'rgb_256p_{tag}.tar').exists():
        return False
    c = store / f'rgb_256p_{tag}.claim'
    me = os.environ.get('WORKER_ID') or socket.gethostname()
    if c.exists():
        holder = c.read_text().split()[0] if c.read_text().strip() else ''
        if holder != me and time.time() - c.stat().st_mtime < stale_minutes * 60:
            return False
    c.write_text(f'{me} {time.ctime()}\n')
    time.sleep(5)  # Drive is eventually consistent: if two machines raced, the later write wins
    return c.read_text().split()[0] == me


def prepare(zip_on_store):
    """Copy one zip to local disk and extract its videos (original names, no inner folder)."""
    work = TMP / tag_of(zip_on_store)
    shutil.rmtree(work, ignore_errors=True)
    (work / 'rgb').mkdir(parents=True)
    local_zip = work / zip_on_store.name
    shutil.copy(zip_on_store, local_zip)
    with zipfile.ZipFile(local_zip) as zf:
        for info in zf.infolist():
            if not info.is_dir():
                with zf.open(info) as s, open(work / 'rgb' / Path(info.filename).name, 'wb') as d:
                    shutil.copyfileobj(s, d, 1 << 20)
    local_zip.unlink()
    return work


def finish(work, tag, store, workers, t0):
    df = shrink_dir(work / 'rgb', OUT, workers=workers, tag=tag)
    final = store / f'rgb_256p_{tag}.tar'
    tar_tmp = work / final.name
    with tarfile.open(tar_tmp, 'w') as tf:
        for sid in df.sample_id:
            tf.add(OUT / f'{sid}_rgb.mp4', arcname=f'{sid}_rgb.mp4')
        tf.add(OUT / f'frame_counts_{tag}.csv', arcname=f'frame_counts_{tag}.csv')
    shutil.copy(tar_tmp, store / (final.name + '.part'))
    (store / (final.name + '.part')).rename(final)  # only complete archives get the final name
    (store / f'rgb_256p_{tag}.claim').unlink(missing_ok=True)
    shutil.rmtree(work)
    print(f'DONE {tag}: {len(df)} videos, {final.stat().st_size / 2**20:.0f} MB, '
          f'{(time.time() - t0) / 60:.1f} min since start', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zips', required=True, type=Path)
    ap.add_argument('--store', required=True, type=Path)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--shard', default='0/1',
                    help='k/n: this machine handles zips k, k+n, k+2n, ... (several shares: 0,1/5)')
    ap.add_argument('--reverse', action='store_true', help='process this shard from the end (to help a slower machine)')
    ap.add_argument('--restore', action='store_true')
    args = ap.parse_args()
    args.store.mkdir(parents=True, exist_ok=True)
    if args.restore:
        restore(args.store)
        return

    ks, n = args.shard.split('/')
    ks, n = {int(k) for k in ks.split(',')}, int(n)
    zips = [z for i, z in enumerate(sorted(args.zips.glob('nturgbd_rgb_s*.zip'))) if i % n in ks]
    k = args.shard.split('/')[0]
    todo = [z for z in zips if not (args.store / f'rgb_256p_{tag_of(z)}.tar').exists()]
    print(f'shard {k}/{n}: {len(zips)} zips, {len(zips) - len(todo)} already done, {len(todo)} to process: '
          f'{[tag_of(z) for z in todo]}', flush=True)
    if args.reverse:
        todo = todo[::-1]
    t0 = time.time()

    def next_job():
        """Claim the next zip nobody else is working on, prepare it; None when nothing is left."""
        while todo:
            z = todo.pop(0)
            if claim(z, args.store):
                return z, prepare(z)
        return None

    with ThreadPoolExecutor(1) as io:  # prefetch: claim + extract the next zip while the current one is shrunk
        nxt = io.submit(next_job)
        while (job := nxt.result()) is not None:
            nxt = io.submit(next_job)
            z, work = job
            finish(work, tag_of(z), args.store, args.workers, t0)
    done = len(list(args.store.glob('rgb_256p_s*.tar')))
    print(f'shard {k}/{n} finished. {done}/32 setups on store in total')


if __name__ == '__main__':
    main()
