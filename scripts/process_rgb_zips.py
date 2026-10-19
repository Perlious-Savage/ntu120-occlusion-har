"""Shrink every NTU RGB zip one at a time (a machine with too little disk for all 269 GB, e.g. Colab).

Per zip: copy to local disk, extract the videos, shrink_rgb, pack the shrunk videos and their frame
counts into one .tar on the store (e.g. Google Drive), delete the extracted videos. The next zip is
copied and extracted in the background while the current one is being shrunk, so the CPUs never wait
on I/O. Zips whose .tar already exists on the store are skipped, so the run resumes after a disconnect.

--shard k/n processes every n-th zip starting at k, so n machines can split the work.
--restore unpacks all .tar files from the store into data/processed/rgb_256p (start of a session).

Usage: python scripts/process_rgb_zips.py --zips <dir with nturgbd_rgb_s0XX.zip> --store <dir>
           [--shard 0/1] [--restore]
"""
import argparse
import shutil
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
    shutil.rmtree(work)
    print(f'DONE {tag}: {len(df)} videos, {final.stat().st_size / 2**20:.0f} MB, '
          f'{(time.time() - t0) / 60:.1f} min since start', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zips', required=True, type=Path)
    ap.add_argument('--store', required=True, type=Path)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--shard', default='0/1', help='k/n: this machine handles zips k, k+n, k+2n, ...')
    ap.add_argument('--restore', action='store_true')
    args = ap.parse_args()
    args.store.mkdir(parents=True, exist_ok=True)
    if args.restore:
        restore(args.store)
        return

    k, n = map(int, args.shard.split('/'))
    zips = sorted(args.zips.glob('nturgbd_rgb_s*.zip'))[k::n]
    todo = [z for z in zips if not (args.store / f'rgb_256p_{tag_of(z)}.tar').exists()]
    print(f'shard {k}/{n}: {len(zips)} zips, {len(zips) - len(todo)} already done, {len(todo)} to process: '
          f'{[tag_of(z) for z in todo]}', flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(1) as io:  # prefetch: extract zip i+1 while zip i is being shrunk
        nxt = io.submit(prepare, todo[0]) if todo else None
        for i, z in enumerate(todo):
            work = nxt.result()
            nxt = io.submit(prepare, todo[i + 1]) if i + 1 < len(todo) else None
            finish(work, tag_of(z), args.store, args.workers, t0)
    done = len(list(args.store.glob('rgb_256p_s*.tar')))
    print(f'shard {k}/{n} finished. {done}/32 setups on store in total')


if __name__ == '__main__':
    main()
