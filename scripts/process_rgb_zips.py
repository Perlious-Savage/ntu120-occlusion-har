"""Shrink every NTU RGB zip one at a time (a machine with too little disk for all 269 GB, e.g. Colab).

Per zip: copy to local disk, extract videos into data/tmp/rgb, shrink_rgb, pack the shrunk videos
and their frame counts into one .tar on the store (e.g. Google Drive), delete the raw videos.
Zips whose .tar already exists on the store are skipped, so the run resumes after a disconnect.
--restore unpacks all .tar files from the store into data/processed/rgb_256p (start of a session).

Usage: python scripts/process_rgb_zips.py --zips <dir with nturgbd_rgb_s0XX.zip> --store <dir> [--restore]
"""
import argparse
import shutil
import sys
import tarfile
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.shrink_rgb import shrink_dir  # noqa: E402

OUT, TMP = Path('data/processed/rgb_256p'), Path('data/tmp')  # TMP is wiped per zip; never point it at data/raw
RAW = TMP / 'rgb'


def restore(store):
    OUT.mkdir(parents=True, exist_ok=True)
    tars = sorted(store.glob('rgb_256p_s*.tar'))
    for t in tars:
        with tarfile.open(t) as tf:
            tf.extractall(OUT, filter='data')
    print(f'restored {len(tars)} setups, {len(list(OUT.glob("*.mp4")))} videos into {OUT}')


def process(zip_on_store, store, workers):
    tag = zip_on_store.stem.split('_')[-1]  # s001
    final = store / f'rgb_256p_{tag}.tar'
    if final.exists():
        print(f'skip {tag} (already on store)')
        return
    t0 = time.time()
    shutil.rmtree(TMP, ignore_errors=True)
    RAW.mkdir(parents=True)
    local_zip = TMP / zip_on_store.name
    shutil.copy(zip_on_store, local_zip)
    with zipfile.ZipFile(local_zip) as zf:  # flat: original filenames, without the zip's inner folder
        for info in zf.infolist():
            if not info.is_dir():
                with zf.open(info) as s, open(RAW / Path(info.filename).name, 'wb') as d:
                    shutil.copyfileobj(s, d, 1 << 20)
    local_zip.unlink()
    df = shrink_dir(RAW, OUT, workers=workers, tag=tag)

    tar_tmp = TMP / final.name
    with tarfile.open(tar_tmp, 'w') as tf:
        for sid in df.sample_id:
            tf.add(OUT / f'{sid}_rgb.mp4', arcname=f'{sid}_rgb.mp4')
        tf.add(OUT / f'frame_counts_{tag}.csv', arcname=f'frame_counts_{tag}.csv')
    shutil.copy(tar_tmp, store / (final.name + '.part'))
    (store / (final.name + '.part')).rename(final)  # only complete archives get the final name
    shutil.rmtree(TMP)
    print(f'DONE {tag}: {len(df)} videos, {final.stat().st_size / 2**20:.0f} MB, {(time.time() - t0) / 60:.1f} min',
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zips', required=True, type=Path)
    ap.add_argument('--store', required=True, type=Path)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--restore', action='store_true')
    args = ap.parse_args()
    args.store.mkdir(parents=True, exist_ok=True)
    if args.restore:
        restore(args.store)
        return
    zips = sorted(args.zips.glob('nturgbd_rgb_s*.zip'))
    print(f'{len(zips)} RGB zips found')
    for z in zips:
        process(z, args.store, args.workers)
    print(f'all done: {len(list(args.store.glob("rgb_256p_s*.tar")))}/{len(zips)} setups on store')


if __name__ == '__main__':
    main()
