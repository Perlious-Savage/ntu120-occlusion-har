"""Scan raw NTU120 files, parse sample IDs, match RGB and skeleton files, write data/metadata/metadata.csv.

Usage: python scripts/build_metadata.py [--config configs/data.yaml]
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.ntu import format_sample_id, parse_sample_id  # noqa: E402

RGB_SUFFIX, SKELETON_SUFFIX = '_rgb.avi', '.skeleton'
OFFICIAL_MISSING = 'NTU_RGBD120_samples_with_missing_skeletons.txt'


def scan(folder, suffix):
    """{sample_id: filename} for files ending in suffix; other files are returned separately."""
    found, other = {}, []
    for entry in os.scandir(folder):
        if entry.name.endswith(suffix):
            found[entry.name[:-len(suffix)]] = entry.name
        else:
            other.append(entry.name)
    return found, other


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/data.yaml')
    cfg = yaml.safe_load(open(ap.parse_args().config))
    root = Path(cfg['data_root'])

    rgb, rgb_other = scan(root / 'raw/rgb', RGB_SUFFIX)
    skel, skel_other = scan(root / 'raw/skeleton', SKELETON_SUFFIX)
    official_missing = {l.strip() for l in open(root / 'metadata' / OFFICIAL_MISSING) if l.strip().startswith('S')}

    rows, bad_ids = [], []
    for sid in sorted(rgb.keys() | skel.keys()):
        try:
            row = parse_sample_id(sid)
        except ValueError as e:
            bad_ids.append(str(e))
            continue
        assert format_sample_id(**row) == sid, f'parser round-trip failed for {sid}'
        row.update(
            rgb_path=f'raw/rgb/{rgb[sid]}' if sid in rgb else '',
            skeleton_path=f'raw/skeleton/{skel[sid]}' if sid in skel else '',
            rgb_exists=sid in rgb,
            skeleton_exists=sid in skel,
            skeleton_official_missing=sid in official_missing)
        rows.append(row)

    df = pd.DataFrame(rows)
    out = root / 'metadata/metadata.csv'
    df.to_csv(out, index=False)

    both = df.rgb_exists & df.skeleton_exists
    print(f'parsed and round-trip verified {len(df)} sample IDs -> {out}')
    print(f'  matched RGB + skeleton : {both.sum()}')
    print(f'  RGB missing            : {(~df.rgb_exists).sum()}')
    print(f'  skeleton missing       : {(~df.skeleton_exists).sum()}')
    print(f'  officially listed as missing/incomplete skeleton: {df.skeleton_official_missing.sum()}'
          f' (of {len(official_missing)} in NTU list)')
    print(f'  labels range           : {df.label.min()}-{df.label.max()}')
    for name, items in [('unparseable IDs', bad_ids), ('unexpected files in raw/rgb', rgb_other),
                        ('unexpected files in raw/skeleton', skel_other)]:
        if items:
            print(f'  WARNING {len(items)} {name}, e.g. {items[:3]}')


if __name__ == '__main__':
    main()
