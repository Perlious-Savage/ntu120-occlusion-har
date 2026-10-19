"""Dataset statistics (Milestone 1, step 13).

Needs: metadata.csv, skeleton_integrity_report.csv, rgb_frame_counts.csv, splits/*.csv
Writes: data/metadata/dataset_statistics.txt and dataset_statistics_per_class.csv
Usage: python scripts/dataset_statistics.py [--config configs/data.yaml]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.ntu import read_metadata  # noqa: E402


def describe(s):
    return f'min {s.min():.0f}, max {s.max():.0f}, mean {s.mean():.1f}, median {s.median():.0f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/data.yaml')
    root = Path(yaml.safe_load(open(ap.parse_args().config))['data_root'])
    meta_dir, split_dir = root / 'metadata', root / 'splits'

    meta = read_metadata(root)
    report = pd.read_csv(meta_dir / 'skeleton_integrity_report.csv')
    rgb_len = pd.read_csv(meta_dir / 'rgb_frame_counts.csv')
    splits = {p.stem: pd.read_csv(p) for p in sorted(split_dir.glob('*.csv'))}
    matched = meta[meta.rgb_exists & meta.skeleton_exists]
    usable = pd.concat([splits['xsub_train'], splits['xsub_test']])

    lines = [
        'NTU RGB+D 120 dataset statistics',
        f'Total sample IDs found:                {len(meta)}',
        f'Matched multimodal samples (RGB+skel): {len(matched)}',
        f'Usable after integrity filtering:      {len(usable)}',
        f'Action classes: {meta.action_id.nunique()}   subjects: {meta.subject_id.nunique()}   '
        f'setups: {meta.setup_id.nunique()}   cameras: {meta.camera_id.nunique()}',
        '',
        *[f'{name:22} {len(df):7} samples' for name, df in splits.items()],
        '',
        f'RGB length (frames):      {describe(rgb_len.rgb_num_frames)}',
        f'Skeleton length (frames): {describe(report.num_frames.dropna())}',
        '',
        f'Missing RGB files:        {(~meta.rgb_exists).sum()}',
        f'Missing skeleton files:   {(~meta.skeleton_exists).sum()}',
        f'Invalid skeleton samples (fatal integrity issue): {report.fatal.sum()}',
        f'On official NTU missing-skeleton list:            {meta.skeleton_official_missing.sum()}',
        f'Skeletons with any issue (incl. non-fatal):       {(report.issues.fillna("") != "").sum()}',
        f'RGB vs skeleton length differs (matched samples): '
        f'{(matched.merge(rgb_len, on="sample_id").merge(report, on="sample_id").eval("rgb_num_frames != num_frames")).sum()}',
    ]
    text = '\n'.join(lines)
    (meta_dir / 'dataset_statistics.txt').write_text(text + '\n')
    print(text)

    per_class = pd.DataFrame({'all_matched': matched.groupby('action_id').size()})
    for name in ['xsub_train', 'xsub_test', 'xset_train', 'xset_test']:
        per_class[name] = splits[name].groupby('action_id').size()
    per_class.fillna(0).astype(int).to_csv(meta_dir / 'dataset_statistics_per_class.csv')
    print(f'\nper-class counts -> {meta_dir / "dataset_statistics_per_class.csv"}')


if __name__ == '__main__':
    main()
