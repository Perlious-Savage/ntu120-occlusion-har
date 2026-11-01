"""Create official X-Sub / X-Set train/test splits and internal train/val splits.

A sample is usable when both RGB and skeleton exist, it is not on NTU's official missing-skeleton
list, and check_skeletons.py found no fatal problem. The same usable set is used for every
modality so RGB, skeleton and later fusion results are directly comparable.

Usage: python scripts/make_splits.py [--config configs/data.yaml]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.ntu import read_metadata, xset_split, xsub_split  # noqa: E402


def subject_aware_val(train, frac, seed, tries=10000, tolerance=0.05, balance=0.75):
    """Hold out whole subjects so no validation subject appears in internal training.
    Subjects differ a lot in size (one NTU120 subject has 15% of X-Sub train) and NTU60 / NTU120 classes
    were recorded with different subjects, so subjects are added in a seeded random order until they hold
    `frac` of the training SAMPLES. The first draw (seeds tried in order from `seed`) is accepted that is
    within `tolerance` of that target and gives every class at least `balance` x its fair share."""
    sizes = train.groupby('subject_id').size()
    target, classes = frac * len(train), sorted(set(train.label))
    best = None  # (smallest class count, draw) of the draws near the target, used if none is balanced
    for t in range(tries):
        val_subj, n = [], 0
        for s in np.random.default_rng(seed + t).permutation(sizes.index.to_numpy()):
            if n >= target:
                break
            val_subj.append(int(s))
            n += sizes[s]
        val = train.subject_id.isin(val_subj)
        per_class = train.label[val].value_counts().reindex(classes, fill_value=0)
        if abs(n - target) > tolerance * target:
            continue
        if per_class.min() >= balance * target / len(classes):
            return val, sorted(val_subj), seed + t
        if best is None or per_class.min() > best[0]:
            best = (per_class.min(), (val, sorted(val_subj), seed + t))
    if best is None:  # tiny local subsets: too few subjects to get near the target at all
        print('  WARNING: no subject draw near the target size; using the first draw')
        return val, sorted(val_subj), seed + t
    print(f'  WARNING: no balanced subject draw; using the most balanced one ({best[0]} samples in its smallest class)')
    return best[1]


def stratified_val(train, frac, seed):
    if train.label.value_counts().min() < 2:  # stratification impossible (tiny local subsets)
        print('  WARNING: some class has <2 samples; falling back to unstratified split')
        _, val_idx = train_test_split(train.index, test_size=frac, random_state=seed)
    else:
        _, val_idx = train_test_split(train.index, test_size=frac, random_state=seed, stratify=train.label)
    return train.index.isin(val_idx)


def save(df, path):
    df.to_csv(path, index=False)
    print(f'  {path.name:28} {len(df):6} samples, {df.label.nunique():3} classes, '
          f'{df.subject_id.nunique():3} subjects, {df.setup_id.nunique():2} setups')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/data.yaml')
    cfg = yaml.safe_load(open(ap.parse_args().config))
    root, seed, frac = Path(cfg['data_root']), cfg['seed'], cfg['val_fraction']
    out = root / 'splits'
    out.mkdir(exist_ok=True)

    meta = read_metadata(root)
    report = pd.read_csv(root / 'metadata/skeleton_integrity_report.csv')
    fatal = set(report.sample_id[report.fatal])
    usable = meta[meta.rgb_exists & meta.skeleton_exists & ~meta.skeleton_official_missing
                  & ~meta.sample_id.isin(fatal)].reset_index(drop=True)
    print(f'usable samples: {len(usable)} of {len(meta)} '
          f'(excluded: {len(meta) - len(usable)} missing a modality / official-missing / fatal skeleton)')
    assert usable.label.between(0, 119).all(), 'labels must be 0-119'

    for proto, rule, col in [('xsub', xsub_split, 'subject_id'), ('xset', xset_split, 'setup_id')]:
        print(f'{proto}:')
        is_train = usable[col].map(rule) == 'train'
        train, test = usable[is_train].reset_index(drop=True), usable[~is_train].reset_index(drop=True)
        assert not set(train.sample_id) & set(test.sample_id), f'{proto}: train/test overlap'
        save(train, out / f'{proto}_train.csv')
        save(test, out / f'{proto}_test.csv')
        if train.empty:
            print(f'  WARNING: {proto} train is empty (expected only with partial local data); no val split')
            continue
        if proto == 'xsub':
            val, val_subjects, used_seed = subject_aware_val(train, frac, seed)
            print(f'  validation subjects (seed {used_seed}): {val_subjects}')
        else:
            val = stratified_val(train, frac, seed)
        save(train[~val], out / f'{proto}_train_internal.csv')
        save(train[val], out / f'{proto}_val.csv')
        assert not set(train.sample_id[val]) & set(train.sample_id[~val])
        assert not set(train.sample_id[val]) & set(test.sample_id)


if __name__ == '__main__':
    main()
