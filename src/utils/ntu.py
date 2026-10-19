"""NTU RGB+D 120 sample-ID parsing and the official evaluation protocols."""
import re

SAMPLE_ID_RE = re.compile(r'S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})')

# Valid ranges of each ID field in NTU RGB+D 120.
FIELD_RANGES = {'setup_id': (1, 32), 'camera_id': (1, 3), 'subject_id': (1, 106),
                'replication_id': (1, 2), 'action_id': (1, 120)}

# Official NTU120 cross-subject (X-Sub) training subjects; all other subjects are test.
XSUB_TRAIN_SUBJECTS = frozenset({
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 45, 46, 47, 49,
    50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92,
    93, 94, 95, 97, 98, 100, 103})

# Kinect v2 skeleton bones as 0-indexed joint pairs (the NTU / CTR-GCN graph).
NTU_BONES = [(a - 1, b - 1) for a, b in [
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10),
    (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23),
    (23, 8), (24, 25), (25, 12)]]


def parse_sample_id(sample_id):
    """'S001C002P003R002A013' -> dict of integer fields plus label = action_id - 1 (0-119)."""
    m = SAMPLE_ID_RE.fullmatch(sample_id)
    if not m:
        raise ValueError(f'not an NTU sample id: {sample_id!r}')
    fields = dict(zip(FIELD_RANGES, map(int, m.groups())))
    for name, (lo, hi) in FIELD_RANGES.items():
        if not lo <= fields[name] <= hi:
            raise ValueError(f'{sample_id}: {name}={fields[name]} outside {lo}-{hi}')
    return {'sample_id': sample_id, **fields, 'label': fields['action_id'] - 1}


def format_sample_id(setup_id, camera_id, subject_id, replication_id, action_id, **_):
    """Inverse of parse_sample_id, used to verify the parser round-trips every filename."""
    return f'S{setup_id:03d}C{camera_id:03d}P{subject_id:03d}R{replication_id:03d}A{action_id:03d}'


def read_metadata(data_root):
    """metadata.csv with paths as strings (empty = file missing) instead of mixed str/NaN."""
    import pandas as pd
    return pd.read_csv(f'{data_root}/metadata/metadata.csv', dtype={'rgb_path': str, 'skeleton_path': str},
                       keep_default_na=False)


def xsub_split(subject_id):
    return 'train' if subject_id in XSUB_TRAIN_SUBJECTS else 'test'


def xset_split(setup_id):
    return 'train' if setup_id % 2 == 0 else 'test'
