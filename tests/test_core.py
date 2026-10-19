"""Core correctness checks. Run: python tests/test_core.py   (or pytest)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.temporal import uniform_temporal_indices  # noqa: E402
from src.utils.ntu import format_sample_id, parse_sample_id, xset_split, xsub_split  # noqa: E402


def test_parse_sample_id():
    r = parse_sample_id('S018C002P042R001A120')
    assert (r['setup_id'], r['camera_id'], r['subject_id'], r['replication_id'], r['action_id']) == (18, 2, 42, 1, 120)
    assert r['label'] == 119
    assert parse_sample_id('S001C001P001R001A001')['label'] == 0
    assert format_sample_id(**r) == 'S018C002P042R001A120'
    for bad in ['S001C001P001R001A121', 'S033C001P001R001A001', 'S001C004P001R001A001', 'S001C001P001R003A001',
                'S001C001P001R001A001_rgb', 'garbage']:
        try:
            parse_sample_id(bad)
        except ValueError:
            continue
        raise AssertionError(f'{bad} should be rejected')


def test_protocols():
    assert xsub_split(1) == 'train' and xsub_split(3) == 'test' and xsub_split(103) == 'train'
    assert xset_split(2) == 'train' and xset_split(1) == 'test' and xset_split(32) == 'train'


def test_uniform_temporal_indices():
    assert uniform_temporal_indices(16, 16).tolist() == list(range(16))
    idx = uniform_temporal_indices(103, 16)
    assert idx[0] == 0 and idx[-1] == 102 and len(idx) == 16 and (np.diff(idx) >= 0).all()
    # same normalised positions for slightly different lengths -> nearly the same relative time
    a, b = uniform_temporal_indices(100, 16) / 99, uniform_temporal_indices(103, 16) / 102
    assert np.abs(a - b).max() < 0.01
    assert uniform_temporal_indices(5, 16).max() == 4  # short sequences repeat frames


def test_skeleton_loader():
    """3 frames: body A in frames 0-2, body B only in frame 1, joint j has x = 100*frame + j."""
    import tempfile
    from src.datasets.skeleton_loader import check_skeleton, load_skeleton
    def body(bid, t):
        joints = '\n'.join(f'{100 * t + j} {j} 1 0 0 {10 * j} {20 * j} 0 0 0 0 2' for j in range(25))
        return f'{bid} 0 0 0 0 0 0 0 0 2\n25\n{joints}'
    frames = [f'1\n{body("A", 0)}', f'2\n{body("A", 1)}\n{body("B", 1)}', f'1\n{body("A", 2)}']
    path = Path(tempfile.mkdtemp()) / 'S001C001P001R001A001.skeleton'
    path.write_text('3\n' + '\n'.join(frames) + '\n')
    s = load_skeleton(path, max_bodies=2)
    assert s['xyz'].shape == (3, 3, 25, 2) and s['color'].shape == (2, 3, 25, 2)
    assert s['num_bodies_total'] == 2 and s['max_bodies_per_frame'] == 2
    # A moves (x changes every frame), B appears once -> A ranked first by motion
    assert s['xyz'][0, :, 7, 0].tolist() == [7, 107, 207]       # x of joint 7 over time
    assert s['xyz'][0, :, 7, 1].tolist() == [0, 107, 0]         # B only tracked in frame 1
    assert s['color'][:, 1, 3, 0].tolist() == [30, 60]
    assert check_skeleton(path)[1] == []


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
