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


def _skeleton_file(frames):
    """frames: per frame a list of (body_id, x_offset, width) -> path of a synthetic .skeleton file.
    Joint j of a body sits at x = x_offset + width * j / 24, y = j / 24 (a 1 m tall upright figure when width < 0.8)."""
    import tempfile
    def body(bid, x0, w):
        joints = '\n'.join(f'{x0 + w * j / 24} {j / 24} 3 0 0 {10 * j} {20 * j} 0 0 0 0 2' for j in range(25))
        return f'{bid} 0 0 0 0 0 0 0 0 2\n25\n{joints}'
    text = [f'{len(fr)}\n' + '\n'.join(body(*b) for b in fr) if fr else '0' for fr in frames]
    path = Path(tempfile.mkdtemp()) / 'S001C001P001R001A001.skeleton'
    path.write_text(f'{len(frames)}\n' + '\n'.join(text) + '\n')
    return path


def test_skeleton_loader():
    from src.datasets.skeleton_loader import check_skeleton, load_skeleton
    # one person walking right (x grows 0.01 m per frame) for 30 frames, frame 0 has nobody
    path = _skeleton_file([[]] + [[('A', 0.01 * t, 0.3)] for t in range(30)])
    s = load_skeleton(path)
    assert s['xyz'].shape == (3, 31, 25, 2) and s['color'].shape == (2, 31, 25, 2)
    assert s['num_actors'] == 1 and s['num_frames'] == 31
    assert not s['xyz'][:, 0].any()                                   # untracked frame stays zero
    assert np.isclose(s['xyz'][0, 5, 0, 0], 0.04)                     # true frame index kept (frame 5 = t 4)
    assert np.array_equal(s['xyz'][..., 0], s['xyz'][..., 1])         # single actor copied to slot 2
    assert s['color'][:, 1, 3, 0].tolist() == [30, 60]
    assert check_skeleton(path)[1] == ['frames_without_body(1/31)']


def test_actor_selection():
    """The official NTU denoising: noise bodies dropped, a person whose ID changed is merged back."""
    from src.datasets.skeleton_loader import load_skeleton
    T = 40
    frames = []
    for t in range(T):
        fr = [('P1a' if t < 20 else 'P1b', 0.02 * t, 0.3),  # person 1, tracking ID changes at frame 20
              ('P2', 1.5, 0.3)]                            # person 2, standing still
        if t % 2:
            fr.append(('chair', -1 + 0.05 * t, 1.6))        # wider than tall in every frame, jittery
        if 5 <= t < 12:
            fr.append(('blip', 3, 0.3))                     # tracked only 7 frames
        frames.append(fr)
    s = load_skeleton(_skeleton_file(frames))
    assert sorted(s['noise_bodies']) == ['blip', 'chair']
    assert s['actor_bodies'] == [['P1a', 'P1b'], ['P2']] and s['num_actors'] == 2
    assert np.isclose(s['xyz'][0, 30, 0, 0], 0.6) and np.isclose(s['xyz'][0, 30, 0, 1], 1.5)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
