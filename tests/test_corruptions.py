"""Corruption checks: identity at S0, determinism, severity schedules, person-slot handling.
Run: python tests/test_corruptions.py   (or pytest)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.corruptions.base import sample_seed  # noqa: E402
from src.corruptions.registry import CORRUPTIONS, RGB, SKELETON, get  # noqa: E402
from src.corruptions.skeleton_part_dropout import UPPER_BODY  # noqa: E402

rng = np.random.default_rng(0)
CLIP = rng.integers(1, 256, size=(16, 112, 112, 3), dtype=np.uint8)   # no pixel is 0 before corruption
SKEL = rng.normal(0, 0.5, size=(3, 64, 25, 2)).astype(np.float32)     # two distinct persons, all tracked
SKEL_SINGLE = np.repeat(SKEL[..., :1], 2, axis=-1)                    # one actor copied to slot 2
BOX = (20.0, 10.0, 90.0, 110.0)
COLOR = rng.uniform(0, 112, size=(2, 64, 25, 2))                     # joint positions in clip pixels
CTX = {'boxes': [BOX], 'color': COLOR}


def inp(name):
    return CLIP if get(name).modality == 'rgb' else SKEL


def test_identity_and_determinism():
    for name, c in CORRUPTIONS.items():
        x = inp(name)
        assert c.apply(x, 0, 1) is x
        for s in range(1, 6):
            a, b = c.apply(x, s, 123, **CTX), c.apply(x, s, 123, **CTX)
            assert np.array_equal(a, b), f'{name} S{s} not deterministic'
            assert not np.array_equal(a, x), f'{name} S{s} changed nothing'
            assert a.shape == x.shape and a.dtype == x.dtype
        assert sample_seed('S001C001P001R001A001', name, 3) == sample_seed('S001C001P001R001A001', name, 3)
        assert sample_seed('S001C001P001R001A001', name, 3) != sample_seed('S001C001P001R001A002', name, 3)


def test_rgb_schedules():
    occ = [(get('rgb_occlusion').apply(CLIP, s, 7, boxes=[BOX])[0] == 0).all(-1).mean() for s in range(1, 6)]
    box_area = (BOX[2] - BOX[0]) * (BOX[3] - BOX[1]) / 112 ** 2
    for s, f in enumerate(occ, 1):
        assert abs(f - 0.1 * s * box_area) < 0.02, (s, f)          # covers 10-50% of the person box
    masked = [(get('rgb_temporal_mask').apply(CLIP, s, 7) == 0).all(axis=(1, 2, 3)).sum() for s in range(1, 6)]
    assert masked == [2, 3, 5, 6, 8], masked
    blur = [np.abs(get('rgb_blur').apply(CLIP, s, 7).astype(float) - CLIP).mean() for s in range(1, 6)]
    assert all(a < b for a, b in zip(blur, blur[1:])), blur      # stronger blur, larger change


def test_occlusion_one_rectangle_per_person():
    left, right = (5.0, 10.0, 35.0, 100.0), (75.0, 10.0, 105.0, 100.0)
    out = get('rgb_occlusion').apply(CLIP, 5, 3, boxes=[left, right, (np.nan,) * 4])
    black = (out[0] == 0).all(-1)
    assert black[:, 5:35].mean() > 0.1 and black[:, 75:105].mean() > 0.1   # both people occluded
    assert not black[:, 40:70].any()                                      # nothing between them


def test_coupled_occlusion_hides_the_joints_behind_the_rgb_occluder():
    for s in range(1, 6):
        seed = sample_seed('S001C001P001R001A001', 'rgb_occlusion', s)
        black = (get('rgb_occlusion').apply(CLIP, s, seed, boxes=[BOX])[0] == 0).all(-1)
        out = get('skeleton_coupled_occlusion').apply(SKEL, s, seed, **CTX)
        behind = black[COLOR[1].astype(int), COLOR[0].astype(int)]          # (T, V, M) joint pixel is black
        assert behind.any() and np.array_equal(~out.any(axis=0), behind), s


def test_skeleton_schedules():
    drop = [1 - get('skeleton_joint_dropout').apply(SKEL, s, 7)[..., 0].any(axis=0).mean() for s in range(1, 6)]
    for got, want in zip(drop, (0.05, 0.10, 0.20, 0.30, 0.40)):
        assert abs(got - want) < 0.02, (got, want)
    lost = [25 - get('skeleton_part_dropout').apply(SKEL, s, 7)[:, 0, :, 0].any(axis=0).sum() for s in range(1, 6)]
    assert lost == [4, 6, 9, 16, 20], lost
    s4 = get('skeleton_part_dropout').apply(SKEL, 4, 7)
    assert not s4[:, :, UPPER_BODY].any() and s4[:, :, 0].all()
    noise = [np.std(get('skeleton_noise').apply(SKEL, s, 7) - SKEL) for s in range(1, 6)]
    assert all(a < b for a, b in zip(noise, noise[1:])), noise
    zeros = SKEL.copy(); zeros[:, :10] = 0                          # untracked frames stay untracked
    assert not get('skeleton_noise').apply(zeros, 5, 7)[:, :10].any()


def test_controls_hit_the_same_frames_and_joints_without_sentinels():
    for s in range(1, 6):
        seed = sample_seed('S001C001P001R001A001', 'rgb_temporal_mask', s)
        black = (get('rgb_temporal_mask').apply(CLIP, s, seed) == 0).all(axis=(1, 2, 3))
        frz = get('rgb_temporal_freeze').apply(CLIP, s, seed)
        changed = (frz != CLIP).any(axis=(1, 2, 3))
        assert np.array_equal(changed, black), s                       # exactly the masked frames
        a = int(np.flatnonzero(black)[0])
        src = CLIP[a - 1] if a else CLIP[np.flatnonzero(black)[-1] + 1]
        assert all(np.array_equal(f, src) for f in frz[black]) and (frz != 0).any(axis=-1).all()
        seed = sample_seed('S001C001P001R001A001', 'skeleton_joint_dropout', s)
        gone = ~get('skeleton_joint_dropout').apply(SKEL, s, seed).any(axis=0)
        ramp = np.broadcast_to(np.linspace(-1, 1, 64)[None, :, None, None], SKEL.shape).astype(np.float32) + 2
        gone_r = ~get('skeleton_joint_dropout').apply(ramp, s, seed).any(axis=0)
        itp = get('skeleton_joint_interp').apply(ramp, s, seed)
        assert gone.any() and itp.all()                                # no zeros left
        kept, t = ~gone_r, np.arange(64)[:, None, None]
        first = np.where(kept.any(0), kept.argmax(0), 64)
        last = np.where(kept.any(0), 63 - kept[::-1].argmax(0), -1)
        inner = gone_r & (t > first) & (t < last)                       # ends are held flat, not extrapolated
        assert np.allclose(itp[:, inner], ramp[:, inner], atol=1e-5), s  # a linear trajectory is recovered


def test_coupled_interp_refills_exactly_the_coupled_joints():
    ramp = np.broadcast_to(np.linspace(-1, 1, 64)[None, :, None, None], SKEL.shape).astype(np.float32) + 2
    for s in range(1, 6):                                          # single actor (copied slot): check slot 0
        seed = sample_seed('S001C001P001R001A001', 'rgb_occlusion', s)
        hidden = ~get('skeleton_coupled_occlusion').apply(ramp, s, seed, **CTX)[..., 0].any(axis=0)
        out = get('skeleton_coupled_interp').apply(ramp, s, seed, **CTX)
        changed = (out[..., 0] != ramp[..., 0]).any(axis=0)
        assert hidden.any() and not (changed & ~hidden).any(), s     # only joints behind the box change
        assert out.all() and np.array_equal(out[..., 0], out[..., 1])  # no zeros; the copy stays a copy


def test_single_actor_copy_stays_a_copy():
    for name in SKELETON:
        out = get(name).apply(SKEL_SINGLE, 3, 9, **CTX)
        assert np.array_equal(out[..., 0], out[..., 1]), name
        two = get(name).apply(SKEL, 3, 9, **CTX)
        if name != 'skeleton_part_dropout':                        # parts: same joints for everyone by design
            d0, d1 = two[..., 0] - SKEL[..., 0], two[..., 1] - SKEL[..., 1]
            assert not np.allclose(d0, d1), name                   # two real people: independent corruption


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
