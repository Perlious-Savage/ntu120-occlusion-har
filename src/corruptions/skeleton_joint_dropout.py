"""Skeleton random joint dropout. Level = fraction of each person's tracked joint observations
(frame, joint pairs) set to zero (S1-S5: 5, 10, 20, 30, 40%), drawn independently (seeded).
Zero is the dataset's own "not tracked" value.

skeleton_joint_interp (review item A5) is the information-matched control: the SAME observations
(same seed via `seed_as`) are refilled by linear interpolation in time from the joint's nearest kept
frames (held flat at the ends), the way Kinect reports an inferred joint, instead of a zero sentinel.
"""
import numpy as np

from .base import Corruption, skeleton_slots


def dropped(x, p, rng):
    """-> [(slot, (T, V) bool mask of dropped observations)], drawing exactly as the dropout does."""
    out = []
    for m in skeleton_slots(x):
        tracked = x[..., m].any(axis=0)                        # (T, V)
        out.append((m, tracked & (rng.random(tracked.shape) < p)))
    return out


class SkeletonJointDropout(Corruption):
    name = 'skeleton_joint_dropout'
    modality = 'skeleton'

    def _apply(self, x, p, rng, **_):
        masks = dropped(x, p, rng)
        for m, drop in masks:
            x[:, drop, m] = 0
        if len(masks) == 1:
            x[..., 1] = x[..., 0]
        return x


class SkeletonJointInterp(Corruption):
    name = 'skeleton_joint_interp'
    modality = 'skeleton'
    seed_as = 'skeleton_joint_dropout'
    extra = True

    def _apply(self, x, p, rng, **_):
        masks = dropped(x, p, rng)
        t = np.arange(x.shape[1])
        for m, drop in masks:
            keep = x[..., m].any(axis=0) & ~drop               # (T, V)
            for v in np.flatnonzero(drop.any(axis=0)):
                k = keep[:, v]
                if k.any():
                    for c in range(3):
                        x[c, drop[:, v], v, m] = np.interp(t[drop[:, v]], t[k], x[c, k, v, m])
                else:                                          # no kept frame left: nothing to infer from
                    x[:, drop[:, v], v, m] = 0
        if len(masks) == 1:
            x[..., 1] = x[..., 0]
        return x
