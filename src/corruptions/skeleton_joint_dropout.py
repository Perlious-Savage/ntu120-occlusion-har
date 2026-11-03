"""Skeleton random joint dropout. Level = fraction of each person's tracked joint observations
(frame, joint pairs) set to zero (S1-S5: 5, 10, 20, 30, 40%), drawn independently (seeded).
Zero is the dataset's own "not tracked" value."""
import numpy as np

from .base import Corruption, skeleton_slots


class SkeletonJointDropout(Corruption):
    name = 'skeleton_joint_dropout'
    modality = 'skeleton'

    def _apply(self, x, p, rng, **_):
        slots = skeleton_slots(x)
        for m in slots:
            tracked = x[..., m].any(axis=0)                    # (T, V)
            drop = tracked & (rng.random(tracked.shape) < p)
            x[:, drop, m] = 0
        if slots == [0]:
            x[..., 1] = x[..., 0]
        return x
