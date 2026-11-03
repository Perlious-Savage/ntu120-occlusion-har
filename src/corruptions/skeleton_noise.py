"""Skeleton joint-coordinate noise: i.i.d. Gaussian added to tracked joints only.

sigma = alpha * body scale, alpha (S1-S5) = 0.01, 0.02, 0.04, 0.08, 0.12. Body scale = median
over tracked frames of the person's vertical (y) extent, i.e. about their height in metres
(about 1.6 standing, less when seated); computed per person from the clean input.
"""
import numpy as np

from .base import Corruption, skeleton_slots


def body_scale(person):
    """person: (3, T, V) -> median per-frame y extent of tracked joints (fallback 1.0 m)."""
    y = person[1]
    tracked = person.any(axis=0)
    ext = [np.ptp(y[t, tracked[t]]) for t in range(y.shape[0]) if tracked[t].sum() >= 2]
    return float(np.median(ext)) if ext else 1.0


class SkeletonNoise(Corruption):
    name = 'skeleton_noise'
    modality = 'skeleton'

    def _apply(self, x, alpha, rng, **_):
        slots = skeleton_slots(x)
        for m in slots:
            tracked = x[..., m].any(axis=0)                    # (T, V)
            sigma = alpha * body_scale(x[..., m])
            noise = rng.normal(0, sigma, size=x[..., m].shape).astype(x.dtype)
            x[..., m] += noise * tracked
        if slots == [0]:
            x[..., 1] = x[..., 0]
        return x
