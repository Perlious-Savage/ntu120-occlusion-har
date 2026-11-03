"""RGB temporal masking: one contiguous block of frames set to black.

Level = fraction of the clip's frames (S1-S5: 10-50%); with 16 frames that is 2, 3, 5, 6, 8
frames (rounded, at least 1). The block's start is uniform over valid positions (seeded).
"""
from .base import Corruption


class RGBTemporalMask(Corruption):
    name = 'rgb_temporal_mask'
    modality = 'rgb'

    def _apply(self, x, frac, rng, **_):
        t = len(x)
        n = max(1, int(round(frac * t)))
        start = int(rng.integers(0, t - n + 1))
        x[start:start + n] = 0
        return x
