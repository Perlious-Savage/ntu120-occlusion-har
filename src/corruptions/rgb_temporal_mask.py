"""RGB temporal masking: one contiguous block of frames set to black.

Level = fraction of the clip's frames (S1-S5: 10-50%); with 16 frames that is 2, 3, 5, 6, 8
frames (rounded, at least 1). The block's start is uniform over valid positions (seeded).

rgb_temporal_freeze (review item A5) is the information-matched control: the SAME block (same seed via
`seed_as`) repeats the last frame before it (the first frame after it if the block starts the clip),
like a stalled stream. No black frames, so a gap to the black mask measures out-of-distribution
fragility rather than lost information.
"""
from .base import Corruption


def block(t, frac, rng):
    n = max(1, int(round(frac * t)))
    start = int(rng.integers(0, t - n + 1))
    return start, start + n


class RGBTemporalMask(Corruption):
    name = 'rgb_temporal_mask'
    modality = 'rgb'

    def _apply(self, x, frac, rng, **_):
        a, b = block(len(x), frac, rng)
        x[a:b] = 0
        return x


class RGBTemporalFreeze(Corruption):
    name = 'rgb_temporal_freeze'
    modality = 'rgb'
    seed_as = 'rgb_temporal_mask'
    extra = True

    def _apply(self, x, frac, rng, **_):
        a, b = block(len(x), frac, rng)
        x[a:b] = x[a - 1] if a > 0 else x[b]
        return x
