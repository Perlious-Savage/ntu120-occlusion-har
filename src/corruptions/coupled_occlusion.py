"""Coupled occlusion, skeleton side: the occluder that blacks out RGB pixels also hides the joints behind it.

Uses the rgb_occlusion rectangles of the same sample and severity (same levels and seed via `seed_as`),
so the RGB input of a coupled condition is exactly rgb_occlusion_sN. Every joint of any actor whose
colour-camera projection falls inside a rectangle in that frame is set to zero. Context: `boxes`
(clip pixels, as for rgb_occlusion), `color` (2, T, V, M) joint positions in clip pixels, `size`.
"""
import numpy as np

from .base import Corruption
from .rgb_occlusion import occluders


class SkeletonCoupledOcclusion(Corruption):
    name = 'skeleton_coupled_occlusion'
    modality = 'skeleton'
    seed_as = 'rgb_occlusion'

    def _apply(self, x, frac, rng, boxes=None, color=None, size=112, **_):
        u, v = color
        hidden = np.zeros(x.shape[1:], bool)
        for r0, r1, c0, c1 in occluders(boxes, frac, rng, size, size):
            hidden |= (v >= r0) & (v < r1) & (u >= c0) & (u < c1)
        x[:, hidden] = 0
        return x
