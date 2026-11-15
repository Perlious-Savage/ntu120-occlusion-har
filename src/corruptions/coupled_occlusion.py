"""Coupled occlusion, skeleton side: the occluder that blacks out RGB pixels also hides the joints behind it.

Uses the rgb_occlusion rectangles of the same sample and severity (same levels and seed via `seed_as`),
so the RGB input of a coupled condition is exactly rgb_occlusion_sN. Every joint of any actor whose
colour-camera projection falls inside a rectangle in that frame is set to zero. Context: `boxes`
(clip pixels, as for rgb_occlusion), `color` (2, T, V, M) joint positions in clip pixels, `size`.
"""
import numpy as np

from .base import Corruption, skeleton_slots
from .rgb_occlusion import occluders
from .skeleton_joint_dropout import interpolate


class SkeletonCoupledOcclusion(Corruption):
    name = 'skeleton_coupled_occlusion'
    modality = 'skeleton'
    seed_as = 'rgb_occlusion'
    extra = True
    needs_color = True

    def _apply(self, x, frac, rng, boxes=None, color=None, size=112, **_):
        u, v = color
        hidden = np.zeros(x.shape[1:], bool)
        for r0, r1, c0, c1 in occluders(boxes, frac, rng, size, size):
            hidden |= (v >= r0) & (v < r1) & (u >= c0) & (u < c1)
        x[:, hidden] = 0
        return x


class SkeletonCoupledInterp(Corruption):
    """Control for the coupled occlusion (review item A5 applied to B5): the SAME joints (behind the same
    rectangle) are refilled by interpolation in time instead of zero, like Kinect's inferred joints."""
    name = 'skeleton_coupled_interp'
    modality = 'skeleton'
    seed_as = 'rgb_occlusion'
    extra = True
    needs_color = True

    def _apply(self, x, frac, rng, boxes=None, color=None, size=112, **_):
        u, v = color
        hidden = np.zeros(x.shape[1:], bool)
        for r0, r1, c0, c1 in occluders(boxes, frac, rng, size, size):
            hidden |= (v >= r0) & (v < r1) & (u >= c0) & (u < c1)
        slots = skeleton_slots(x)
        for m in slots:
            interpolate(x, m, hidden[..., m] & x[..., m].any(axis=0))
        if slots == [0]:
            x[..., 1] = x[..., 0]
        return x
