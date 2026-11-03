"""RGB spatial occlusion: one black rectangle per person, the same in every frame.

Level = fraction of each person's bounding box covered (S1-S5: 10-50%). Boxes come from the
skeleton's colour-camera projection, one per actor (context `boxes`: list of (x0, y0, x1, y1) in
clip pixels); without a valid box, the central half of the frame is used. Each rectangle's aspect
ratio is drawn in [0.5, 2] and its position uniformly inside its box (seeded).
"""
import numpy as np

from .base import Corruption


class RGBOcclusion(Corruption):
    name = 'rgb_occlusion'
    modality = 'rgb'
    levels = (0.10, 0.20, 0.30, 0.40, 0.50)

    def _apply(self, x, frac, rng, boxes=None, **_):
        _, h, w, _ = x.shape
        boxes = [b for b in (boxes or []) if np.all(np.isfinite(b))] or [(w / 4, h / 4, 3 * w / 4, 3 * h / 4)]
        for box in boxes:
            x0, y0, x1, y1 = (float(np.clip(v, 0, lim)) for v, lim in zip(box, (w, h, w, h)))
            bw, bh = max(x1 - x0, 2.0), max(y1 - y0, 2.0)
            aspect = np.exp(rng.uniform(np.log(0.5), np.log(2.0)))
            ow = min(bw, np.sqrt(frac * bw * bh * aspect))
            oh = min(bh, frac * bw * bh / ow)
            ox = x0 + rng.uniform(0, bw - ow)
            oy = y0 + rng.uniform(0, bh - oh)
            x[:, int(round(oy)):int(round(oy + oh)), int(round(ox)):int(round(ox + ow))] = 0
        return x


def boxes_in_clip(row, resize_hw=(128, 171), size=112):
    """Per-actor boxes of a person_boxes.csv row (1920 x 1080 colour pixels) mapped into the
    centre-cropped model input: resize to resize_hw, then crop size x size."""
    sx, sy = resize_hw[1] / 1920, resize_hw[0] / 1080
    top, left = (resize_hw[0] - size) // 2, (resize_hw[1] - size) // 2
    return [(row['x0' + k] * sx - left, row['y0' + k] * sy - top, row['x1' + k] * sx - left, row['y1' + k] * sy - top)
            for k in ('', '_2')]
