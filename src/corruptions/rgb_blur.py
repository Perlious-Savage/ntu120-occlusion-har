"""RGB Gaussian blur of every frame. Level = kernel size (S1-S5: 3, 5, 7, 11, 15 px), sigma from
OpenCV's rule for the kernel size. Applied at model-input resolution (112 x 112). Deterministic."""
import cv2

from .base import Corruption


class RGBBlur(Corruption):
    name = 'rgb_blur'
    modality = 'rgb'
    levels = (3, 5, 7, 11, 15)

    def _apply(self, x, k, rng, **_):
        for t in range(len(x)):
            x[t] = cv2.GaussianBlur(x[t], (k, k), 0)
        return x
