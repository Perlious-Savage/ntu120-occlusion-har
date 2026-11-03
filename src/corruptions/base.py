"""Common corruption interface: apply(x, severity, seed) -> corrupted copy (Milestone 3).

RGB corruptions take the model-input clip as uint8 (T, H, W, 3); skeleton corruptions take the
CTR-GCN input as float (3, T, V, M). Severity 0 returns the input unchanged; S1-S5 index `levels`.
The same (input, severity, seed) always gives the same output.
"""
import zlib

import numpy as np

SEVERITIES = (1, 2, 3, 4, 5)


class Corruption:
    name = ''
    modality = ''   # 'rgb' or 'skeleton'
    levels = ()     # S1-S5 parameters, set from configs/corruptions.yaml by registry.py

    def apply(self, x, severity, seed, **context):
        if severity == 0:
            return x
        if severity not in SEVERITIES:
            raise ValueError(f'severity must be 0-5, got {severity}')
        rng = np.random.default_rng(seed)
        return self._apply(np.array(x, copy=True), self.levels[severity - 1], rng, **context)

    def _apply(self, x, level, rng, **context):
        raise NotImplementedError


def sample_seed(sample_id, corruption, severity, base_seed=42):
    """Stable per-sample seed: the same sample always gets the same corruption, on any machine."""
    return zlib.crc32(f'{sample_id}|{corruption}|{severity}|{base_seed}'.encode())


def skeleton_slots(x):
    """Person slots to corrupt independently. A single actor copied into slot 2 (official CTR-GCN
    convention) is corrupted once and copied again, so the copy carries no extra clean information."""
    return [0] if np.array_equal(x[..., 0], x[..., 1]) else [0, 1]
