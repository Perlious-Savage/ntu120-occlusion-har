"""Temporal sampling shared by RGB and skeleton so both modalities see the same moments."""
import numpy as np


def uniform_temporal_indices(num_frames, target_frames):
    """Indices of target_frames evenly spaced normalised positions 0..1 (both ends included) mapped
    into a num_frames sequence. Calling this with each modality's own length and the same
    target_frames gives the same normalised positions, even when the frame counts differ slightly."""
    if num_frames < 1:
        raise ValueError(f'cannot sample from a sequence of {num_frames} frames')
    positions = np.linspace(0.0, 1.0, target_frames)
    return np.rint(positions * (num_frames - 1)).astype(np.int64)
