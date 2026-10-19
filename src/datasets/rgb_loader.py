"""Decode NTU RGB videos and turn sampled frames into 3 x T x H x W clips."""
import cv2
import numpy as np
import torch

# Parallelism comes from DataLoader workers; OpenCV's own thread pool in every worker oversubscribes the CPU.
cv2.setNumThreads(1)


def read_video_frames(path, resize_hw=(128, 171)):
    """Decode every frame (exact frame count, no unreliable seeking) -> uint8 RGB (N, H, W, 3).
    Frames are resized while decoding so a 1080p video never sits in memory at full size."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (resize_hw[1], resize_hw[0]), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        raise IOError(f'could not decode any frame from {path}')
    return np.stack(frames)[..., ::-1]  # BGR -> RGB


def to_clip_tensor(frames, size=112):
    """(T, H, W, 3) uint8 -> center-cropped float tensor (3, T, size, size) in [0, 1]."""
    _, h, w, _ = frames.shape
    top, left = (h - size) // 2, (w - size) // 2
    clip = np.ascontiguousarray(frames[:, top:top + size, left:left + size])
    return torch.from_numpy(clip).permute(3, 0, 1, 2).float().div_(255)
