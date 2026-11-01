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


def read_selected_frames(path, select, resize_hw=(128, 171)):
    """-> (indices, uint8 RGB frames (len(indices), H, W, 3)) for indices = select(num_frames).

    Same frames as read_video_frames(path)[indices], but frames are decoded only up to the last one
    needed, and only the selected ones are colour-converted and resized (most of the per-video cost).
    The frame count comes from the container header, which matched a full decode for every NTU video
    (checked when the videos were re-encoded); a video shorter than its header falls back to a full decode.
    """
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n > 0:
        idx = select(n)
        wanted, got = set(idx.tolist()), {}
        for i in range(int(idx.max()) + 1):
            if not cap.grab():
                break
            if i in wanted:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                got[i] = cv2.resize(frame, (resize_hw[1], resize_hw[0]), interpolation=cv2.INTER_AREA)
        if len(got) == len(wanted):
            cap.release()
            return idx, np.stack([got[i] for i in idx])[..., ::-1]
    cap.release()
    frames = read_video_frames(path, resize_hw)
    idx = select(len(frames))
    return idx, frames[idx]


def to_clip_tensor(frames, size=112):
    """(T, H, W, 3) uint8 -> center-cropped float tensor (3, T, size, size) in [0, 1]."""
    _, h, w, _ = frames.shape
    top, left = (h - size) // 2, (w - size) // 2
    clip = np.ascontiguousarray(frames[:, top:top + size, left:left + size])
    return torch.from_numpy(clip).permute(3, 0, 1, 2).float().div_(255)
