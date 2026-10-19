"""Read NTU .skeleton files into CTR-GCN ordered arrays (C x T x V x M) and check their integrity.

File format: frame count; then per frame a body count; per body one info line (10 values, first is
the body ID), a joint count, and one line per joint with 12 values:
x y z depthX depthY colorX colorY orientW orientX orientY orientZ trackingState.
"""
import numpy as np

NUM_JOINTS = 25
FATAL_ISSUES = ('parse_error', 'empty_sequence', 'no_bodies', 'nan', 'inf', 'all_zero')


def read_skeleton_file(path):
    """Raw parse -> (frame count from header, {body_id: {'frames', 'xyz', 'color'}}, max bodies in one frame)."""
    with open(path) as f:
        lines = iter(f.read().split('\n'))
    num_frames = int(next(lines))
    bodies, max_per_frame = {}, 0
    for t in range(num_frames):
        num_bodies = int(next(lines))
        max_per_frame = max(max_per_frame, num_bodies)
        for _ in range(num_bodies):
            body_id = next(lines).split()[0]
            num_joints = int(next(lines))
            if num_joints != NUM_JOINTS:
                raise ValueError(f'frame {t}: {num_joints} joints, expected {NUM_JOINTS}')
            joints = np.array([next(lines).split()[:7] for _ in range(num_joints)], dtype=np.float32)
            b = bodies.setdefault(body_id, {'frames': [], 'xyz': [], 'color': []})
            b['frames'].append(t)
            b['xyz'].append(joints[:, 0:3])
            b['color'].append(joints[:, 5:7])
    return num_frames, bodies, max_per_frame


def load_skeleton(path, max_bodies=2):
    """-> dict with xyz (3, T, 25, M) camera-space metres, color (2, T, 25, M) RGB pixel coords, counts.

    Frames where a selected body is not tracked stay zero. When more than max_bodies people were
    tracked, the ones with the most motion are kept.
    """
    # ponytail: motion-energy ranking only; CTR-GCN's full denoising heuristics (length/spread
    # thresholds) are not replicated. Add them if the skeleton baseline falls well short of the paper.
    num_frames, bodies, max_per_frame = read_skeleton_file(path)
    ranked = sorted(bodies.values(), key=lambda b: np.stack(b['xyz']).var(axis=0).sum(), reverse=True)
    xyz = np.zeros((3, num_frames, NUM_JOINTS, max_bodies), np.float32)
    color = np.zeros((2, num_frames, NUM_JOINTS, max_bodies), np.float32)
    for m, b in enumerate(ranked[:max_bodies]):
        # select body m first (view), then frames; [:, frames, :, m] would move the frame axis to the front
        xyz[..., m][:, b['frames']] = np.stack(b['xyz']).transpose(2, 0, 1)
        color[..., m][:, b['frames']] = np.stack(b['color']).transpose(2, 0, 1)
    return {'xyz': xyz, 'color': color, 'num_frames': num_frames,
            'num_bodies_total': len(bodies), 'max_bodies_per_frame': max_per_frame}


def check_skeleton(path, max_bodies=2):
    """-> (loaded sample or None, list of issue strings). Never raises."""
    try:
        s = load_skeleton(path, max_bodies)
    except Exception as e:  # malformed file of any kind is recorded, not fatal to the scan
        return None, [f'parse_error: {type(e).__name__}: {e}']
    issues = []
    if s['num_frames'] == 0:
        issues.append('empty_sequence')
    if s['num_bodies_total'] == 0:
        issues.append('no_bodies')
    if np.isnan(s['xyz']).any():
        issues.append('nan')
    if np.isinf(s['xyz']).any():
        issues.append('inf')
    if s['num_frames'] and s['num_bodies_total'] and not s['xyz'].any():
        issues.append('all_zero')
    if s['max_bodies_per_frame'] > max_bodies:
        issues.append(f'too_many_bodies({s["max_bodies_per_frame"]})')
    empty = int((~s['xyz'].any(axis=(0, 2, 3))).sum())
    if s['num_bodies_total'] and empty:
        issues.append(f'frames_without_body({empty}/{s["num_frames"]})')
    return s, issues


def is_fatal(issues):
    return any(i.startswith(FATAL_ISSUES) for i in issues)
