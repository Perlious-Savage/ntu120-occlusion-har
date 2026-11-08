"""Read NTU .skeleton files into CTR-GCN ordered arrays (C x T x V x M) and check their integrity.

File format: frame count; then per frame a body count; per body one info line (10 values, first is
the body ID), a joint count, and one line per joint with 12 values:
x y z depthX depthY colorX colorY orientW orientX orientY orientZ trackingState.
"""
import numpy as np

NUM_JOINTS = 25
FATAL_ISSUES = ('parse_error', 'empty_sequence', 'no_bodies', 'nan', 'inf', 'all_zero')


def read_skeleton_file(path):
    """Raw parse -> (frame count from header, {body_id: {'frames', 'xyz', 'color', 'state'}}, max bodies in one frame).
    state is the Kinect trackingState per joint: 0 not tracked, 1 inferred (not seen by the camera), 2 tracked."""
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
            joints = np.array([next(lines).split()[:12] for _ in range(num_joints)], dtype=np.float32)
            b = bodies.setdefault(body_id, {'frames': [], 'xyz': [], 'color': [], 'state': []})
            b['frames'].append(t)
            b['xyz'].append(joints[:, 0:3])
            b['color'].append(joints[:, 5:7])
            b['state'].append(joints[:, 11])
    return num_frames, bodies, max_per_frame


# Official NTU denoising thresholds (Microsoft SGN data_gen/get_raw_denoised_data.py, used unchanged by CTR-GCN)
NOISE_LEN = 11            # a body tracked for <= 11 frames is noise
NOISE_SPREAD = 0.8        # a frame is noisy when the body's x-range > 0.8 * its y-range (wider than tall)
NOISE_SPREAD_RATE = 0.69754  # a body with >= this share of noisy frames is noise


def _motion(xyz):
    """Official 'motion': total variance of all the body's joint positions (T, 25, 3)."""
    return float(np.var(xyz.reshape(-1, 3), axis=0).sum())


def select_actors(bodies):
    """Official NTU body selection -> ([body ids of actor 1], [body ids of actor 2]), and ids removed as noise.

    With one tracked body it is the actor. Otherwise: drop bodies tracked <= 11 frames, then drop bodies
    that are wider than tall in >= 69.754% of their frames (Kinect false detections such as furniture),
    rank the rest by motion, and give the top one to actor 1. Each further body joins actor 1 if its
    frames do not overlap actor 1's span (a person whose tracking ID changed), else actor 2 under the
    same rule, else it is dropped. Mirrors get_raw_denoised_data.py, including its actor-2 span starting
    at frame 0, but uses true frame numbers so RGB and skeleton frames stay aligned.
    """
    ids = list(bodies)
    if len(ids) == 1:
        return [ids, []], []
    xyz = {k: np.stack(bodies[k]['xyz']) for k in ids}
    motion = {k: _motion(xyz[k]) for k in ids}
    kept = [k for k in ids if len(bodies[k]['frames']) > NOISE_LEN]
    for k in list(kept):
        if len(kept) == 1:
            break
        x, y = xyz[k][:, :, 0], xyz[k][:, :, 1]
        valid = np.ptp(x, axis=1) <= NOISE_SPREAD * np.ptp(y, axis=1)
        if valid.all():
            continue
        if 1 - valid.mean() >= NOISE_SPREAD_RATE:
            kept.remove(k)
        else:
            motion[k] = min(motion[k], _motion(xyz[k][valid]))
    if not kept:  # the official code fails here; keep the longest-tracked body instead
        kept = [max(ids, key=lambda k: len(bodies[k]['frames']))]
    order = sorted(kept, key=lambda k: motion[k], reverse=True)

    actors, (s1, e1), (s2, e2) = [[order[0]], []], (bodies[order[0]]['frames'][0], bodies[order[0]]['frames'][-1]), (0, 0)
    for k in order[1:]:
        s, e = bodies[k]['frames'][0], bodies[k]['frames'][-1]
        if min(e1, e) - max(s1, s) <= 0:
            actors[0].append(k)
            s1, e1 = min(s, s1), max(e, e1)
        elif min(e2, e) - max(s2, s) <= 0:
            actors[1].append(k)
            s2, e2 = min(s, s2), max(e, e2)
    return actors, [k for k in ids if k not in kept]


def load_skeleton(path, max_bodies=2, duplicate_single=True):
    """-> dict with xyz (3, T, 25, 2) camera-space metres, color (2, T, 25, 2) RGB pixel coords,
    state (T, 25, 2) Kinect tracking state (0 = no body in that frame), counts.

    Up to two actors chosen by the official NTU denoising (select_actors). Frames where an actor is not
    tracked stay zero. With duplicate_single, a sequence with a single actor repeats it in the second
    slot, as the official preprocessing (align_frames) does for CTR-GCN.
    """
    assert max_bodies == 2, 'the official NTU actor selection is defined for two actors'
    num_frames, bodies, max_per_frame = read_skeleton_file(path)
    xyz = np.zeros((3, num_frames, NUM_JOINTS, 2), np.float32)
    color = np.zeros((2, num_frames, NUM_JOINTS, 2), np.float32)
    state = np.zeros((num_frames, NUM_JOINTS, 2), np.uint8)
    actors, noise = select_actors(bodies) if bodies else ([[], []], [])
    for m, ids in enumerate(actors):
        for k in ids:  # later bodies overwrite a shared boundary frame, as in the official code
            b = bodies[k]
            # select actor m first (view), then frames; [:, frames, :, m] would move the frame axis to the front
            xyz[..., m][:, b['frames']] = np.stack(b['xyz']).transpose(2, 0, 1)
            color[..., m][:, b['frames']] = np.stack(b['color']).transpose(2, 0, 1)
            state[..., m][b['frames']] = np.stack(b['state'])
    num_actors = sum(bool(a) for a in actors)
    if duplicate_single and num_actors == 1:
        xyz[..., 1], color[..., 1], state[..., 1] = xyz[..., 0], color[..., 0], state[..., 0]
    return {'xyz': xyz, 'color': color, 'state': state, 'num_frames': num_frames, 'num_actors': num_actors,
            'actor_bodies': actors, 'noise_bodies': noise,
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
    if s['noise_bodies']:
        issues.append(f'noise_bodies_removed({len(s["noise_bodies"])})')
    empty = int((~s['xyz'].any(axis=(0, 2, 3))).sum())
    if s['num_bodies_total'] and empty:
        issues.append(f'frames_without_body({empty}/{s["num_frames"]})')
    return s, issues


def is_fatal(issues):
    return any(i.startswith(FATAL_ISSUES) for i in issues)
