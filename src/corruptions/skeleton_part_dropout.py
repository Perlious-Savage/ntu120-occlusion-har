"""Skeleton body-part dropout: whole joint groups removed in every frame, for every person.

Groups (Kinect v2 / NTU joints, 0-based): left arm 4-7, 21, 22; right arm 8-11, 23, 24;
left leg 12-15; right leg 16-19; upper body = both arms + spine middle, neck, head, spine
shoulder (1, 2, 3, 20); lower body = both legs + spine base (0).
Severity removes increasing structure (joints lost of 25):
  S1 one leg (4)   S2 one arm (6)   S3 lower body (9)   S4 upper body (16)   S5 upper body + one leg (20)
Which leg / arm is chosen at random (seeded).
"""
from .base import Corruption, skeleton_slots

LEFT_ARM = [4, 5, 6, 7, 21, 22]
RIGHT_ARM = [8, 9, 10, 11, 23, 24]
LEFT_LEG = [12, 13, 14, 15]
RIGHT_LEG = [16, 17, 18, 19]
TORSO_HEAD = [1, 2, 3, 20]
UPPER_BODY = LEFT_ARM + RIGHT_ARM + TORSO_HEAD
LOWER_BODY = [0] + LEFT_LEG + RIGHT_LEG


class SkeletonPartDropout(Corruption):
    name = 'skeleton_part_dropout'
    modality = 'skeleton'

    def _apply(self, x, level, rng, **_):
        leg = LEFT_LEG if rng.random() < 0.5 else RIGHT_LEG
        arm = LEFT_ARM if rng.random() < 0.5 else RIGHT_ARM
        joints = {'one_leg': leg, 'one_arm': arm, 'lower_body': LOWER_BODY,
                  'upper_body': UPPER_BODY, 'upper_body_one_leg': UPPER_BODY + leg}[level]
        x[:, :, joints, :] = 0
        return x
