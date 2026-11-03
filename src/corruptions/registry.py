"""Name -> corruption instance, for configs and scripts."""
from .rgb_blur import RGBBlur
from .rgb_occlusion import RGBOcclusion
from .rgb_temporal_mask import RGBTemporalMask
from .skeleton_joint_dropout import SkeletonJointDropout
from .skeleton_noise import SkeletonNoise
from .skeleton_part_dropout import SkeletonPartDropout

CORRUPTIONS = {c.name: c for c in (RGBOcclusion(), RGBBlur(), RGBTemporalMask(),
                                   SkeletonJointDropout(), SkeletonPartDropout(), SkeletonNoise())}
RGB = [n for n, c in CORRUPTIONS.items() if c.modality == 'rgb']
SKELETON = [n for n, c in CORRUPTIONS.items() if c.modality == 'skeleton']


def get(name):
    if name not in CORRUPTIONS:
        raise KeyError(f'unknown corruption {name!r}; known: {sorted(CORRUPTIONS)}')
    return CORRUPTIONS[name]
