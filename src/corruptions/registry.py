"""Name -> corruption instance. S1-S5 levels come from configs/corruptions.yaml (single source)."""
from pathlib import Path

import yaml

from .coupled_occlusion import SkeletonCoupledInterp, SkeletonCoupledOcclusion
from .rgb_blur import RGBBlur
from .rgb_occlusion import RGBOcclusion
from .rgb_temporal_mask import RGBTemporalFreeze, RGBTemporalMask
from .skeleton_joint_dropout import SkeletonJointDropout, SkeletonJointInterp
from .skeleton_noise import SkeletonNoise
from .skeleton_part_dropout import SkeletonPartDropout

CONFIG = Path(__file__).resolve().parents[2] / 'configs/corruptions.yaml'
_cfg = yaml.safe_load(open(CONFIG))

CORRUPTIONS = {}
for cls in (RGBOcclusion, RGBBlur, RGBTemporalMask, SkeletonJointDropout, SkeletonPartDropout, SkeletonNoise,
            SkeletonCoupledOcclusion, RGBTemporalFreeze, SkeletonJointInterp, SkeletonCoupledInterp):
    c = cls()
    c.levels = tuple(_cfg[c.name]['levels'])
    assert len(c.levels) == 5, f'{c.name}: need 5 levels (S1-S5)'
    CORRUPTIONS[c.name] = c
RGB = [n for n, c in CORRUPTIONS.items() if c.modality == 'rgb' and not c.extra]
SKELETON = [n for n, c in CORRUPTIONS.items() if c.modality == 'skeleton' and not c.extra]
EXTRA = [n for n, c in CORRUPTIONS.items() if c.extra]   # evaluated separately (--corruption)


def get(name):
    if name not in CORRUPTIONS:
        raise KeyError(f'unknown corruption {name!r}; known: {sorted(CORRUPTIONS)}')
    return CORRUPTIONS[name]
