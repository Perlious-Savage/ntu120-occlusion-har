"""Baseline model wrappers. Both return logits, or (features, logits) with return_features=True,
where features are the activations that feed the final classifier (used later for fusion)."""
import torch
import torch.nn as nn
from torchvision.models.video import R3D_18_Weights, r3d_18

from .ctrgcn import ntu_rgb_d  # noqa: F401  CTR-GCN resolves its graph class by dotted name; must be imported
from .ctrgcn.ctrgcn import Model as CTRGCNModel

# Kinetics-400 statistics used by torchvision's pretrained video models
KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)


class R3D18(nn.Module):
    """torchvision R3D-18 with a 120-way classifier. Input: B x 3 x T x H x W in [0, 1]."""

    def __init__(self, num_classes=120, pretrained=True):
        super().__init__()
        self.backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1 if pretrained else None)
        self.fc = nn.Linear(self.backbone.fc.in_features, num_classes)  # 512 -> 120
        self.backbone.fc = nn.Identity()
        self.register_buffer('mean', torch.tensor(KINETICS_MEAN).view(1, 3, 1, 1, 1))
        self.register_buffer('std', torch.tensor(KINETICS_STD).view(1, 3, 1, 1, 1))

    def forward(self, x, return_features=False):
        features = self.backbone((x - self.mean) / self.std)
        logits = self.fc(features)
        return (features, logits) if return_features else logits


class CTRGCN(nn.Module):
    """Official CTR-GCN (joint stream). Input: B x 3 x T x 25 x M."""

    def __init__(self, num_classes=120, num_person=2):
        super().__init__()
        self.net = CTRGCNModel(num_class=num_classes, num_point=25, num_person=num_person,
                               graph='src.models.ctrgcn.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})
        self.fc = self.net.fc  # keep the official initialisation; the network now ends at the pooled features
        self.net.fc = nn.Identity()

    def forward(self, x, return_features=False):
        features = self.net(x)  # 256-d, mean-pooled over time, joints and persons
        logits = self.fc(features)
        return (features, logits) if return_features else logits


def build_model(cfg):
    m = cfg['model']
    if m['name'] == 'r3d18':
        return R3D18(cfg['num_classes'], m.get('pretrained', True))
    if m['name'] == 'ctrgcn':
        return CTRGCN(cfg['num_classes'], cfg['input']['max_bodies'])
    raise ValueError(f'unknown model {m["name"]}')
