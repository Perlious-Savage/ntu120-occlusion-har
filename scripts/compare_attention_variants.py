"""Attention-fusion design choice (Milestone 3): 5-fold CV on the clean validation features only.
No test data is used.

A  mix the projected features, one shared classifier (+ auxiliary branch losses)
B  A with the attention output layer zero-initialised (starts exactly at 0.5 / 0.5)
C  mix the logits of refitted branch heads U h (trained by the auxiliary losses)
D  mix the FROZEN branch log-probabilities; the gate reads the projected features.
   This is FusionHead('attention').

Measured (seed 42, CV top-1; RGB weight mean / per-fold collapse):
  X-Sub  A 0.932  B 0.933 (both 1.000 in every fold)   C 0.965 (0.54)          D 0.923 (0.45-0.61)
  X-Set  A 0.929  B 0.928 (both 1.000 in every fold)   C 0.938 (3 of 5 folds 1.000)   D 0.966 (0.49-0.57)
A-C refit classifiers on the same samples the gate learns from: the refitted RGB head is right on
almost all of them, so the gate saturates onto RGB. C's X-Sub CV is also optimistic (test 0.883).
D keeps the frozen backbones, whose errors on the validation split are genuine; it beats average
fusion in CV on both protocols (X-Sub 0.923 vs 0.906, X-Set 0.966 vs 0.958) without collapsing.
Usage: python scripts/compare_attention_variants.py   (needs outputs/m3_robustness/<run>/val/clean)
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import train_fusion as tf  # noqa: E402
from src.models.fusion import FusionHead, projection  # noqa: E402


class RefitAttention(nn.Module):
    """Variants A/B (mix projected features) and C (mix refitted branch logits)."""

    def __init__(self, logit_level, d_rgb, d_sk, d, n, p, zero_init):
        super().__init__()
        self.logit_level = logit_level
        self.proj_rgb, self.proj_sk = projection(d_rgb, d, p), projection(d_sk, d, p)
        self.cls, self.aux_rgb, self.aux_sk = nn.Linear(d, n), nn.Linear(d, n), nn.Linear(d, n)
        self.att = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 2))
        if zero_init:
            nn.init.zeros_(self.att[-1].weight); nn.init.zeros_(self.att[-1].bias)

    def forward(self, f_rgb, f_sk, aux=False):
        h_rgb, h_sk = self.proj_rgb(f_rgb), self.proj_sk(f_sk)
        z_rgb, z_sk = self.aux_rgb(h_rgb), self.aux_sk(h_sk)
        w = self.att(torch.cat([h_rgb, h_sk], 1)).softmax(1)
        logits = (w[:, :1] * z_rgb + w[:, 1:] * z_sk) if self.logit_level else self.cls(w[:, :1] * h_rgb + w[:, 1:] * h_sk)
        return (logits, w, (z_rgb, z_sk)) if aux else (logits, w)


def builder(variant):
    def build(method, d_rgb, d_sk, d, n, p):
        if variant == 'D':
            return FusionHead('attention', d_rgb, d_sk, d, n, p)
        return RefitAttention(variant == 'C', d_rgb, d_sk, d, n, p, zero_init=variant != 'A')
    return build


def main():
    cfg = yaml.safe_load(open(ROOT / 'configs/fusion.yaml'))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for p in ['xsub', 'xset']:
        c = dict(cfg, protocol=p, method='attention', features_dir=str(ROOT / cfg['features_dir']))
        feats = tf.Features(c, p, 'val')
        for variant in 'ABCD':
            fr, fs, pr, ps, y, ids = feats.pair('clean', 'clean', with_logp=variant == 'D')
            if variant == 'A':
                print(f'{p}: RGB val {(pr.argmax(1) == y).mean():.4f}  skeleton val {(ps.argmax(1) == y).mean():.4f}  '
                      f'average fusion {((pr + ps).argmax(1) == y).mean():.4f}')
            tf.FusionHead = builder(variant)                  # fit() builds its head through this name
            acc, per = [], []
            for a, b in tf.cv_folds(ids, y, cfg['cv_folds'], cfg['seed']):
                log, head, _ = tf.fit(c, 'attention', (fr[a], fs[a], y[a]), (fr[b], fs[b], y[b]), dev)
                acc.append([r['val_accuracy'] for r in log])
                per.append(float(tf.infer(head, fr[b], fs[b], dev)[1][:, 0].mean()))
            acc = np.array(acc)
            e = int(acc.mean(0).argmax())
            print(f'  {variant}: CV top-1 {acc.mean(0)[e]:.4f} (epoch {e})  mean RGB weight per fold '
                  + ' '.join(f'{w:.2f}' for w in per), flush=True)


if __name__ == '__main__':
    main()
