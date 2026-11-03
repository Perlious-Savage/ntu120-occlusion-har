"""Attention-fusion design choice (Milestone 3): 5-fold CV on the clean validation features only.
No test data is used.

A  mix the projected features, one shared classifier (+ auxiliary branch losses)
B  A with the attention output layer zero-initialised (starts exactly at 0.5 / 0.5)
C  logit-level: logits = w_rgb * U_rgb h_rgb + w_sk * U_sk h_sk (the branch heads trained by the
   auxiliary losses); this is FusionHead('attention')

Measured (seed 42): A and B saturate onto RGB (weight 1.000 for every sample) on both protocols;
C reaches CV top-1 0.965 (X-Sub) / 0.938 (X-Set) with sample-specific weights.
Usage: python scripts/compare_attention_variants.py   (needs outputs/m3_robustness/<run>/val/clean)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import train_fusion as tf  # noqa: E402
from src.models.fusion import FusionHead  # noqa: E402

FEATURES = ROOT / 'outputs/m3_robustness'


class FeatureAttention(FusionHead):
    """Variants A/B: mix the projected features, one shared classifier."""

    def forward(self, f_rgb, f_sk, aux=False):
        h_rgb, h_sk = self.proj_rgb(f_rgb), self.proj_sk(f_sk)
        w = self.att(torch.cat([h_rgb, h_sk], 1)).softmax(1)
        logits = self.cls(w[:, :1] * h_rgb + w[:, 1:] * h_sk)
        return (logits, w, (self.aux_rgb(h_rgb), self.aux_sk(h_sk))) if aux else (logits, w)


def builder(variant):
    def build(method, d_rgb, d_sk, d, n, p):
        h = (FusionHead if variant == 'C' else FeatureAttention)('attention', d_rgb, d_sk, d, n, p)
        if variant in ('B', 'C'):
            nn.init.zeros_(h.att[-1].weight); nn.init.zeros_(h.att[-1].bias)
        return h
    return build


def main():
    cfg = yaml.safe_load(open(ROOT / 'configs/fusion.yaml'))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for p in ['xsub', 'xset']:
        d = {r: FEATURES / f'{r}_{p}_seed42/val/clean' for r in ('r3d18', 'ctrgcn')}
        fr = np.load(d['r3d18'] / 'features.npy').astype(np.float32)
        fs = np.load(d['ctrgcn'] / 'features.npy').astype(np.float32)
        pr, ps = pd.read_csv(d['r3d18'] / 'predictions.csv'), pd.read_csv(d['ctrgcn'] / 'predictions.csv')
        assert (pr.sample_id == ps.sample_id).all()
        y = pr.true_label.to_numpy()
        print(f'{p}: RGB val {pr.correct.mean():.4f}  skeleton val {ps.correct.mean():.4f}  '
              f'either right {(pr.correct | ps.correct).mean():.4f}')
        folds = tf.cv_folds(pr.sample_id.to_numpy(), y, cfg['cv_folds'], cfg['seed'])
        for variant in 'ABC':
            tf.FusionHead = builder(variant)                  # fit() builds its head through this name
            c = dict(cfg, protocol=p, method='attention')
            acc, ws = [], []
            for a, b in folds:
                log, head, _ = tf.fit(c, 'attention', (fr[a], fs[a], y[a]), (fr[b], fs[b], y[b]), dev)
                acc.append([r['val_accuracy'] for r in log])
                ws.append(tf.infer(head, fr[b], fs[b], dev)[1][:, 0])
            acc, w = np.array(acc), np.concatenate(ws)
            e = int(acc.mean(0).argmax())
            print(f'  {variant}: CV top-1 {acc.mean(0)[e]:.4f} (epoch {e})  w_rgb mean {w.mean():.3f}  sd {w.std():.3f}  '
                  f'samples with w_rgb < 0.5: {(w < 0.5).mean():.3f}', flush=True)


if __name__ == '__main__':
    main()
