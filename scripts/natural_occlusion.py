"""Natural occlusion: clean test accuracy by how much of the skeleton Kinect could not see.

Kinect marks a joint "inferred" when the camera did not see it (occluded by the body itself, the other
person or an object). data/metadata/joint_tracking.csv (scripts/joint_visibility.py) gives the inferred
share of each sample's joint observations. Nothing is synthetic here: the clean test predictions of every
method are grouped by that share.

Inferred share depends on the action (sitting, lying, two-person contact), so a raw split mixes occlusion
with class difficulty. The second table controls for that: each class is split at its own median share,
and the low/high-occlusion accuracies are averaged over classes.

Writes results/milestone3/natural_occlusion.csv and natural_occlusion.md.
Usage: python scripts/natural_occlusion.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m3_report import FUSION, FUSION_SOURCE, LABEL, OUT, PROTOCOLS, ROB, md_table  # noqa: E402

BINS = 5


def predictions(p):
    """sample_id x method -> correct (clean test), for the branches and every fusion method found."""
    src = {'rgb_only': ROB / f'r3d18_{p}_seed42/test/clean', 'skeleton_only': ROB / f'ctrgcn_{p}_seed42/test/clean'}
    src.update({m: Path(FUSION_SOURCE[m]) / f'{m}_{p}_seed42/test/clean+clean' for m in FUSION})
    cols = {m: pd.read_csv(d / 'predictions.csv').set_index('sample_id').correct for m, d in src.items()
            if (d / 'predictions.csv').exists()}
    y = pd.read_csv(src['rgb_only'] / 'predictions.csv').set_index('sample_id').true_label
    return pd.DataFrame(cols).assign(label=y)


def main():
    vis = pd.read_csv('data/metadata/joint_tracking.csv').set_index('sample_id').inferred_frac
    rows, md = [], ['# Natural occlusion (Kinect inferred joints), clean test split', '',
                    'Inferred share = fraction of the actors\' joint observations that Kinect could not see. '
                    'Top-1 accuracy (%).', '']
    for p in PROTOCOLS:
        d = predictions(p)
        d['inferred'] = vis.loc[d.index].to_numpy()
        methods = [m for m in ['rgb_only', 'skeleton_only'] + FUSION if m in d]
        d['bin'] = pd.qcut(d.inferred, BINS, labels=False)
        for b, g in d.groupby('bin'):
            rows.append({'protocol': p, 'split': f'quintile {b + 1}', 'inferred_min': g.inferred.min(),
                         'inferred_max': g.inferred.max(), 'samples': len(g), **{m: g[m].mean() for m in methods}})
        # within-class: split each class at its own median, average the per-class accuracies
        hi = d.inferred > d.groupby('label').inferred.transform('median')
        for name, mask in (('within-class low', ~hi), ('within-class high', hi)):
            g = d[mask]
            rows.append({'protocol': p, 'split': name, 'inferred_min': g.inferred.min(), 'inferred_max': g.inferred.max(),
                         'samples': len(g), **{m: g.groupby('label')[m].mean().mean() for m in methods}})
        t = pd.DataFrame([r for r in rows if r['protocol'] == p])
        t['range'] = [f'{a:.2f}-{b:.2f}' for a, b in zip(t.inferred_min, t.inferred_max)]
        show = t.assign(**{m: (t[m] * 100).round(2) for m in methods})
        md += [f'## {p.upper()}', '', md_table(show, ['split', 'range', 'samples'] + methods,
                                                 ['Split', 'Inferred share', 'Samples'] + [LABEL[m] for m in methods]), '']
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / 'natural_occlusion.csv', index=False)
    (OUT / 'natural_occlusion.md').write_text('\n'.join(md), encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
