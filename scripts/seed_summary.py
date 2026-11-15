"""Fusion-head seed variance (review item A4): mean +- SD over head seeds 42, 1, 2, 3, 4 on frozen features.

Per protocol x setting x method: clean top-1, mCA over the 30 single-modality conditions, and the number of
conditions where fusion is below the healthy (uncorrupted) modality by more than 0.5 points. Average fusion
has no trainable parameters, so it has no seed variance.
Reads  outputs/m3_fusion{,_noaux,_trainfeat}/<method>_<protocol>_seed<k>/summary.csv
       outputs/m3_robustness/<branch>/test/summary.csv (healthy-modality accuracy)
Writes results/milestone3/seed_variance.csv and seed_variance.md
Usage: python scripts/seed_summary.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.corruptions.registry import RGB, SKELETON  # noqa: E402

ROB, OUT = Path('outputs/m3_robustness'), Path('results/milestone3')
SETTINGS = {'val_aux': 'outputs/m3_fusion', 'val_noaux': 'outputs/m3_fusion_noaux', 'trainfeat': 'outputs/m3_fusion_trainfeat'}
SEEDS = (42, 1, 2, 3, 4)
MARGIN = 0.5


def healthy(protocol):
    """{(rgb condition, skeleton condition): accuracy of the uncorrupted modality alone}."""
    r = pd.read_csv(ROB / f'r3d18_{protocol}_seed42/test/summary.csv')
    k = pd.read_csv(ROB / f'ctrgcn_{protocol}_seed42/test/summary.csv')
    acc = lambda d, c, s: float(d[(d.corruption == c) & (d.severity == s)].top1_accuracy.iloc[0])  # noqa: E731
    out = {}
    for c in RGB:
        for s in range(1, 6):
            out[(c, s, 'clean', 0)] = acc(k, 'clean', 0)
    for c in SKELETON:
        for s in range(1, 6):
            out[('clean', 0, c, s)] = acc(r, 'clean', 0)
    return out


def main():
    rows = []
    for p in ('xsub', 'xset'):
        h = healthy(p)
        for setting, d in SETTINGS.items():
            for m in ('concat', 'global', 'attention'):
                for seed in SEEDS:
                    f = Path(d) / f'{m}_{p}_seed{seed}' / 'summary.csv'
                    if not f.exists():
                        continue
                    t = pd.read_csv(f).set_index(['rgb_corruption', 'rgb_severity', 'skeleton_corruption', 'skeleton_severity'])
                    acc = t.top1_accuracy
                    single = [acc[k] for k in h]
                    below = sum((acc[k] - h[k]) * 100 < -MARGIN for k in h)
                    rows.append({'protocol': p, 'setting': setting, 'method': m, 'seed': seed,
                                 'clean': acc[('clean', 0, 'clean', 0)] * 100, 'mCA': np.mean(single) * 100,
                                 'below_healthy': below})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / 'seed_variance.csv', index=False)
    g = df.groupby(['protocol', 'setting', 'method'])
    s = g.agg(seeds=('seed', 'count'), clean_mean=('clean', 'mean'), clean_sd=('clean', 'std'),
              mCA_mean=('mCA', 'mean'), mCA_sd=('mCA', 'std'),
              below_mean=('below_healthy', 'mean'), below_min=('below_healthy', 'min'),
              below_max=('below_healthy', 'max')).reset_index()
    md = ['# Fusion-head seed variance (review item A4)', '',
          f'Head seeds {", ".join(map(str, SEEDS))} on the same frozen branch features. mCA = mean top-1 (%) over the '
          f'30 single-modality conditions. "Below healthy" = conditions where fusion is more than {MARGIN} points '
          'below the uncorrupted modality alone (oracle reference).', '',
          '| Protocol | Setting | Method | Seeds | Clean (mean ± SD) | mCA (mean ± SD) | Below healthy (mean, min-max) |',
          '|---|---|---|---|---|---|---|']
    for _, r in s.iterrows():
        md.append(f'| {r.protocol} | {r.setting} | {r.method} | {r.seeds} | {r.clean_mean:.2f} ± {r.clean_sd:.2f} | '
                  f'{r.mCA_mean:.2f} ± {r.mCA_sd:.2f} | {r.below_mean:.1f} ({r.below_min}-{r.below_max}) |')
    (OUT / 'seed_variance.md').write_text('\n'.join(md), encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
