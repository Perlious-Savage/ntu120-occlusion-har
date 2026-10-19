"""Final results table and class-level RGB vs skeleton comparison (Milestone 2, steps 15-16).

Reads outputs/<model>_<protocol>_seed<seed>/{test_metrics.json, per_class_accuracy.csv} and writes
outputs/baseline_results.md plus, per protocol, outputs/class_comparison_<protocol>.csv.
All conclusions come from measured per-class test accuracy; nothing is assumed per class.
Usage: python scripts/compare_baselines.py [--seed 42] [--margin 0.10]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

OUT = Path('outputs')
MODELS = {'r3d18': 'R3D-18', 'ctrgcn': 'CTR-GCN'}
PROTOCOLS = {'xsub': 'X-Sub', 'xset': 'X-Set'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--margin', type=float, default=0.10, help='accuracy gap that counts as "clearly better"')
    args = ap.parse_args()

    lines = ['# Baseline results (clean NTU RGB+D 120, test split, best-val-accuracy checkpoint)', '',
             '| Model | Protocol | Top-1 Acc. | Top-5 Acc. | Macro-F1 | Best Epoch |', '|---|---|---|---|---|---|']
    for p, pname in PROTOCOLS.items():
        for m, mname in MODELS.items():
            f = OUT / f'{m}_{p}_seed{args.seed}' / 'test_metrics.json'
            if f.exists():
                r = json.load(open(f))['best_val_accuracy']
                lines.append(f'| {mname} | {pname} | {r["top1_accuracy"]:.2%} | {r["top5_accuracy"]:.2%} | '
                             f'{r["macro_f1"]:.4f} | {r["best_epoch"]} |')
            else:
                lines.append(f'| {mname} | {pname} | not run | | | |')

    for p, pname in PROTOCOLS.items():
        runs = {m: OUT / f'{m}_{p}_seed{args.seed}' / 'per_class_accuracy.csv' for m in MODELS}
        if not all(f.exists() for f in runs.values()):
            continue
        rgb, sk = (pd.read_csv(runs[m]).set_index('action_id') for m in MODELS)
        cmp = pd.DataFrame({'support': rgb.support, 'rgb_accuracy': rgb.accuracy, 'skeleton_accuracy': sk.accuracy})
        cmp['rgb_minus_skeleton'] = cmp.rgb_accuracy - cmp.skeleton_accuracy
        cmp.to_csv(OUT / f'class_comparison_{p}.csv')

        def table(df, cols, title):
            lines.extend(['', f'### {pname}: {title}', '', '| Action ID | ' + ' | '.join(cols) + ' |',
                          '|---' * (len(cols) + 1) + '|'])
            for aid, r in df.iterrows():
                lines.append(f'| A{aid:03d} | ' + ' | '.join(f'{r[c]:.1%}' if isinstance(r[c], float) else str(r[c])
                                                           for c in cols) + ' |')

        cols = ['rgb_accuracy', 'skeleton_accuracy', 'rgb_minus_skeleton']
        table(cmp.nsmallest(20, 'rgb_accuracy'), cols, '20 weakest classes for RGB (R3D-18)')
        table(cmp.nsmallest(20, 'skeleton_accuracy'), cols, '20 weakest classes for skeleton (CTR-GCN)')
        table(cmp[cmp.rgb_minus_skeleton >= args.margin].sort_values('rgb_minus_skeleton', ascending=False), cols,
              f'RGB clearly better (by >= {args.margin:.0%})')
        table(cmp[cmp.rgb_minus_skeleton <= -args.margin].sort_values('rgb_minus_skeleton'), cols,
              f'skeleton clearly better (by >= {args.margin:.0%})')

    (OUT / 'baseline_results.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
