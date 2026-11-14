"""Milestone 3 diagnostic with uncertainty (review item A4): is fusion really below the healthy modality?

For every single-modality condition and fusion method: signed gap Acc(fusion) - Acc(healthy) in points,
a 95% paired bootstrap CI (multinomial over the per-sample differences, exact for a paired mean), an
exact McNemar test on the discordant pairs, Holm-corrected over all cells of a protocol. "Healthy" is an
ORACLE selector (it knows which modality is corrupted); two more references bound the headroom:
  either_correct   per-sample oracle: right if either branch is right (upper bound for any late fusion)
  max_confidence   non-oracle selector: per sample, the branch with the higher top probability

Reads  outputs/m3_robustness/<branch>/test/<condition>/predictions.csv, probabilities.npy
       <fusion dir>/<method>_<protocol>_seed42/test/<rgb cond>+<skeleton cond>/predictions.csv
Writes results/milestone3/stats_cells.csv, stats_references.csv, stats.md
Usage: python scripts/m3_stats.py [--fusion-dir outputs/m3_fusion] [--margin 0.5]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m3_report import FUSION, LABEL, NICE, OUT, PROTOCOLS, ROB  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.corruptions.base import SEVERITIES  # noqa: E402
from src.corruptions.registry import RGB, SKELETON  # noqa: E402

B = 10000


def holm(p):
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    out = np.empty_like(adj)
    out[order] = adj
    return out


def paired(a, h, rng):
    """a, h: boolean correctness of fusion and reference on the same samples ->
    (gap pts, ci low, ci high, b = fusion-only right, c = reference-only right, McNemar p)."""
    d = a.astype(int) - h.astype(int)
    n, b, c = len(d), int((d > 0).sum()), int((d < 0).sum())
    draws = rng.multinomial(n, [b / n, c / n, 1 - (b + c) / n], size=B)   # resampling d is multinomial
    boot = (draws[:, 0] - draws[:, 1]) / n * 100
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return d.mean() * 100, *np.percentile(boot, [2.5, 97.5]), b, c, p


def branch(protocol, modality, condition):
    run = f'{"r3d18" if modality == "rgb" else "ctrgcn"}_{protocol}_seed42'
    d = ROB / run / 'test' / condition
    pred = pd.read_csv(d / 'predictions.csv')
    return pred.correct.to_numpy(bool), np.load(d / 'probabilities.npy').astype(np.float32).max(1), pred.sample_id.to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fusion-dir', default='outputs/m3_fusion', help='one training setting for all heads')
    ap.add_argument('--margin', type=float, default=0.5, help='practical margin in points')
    args = ap.parse_args()
    rng = np.random.default_rng(42)
    cells, refs = [], []
    for p in PROTOCOLS:
        conds = [(f'{c}_s{s}', 'clean', 'skeleton_only', c, s) for c in RGB for s in SEVERITIES]
        conds += [('clean', f'{c}_s{s}', 'rgb_only', c, s) for c in SKELETON for s in SEVERITIES]
        for rc, sc, healthy, c, s in conds:
            r_ok, r_conf, ids = branch(p, 'rgb', rc)
            s_ok, s_conf, ids2 = branch(p, 'skeleton', sc)
            assert (ids == ids2).all()
            h_ok = s_ok if healthy == 'skeleton_only' else r_ok
            refs.append({'protocol': p, 'corruption': c, 'severity': s, 'healthy_oracle': h_ok.mean() * 100,
                         'either_correct': (r_ok | s_ok).mean() * 100,
                         'max_confidence': np.where(r_conf >= s_conf, r_ok, s_ok).mean() * 100})
            for m in FUSION:
                f = Path(args.fusion_dir) / f'{m}_{p}_seed42' / 'test' / f'{rc}+{sc}' / 'predictions.csv'
                fp = pd.read_csv(f)
                assert (fp.sample_id.to_numpy() == ids).all(), f
                gap, lo, hi, b, cc, pv = paired(fp.correct.to_numpy(bool), h_ok, rng)
                cells.append({'protocol': p, 'corruption': c, 'severity': s, 'method': m, 'fusion': fp.correct.mean() * 100,
                              'healthy': h_ok.mean() * 100, 'gap': gap, 'ci_low': lo, 'ci_high': hi,
                              'fusion_only_right': b, 'healthy_only_right': cc, 'p_mcnemar': pv})
    cells, refs = pd.DataFrame(cells), pd.DataFrame(refs)
    cells['p_holm'] = cells.groupby('protocol').p_mcnemar.transform(holm)
    cells['below_sig'] = (cells.gap < 0) & (cells.p_holm < 0.05)
    cells['below_margin_sig'] = cells.below_sig & (cells.ci_high < -args.margin)
    OUT.mkdir(parents=True, exist_ok=True)
    cells.to_csv(OUT / 'stats_cells.csv', index=False)
    refs.to_csv(OUT / 'stats_references.csv', index=False)

    md = ['# Fusion vs the healthy modality, with uncertainty (review item A4)', '',
          f'Fusion heads from `{args.fusion_dir}` (one training setting for every head). Gap = Acc(fusion) - '
          'Acc(healthy), points; 95% paired bootstrap CI; exact McNemar, Holm over the 120 cells of a protocol. '
          f'"Healthy" is an oracle selector. Single seed (head seeds pending).', '']
    for p in PROTOCOLS:
        g = cells[cells.protocol == p]
        md += [f'## {p.upper()}', '', '| Method | below (raw) | below, Holm p<0.05 | below by > '
               f'{args.margin} pt (CI), Holm p<0.05 | mean gap | worst gap (95% CI) |', '|---|---|---|---|---|---|']
        for m in FUSION:
            q = g[g.method == m]
            w = q.loc[q.gap.idxmin()]
            md.append(f'| {LABEL[m]} | {int((q.gap < 0).sum())} | {int(q.below_sig.sum())} | {int(q.below_margin_sig.sum())} | '
                      f'{q.gap.mean():+.2f} | {w.gap:+.2f} ({w.ci_low:+.2f}, {w.ci_high:+.2f}) '
                      f'{NICE[w.corruption]} S{w.severity} |')
        r = refs[refs.protocol == p]
        avg = g[g.method == 'average'].set_index(['corruption', 'severity']).fusion
        rr = r.set_index(['corruption', 'severity'])
        md += ['', 'Headroom (mean over the 30 conditions, %):', '',
               '| Average fusion | Healthy (oracle selector) | Max-confidence selector | Either branch correct (oracle) |',
               '|---|---|---|---|',
               f'| {avg.mean():.2f} | {rr.healthy_oracle.mean():.2f} | {rr.max_confidence.mean():.2f} | '
               f'{rr.either_correct.mean():.2f} |', '']
        for m in FUSION:
            f = g[g.method == m].set_index(['corruption', 'severity']).fusion
            closed = ((f - avg) / (rr.either_correct - avg)).mean() * 100
            md.append(f'- {LABEL[m]}: closes {closed:+.1f}% of the average-to-either-correct gap on average.')
        md.append('')
    (OUT / 'stats.md').write_text('\n'.join(md), encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
