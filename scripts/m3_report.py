"""Milestone 3 tables and figures from the robustness and fusion outputs (steps 3.6, 3.9-3.13).

Reads  outputs/m3_robustness/<branch run>/test/summary.csv (+ clean per_class_accuracy.csv)
       outputs/m3_fusion/<method>_<protocol>_seed42/summary.csv
Writes results/milestone3/:
  Milestone3_results.csv           every method x protocol x condition (long format)
  mca.csv, diagnostic.csv, dual_corruption.csv
  milestone3_results.md            single-modality table (3.6), mCA / robustness, diagnostic (3.9), dual (3.10)
  class_modality_<protocol>.csv    per-class RGB vs skeleton accuracy and dominance (3.12)
  fig_*.png                        the 3.13 figures
Usage: python scripts/m3_report.py
"""
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.corruptions.registry import RGB, SKELETON  # noqa: E402

ROB, FUS, OUT = Path('outputs/m3_robustness'), Path('outputs/m3_fusion'), Path('results/milestone3')
PROTOCOLS = ['xsub', 'xset']
FUSION = ['average', 'concat', 'global', 'attention']
METHODS = ['rgb_only', 'skeleton_only'] + FUSION
LABEL = {'rgb_only': 'RGB only (R3D-18)', 'skeleton_only': 'Skeleton only (CTR-GCN)', 'average': 'Average fusion',
         'concat': 'Concat fusion', 'global': 'Global-weight fusion', 'attention': 'Attention fusion'}
# reference categorical palette, fixed order (validated: scripts/validate_palette.js, light mode)
COLOR = dict(zip(METHODS, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']))
MARKER = dict(zip(METHODS, ['o', 's', '^', 'D', 'v', 'P']))
NICE = {'rgb_occlusion': 'RGB occlusion', 'rgb_blur': 'RGB blur', 'rgb_temporal_mask': 'RGB temporal mask',
        'skeleton_joint_dropout': 'Joint dropout', 'skeleton_part_dropout': 'Body-part dropout',
        'skeleton_noise': 'Coordinate noise'}
DOMINANCE_PTS = 5.0   # |RGB - skeleton| class accuracy gap (percentage points) for "dominant"; ~2 s.e. of a difference


def branch(protocol, modality):
    run = f'{"r3d18" if modality == "rgb" else "ctrgcn"}_{protocol}_seed42'
    return pd.read_csv(ROB / run / 'test' / 'summary.csv'), ROB / run / 'test'


def long_table():
    """All methods in one long table: one row per protocol x method x (RGB condition, skeleton condition)."""
    rows = []
    for p in PROTOCOLS:
        rgb, _ = branch(p, 'rgb')
        sk, _ = branch(p, 'skeleton')
        for _, r in rgb.iterrows():        # RGB only: the skeleton is not used, label its condition clean
            rows.append({'protocol': p, 'method': 'rgb_only', 'rgb_corruption': r.corruption, 'rgb_severity': r.severity,
                         'skeleton_corruption': 'clean', 'skeleton_severity': 0, 'top1_accuracy': r.top1_accuracy,
                         'top5_accuracy': r.top5_accuracy, 'macro_f1': r.macro_f1})
        for _, r in sk.iterrows():
            rows.append({'protocol': p, 'method': 'skeleton_only', 'rgb_corruption': 'clean', 'rgb_severity': 0,
                         'skeleton_corruption': r.corruption, 'skeleton_severity': r.severity,
                         'top1_accuracy': r.top1_accuracy, 'top5_accuracy': r.top5_accuracy, 'macro_f1': r.macro_f1})
        for m in FUSION:
            f = FUS / f'{m}_{p}_seed42' / 'summary.csv'
            if f.exists():
                rows += pd.read_csv(f).assign(protocol=p, method=m).to_dict('records')
            else:
                print(f'missing {f}')
    df = pd.DataFrame(rows)
    clean = df[(df.rgb_severity == 0) & (df.skeleton_severity == 0)].set_index(['protocol', 'method']).top1_accuracy
    df['robustness_ratio'] = df.top1_accuracy / [clean[(p, m)] for p, m in zip(df.protocol, df.method)]
    return df


def acc(df, p, m, rc='clean', rs=0, sc='clean', ss=0):
    """Accuracy of a method under (RGB condition, skeleton condition). A single-modality model is
    unaffected by the other modality's corruption, so it reports its own clean accuracy there."""
    if m == 'rgb_only':
        sc, ss = 'clean', 0
    if m == 'skeleton_only':
        rc, rs = 'clean', 0
    rc, sc = (rc if rs else 'clean'), (sc if ss else 'clean')     # severity 0 = the clean condition
    q = df[(df.protocol == p) & (df.method == m) & (df.rgb_corruption == rc) & (df.rgb_severity == rs)
           & (df.skeleton_corruption == sc) & (df.skeleton_severity == ss)]
    return float(q.top1_accuracy.iloc[0]) if len(q) else np.nan


def single_conditions():
    return [(c, s, 'clean', 0) for c in RGB for s in range(1, 6)] + [('clean', 0, c, s) for c in SKELETON for s in range(1, 6)]


def mca_table(df):
    rows = []
    for p in PROTOCOLS:
        for m in METHODS:
            r = [acc(df, p, m, rc, rs, sc, ss) for rc, rs, sc, ss in single_conditions()]
            clean = acc(df, p, m)
            rows.append({'protocol': p, 'method': m, 'clean': clean, 'mCA_rgb_corrupt': np.mean(r[:15]),
                         'mCA_skeleton_corrupt': np.mean(r[15:]), 'mCA_all': np.mean(r),
                         'robustness_ratio_all': np.mean(r) / clean})
    return pd.DataFrame(rows)


def diagnostic(df):
    """Step 3.9: for every single-modality corruption, fusion vs the healthy (uncorrupted) modality."""
    rows = []
    for p in PROTOCOLS:
        for rc, rs, sc, ss in single_conditions():
            healthy = 'skeleton_only' if rs else 'rgb_only'
            row = {'protocol': p, 'corruption': rc if rs else sc, 'severity': rs or ss, 'healthy_modality': healthy}
            for m in METHODS:
                row[m] = acc(df, p, m, rc, rs, sc, ss)
            for m in FUSION:
                row[f'{m}_below_healthy'] = row[m] < row[healthy]
            rows.append(row)
    return pd.DataFrame(rows)


def class_modality(p):
    _, rd = branch(p, 'rgb')
    _, sd = branch(p, 'skeleton')
    names = pd.read_csv('data/metadata/ntu120_action_names.csv')
    r = pd.read_csv(rd / 'clean/per_class_accuracy.csv')
    s = pd.read_csv(sd / 'clean/per_class_accuracy.csv')
    d = names.merge(r[['label', 'support', 'accuracy']].rename(columns={'accuracy': 'rgb_accuracy'}), on='label')
    d = d.merge(s[['label', 'accuracy']].rename(columns={'accuracy': 'skeleton_accuracy'}), on='label')
    d['rgb_minus_skeleton_pts'] = 100 * (d.rgb_accuracy - d.skeleton_accuracy)
    d['dominance'] = np.select([d.rgb_minus_skeleton_pts >= DOMINANCE_PTS, d.rgb_minus_skeleton_pts <= -DOMINANCE_PTS],
                               ['RGB-dominant', 'skeleton-dominant'], 'balanced')
    return d.sort_values('rgb_minus_skeleton_pts', ascending=False)


def pct(x):
    return '' if pd.isna(x) else f'{100 * x:.2f}'


def md_table(df, cols, header):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(str(r[c]) for c in cols) + ' |' for _, r in df.iterrows()]
    return '\n'.join(lines)


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, loc='left')
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis='y', color='#e6e5e0', lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


def severity_figure(df, modality, methods, path, title):
    names = RGB if modality == 'rgb' else SKELETON
    fig, axes = plt.subplots(len(PROTOCOLS), 3, figsize=(13, 7.2), sharey='row')
    for i, p in enumerate(PROTOCOLS):
        for j, c in enumerate(names):
            ax = axes[i, j]
            for m in methods:
                y = [100 * acc(df, p, m, *((c, s, 'clean', 0) if modality == 'rgb' else ('clean', 0, c, s)))
                     for s in range(6)]
                ax.plot(range(6), y, color=COLOR[m], marker=MARKER[m], ms=5, lw=2, label=LABEL[m])
            style(ax, f'{p.upper()}: {NICE[c]}', 'severity (0 = clean)', 'top-1 accuracy (%)' if j == 0 else '')
            ax.set_xticks(range(6))
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=8, frameon=False, loc='lower center', ncol=len(methods))
    fig.suptitle(title, fontsize=11, x=0.01, ha='left'); fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150); plt.close(fig)


def class_scatter(cls, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, p in zip(axes, PROTOCOLS):
        d = cls[p]
        ax.plot([0, 100], [0, 100], color='#a3a29c', lw=1, ls='--')
        ax.scatter(100 * d.skeleton_accuracy, 100 * d.rgb_accuracy, s=22, color=COLOR['rgb_only'],
                   edgecolor='#fcfcfb', linewidth=1)
        for _, r in pd.concat([d.head(4), d.tail(4)]).iterrows():   # largest gaps either way
            ax.annotate(r['name'], (100 * r.skeleton_accuracy, 100 * r.rgb_accuracy), fontsize=7, color='#52514e',
                        xytext=(4, -3), textcoords='offset points')
        n = d.dominance.value_counts()
        style(ax, f'{p.upper()}: {n.get("RGB-dominant", 0)} RGB-dominant, {n.get("balanced", 0)} balanced, '
                  f'{n.get("skeleton-dominant", 0)} skeleton-dominant (±{DOMINANCE_PTS:g} pts)',
              'skeleton accuracy (%)', 'RGB accuracy (%)')
        lo = 100 * min(d.rgb_accuracy.min(), d.skeleton_accuracy.min()) - 3
        ax.set_xlim(lo, 101); ax.set_ylim(lo, 101); ax.set_aspect('equal'); ax.grid(color='#e6e5e0', lw=0.8)
    fig.suptitle('Per-class clean test accuracy: RGB vs skeleton (above the line = RGB better)', fontsize=11, x=0.01, ha='left')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def heatmap(df, path):
    conds = single_conditions()
    cols = [f'{NICE[rc if rs else sc]} S{rs or ss}' for rc, rs, sc, ss in conds]
    fig, axes = plt.subplots(2, 1, figsize=(15, 6.5))
    for ax, p in zip(axes, PROTOCOLS):
        m = np.array([[100 * acc(df, p, mt, *c) for c in conds] for mt in METHODS])
        im = ax.imshow(m, cmap='Blues', vmin=0, vmax=100, aspect='auto')
        ax.set_yticks(range(len(METHODS)), [LABEL[x] for x in METHODS], fontsize=8)
        ax.set_xticks(range(len(cols)), cols if p == PROTOCOLS[-1] else [''] * len(cols), rotation=60, ha='right', fontsize=7)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f'{m[i, j]:.0f}', ha='center', va='center', fontsize=6,
                        color='#fcfcfb' if m[i, j] > 60 else '#0b0b0b')
        ax.set_title(f'{p.upper()}: top-1 accuracy (%) per corruption and severity', fontsize=10, loc='left')
        ax.axvline(14.5, color='#fcfcfb', lw=3)
    fig.colorbar(im, ax=axes, fraction=0.02, label='top-1 accuracy (%)')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = long_table()
    mca, diag = mca_table(df), diagnostic(df)
    dual = df[(df.rgb_severity > 0) & (df.skeleton_severity > 0)].copy()
    cls = {p: class_modality(p) for p in PROTOCOLS}
    df.to_csv(OUT / 'Milestone3_results.csv', index=False)
    mca.to_csv(OUT / 'mca.csv', index=False); diag.to_csv(OUT / 'diagnostic.csv', index=False)
    dual.to_csv(OUT / 'dual_corruption.csv', index=False)
    for p in PROTOCOLS:
        cls[p].to_csv(OUT / f'class_modality_{p}.csv', index=False)

    md = ['# Milestone 3 results', '',
          'Top-1 accuracy (%) on the official test splits. Frozen Milestone 2 backbones (seed 42); fusion heads trained on clean features of the held-out validation split (5-fold CV picks the stopping epoch) '
          'and tested under corruption without retraining. Corruption definitions: '
          '`docs/corruption_protocol.md`.', '']
    for p in PROTOCOLS:
        md += [f'## {p.upper()}', '', '### Single-modality robustness (3.6)', '']
        rows = []
        for m, names in (('rgb_only', RGB), ('skeleton_only', SKELETON)):
            for c in names:
                v = [acc(df, p, m, *((c, s, 'clean', 0) if m == 'rgb_only' else ('clean', 0, c, s))) for s in range(6)]
                rows.append({'model': f'{"R3D-18" if m == "rgb_only" else "CTR-GCN"}: {NICE[c]}',
                             **{k: pct(x) for k, x in zip(['Clean', 'S1', 'S2', 'S3', 'S4', 'S5'], v)},
                             'mean S1-S5': pct(np.mean(v[1:])), 'robustness ratio': f'{np.mean(v[1:]) / v[0]:.3f}'})
        r = pd.DataFrame(rows)
        md += [md_table(r, r.columns, r.columns), '']
        q = mca[mca.protocol == p].assign(**{c: lambda d, c=c: d[c].map(pct) for c in
                                             ['clean', 'mCA_rgb_corrupt', 'mCA_skeleton_corrupt', 'mCA_all']})
        q['method'] = q.method.map(LABEL); q['robustness_ratio_all'] = q.robustness_ratio_all.map('{:.3f}'.format)
        md += ['### Clean accuracy, mean corruption accuracy (mCA) and robustness ratio', '',
               'mCA = mean top-1 over S1-S5 of the three corruptions of one modality (the other modality clean); '
               '"all" averages all 30 single-modality conditions. Robustness ratio = mCA_all / clean.', '',
               md_table(q, ['method', 'clean', 'mCA_rgb_corrupt', 'mCA_skeleton_corrupt', 'mCA_all', 'robustness_ratio_all'],
                        ['Method', 'Clean', 'mCA (RGB corrupted)', 'mCA (skeleton corrupted)', 'mCA (all)', 'Robustness ratio']), '']
        d = diag[diag.protocol == p]
        n_below = {m: int(d[f'{m}_below_healthy'].sum()) for m in FUSION}
        md += ['### Diagnostic: fusion vs the healthy modality (3.9)', '',
               'Healthy = the uncorrupted modality alone. **Bold** = fusion below the healthy modality, '
               'i.e. Acc(fusion) < Acc(healthy). Conditions below healthy (of 30): '
               + ', '.join(f'{LABEL[m]} {n_below[m]}' for m in FUSION) + '.', '']
        t = d.copy()
        t['condition'] = t.corruption.map(NICE) + ' S' + t.severity.astype(str)
        for m in METHODS:
            t[m] = [f'**{pct(v)}**' if m in FUSION and b else pct(v)
                    for v, b in zip(t[m], t.get(f'{m}_below_healthy', pd.Series(False, index=t.index)))]
        md += [md_table(t, ['condition'] + METHODS, ['Condition'] + [LABEL[m] for m in METHODS]), '']
        du = dual[dual.protocol == p]
        if len(du):
            w = du.pivot_table(index=['rgb_corruption', 'rgb_severity', 'skeleton_corruption', 'skeleton_severity'],
                               columns='method', values='top1_accuracy').reset_index()
            w['condition'] = (w.rgb_corruption.map(NICE) + ' S' + w.rgb_severity.astype(str) + ' + '
                              + w.skeleton_corruption.map(NICE) + ' S' + w.skeleton_severity.astype(str))
            for m in METHODS:
                if m == 'rgb_only':
                    w[m] = [acc(df, p, m, rc, rs) for rc, rs in zip(w.rgb_corruption, w.rgb_severity)]
                elif m == 'skeleton_only':
                    w[m] = [acc(df, p, m, sc=sc, ss=ss) for sc, ss in zip(w.skeleton_corruption, w.skeleton_severity)]
                w[m] = w[m].map(pct) if m in w else ''
            md += ['### Dual corruption (3.10)', '', 'Both modalities corrupted independently.', '',
                   md_table(w, ['condition'] + METHODS, ['Condition'] + [LABEL[m] for m in METHODS]), '']
        c = cls[p]
        md += [f'### Class-level modality analysis (3.12)', '',
               f'Per-class clean accuracy; dominant = gap of at least {DOMINANCE_PTS:g} percentage points '
               f'(about 2 standard errors of a per-class difference at ~400-500 test samples per class). '
               f'Counts: ' + ', '.join(f'{k} {v}' for k, v in c.dominance.value_counts().items()) +
               f'. Full list: `class_modality_{p}.csv`.', '']
        top = pd.concat([c.head(10), c.tail(10)]).assign(
            rgb=lambda x: x.rgb_accuracy.map(pct), sk=lambda x: x.skeleton_accuracy.map(pct),
            gap=lambda x: x.rgb_minus_skeleton_pts.map('{:+.1f}'.format))
        md += [md_table(top, ['action_id', 'name', 'support', 'rgb', 'sk', 'gap', 'dominance'],
                        ['Action', 'Name', 'Test samples', 'RGB acc', 'Skeleton acc', 'RGB - skeleton (pts)', 'Class type']), '']
    both = cls['xsub'].merge(cls['xset'][['label', 'dominance']], on='label', suffixes=('_xsub', '_xset'))
    agree = both[(both.dominance_xsub == both.dominance_xset) & (both.dominance_xsub != 'balanced')]
    md += ['## Classes with the same dominance on both protocols', '',
           ', '.join(f'{r["name"]} ({r.dominance_xsub})' for _, r in agree.iterrows()) or 'none', '']
    (OUT / 'milestone3_results.md').write_text('\n'.join(md), encoding='utf-8')

    severity_figure(df, 'rgb', METHODS, OUT / 'fig_accuracy_vs_rgb_severity.png',
                    'Top-1 accuracy vs RGB corruption severity (skeleton clean)')
    severity_figure(df, 'skeleton', METHODS, OUT / 'fig_accuracy_vs_skeleton_severity.png',
                    'Top-1 accuracy vs skeleton corruption severity (RGB clean)')
    for modality, healthy in (('rgb', 'skeleton_only'), ('skeleton', 'rgb_only')):
        severity_figure(df, modality, [healthy] + FUSION, OUT / f'fig_fusion_vs_healthy_{modality}_corrupted.png',
                        f'Fusion vs the healthy modality, {LABEL[healthy]}, when {modality.upper() if modality == "rgb" else modality} is corrupted')
    class_scatter(cls, OUT / 'fig_class_rgb_vs_skeleton.png')
    heatmap(df, OUT / 'fig_heatmap_corruptions.png')
    ex = Path('outputs/corruption_examples')
    if ex.exists():
        (OUT / 'corruption_examples').mkdir(exist_ok=True)
        for f in sorted(ex.glob('0[1-3]_*.png')):
            shutil.copy(f, OUT / 'corruption_examples' / f.name)
    print(f'-> {OUT}: {len(df)} result rows')
    print(mca.to_string(index=False, float_format='%.4f'))


if __name__ == '__main__':
    main()
