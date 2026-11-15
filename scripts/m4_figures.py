"""Milestone 4 figures from results/milestone4/fusion/*.csv (seed 42; Setting B = mean over the three rotations).

fig_severity_<protocol>.png      4.16: top-1 vs severity S0-S5 for attention, entropy and the proposed method,
                                 one panel per corruption family (part dropout plotted as categories, B8).
fig_weight_response_<protocol>.png  4.18 / B8: mean RGB weight vs severity (RGB families should push it down,
                                 skeleton families up), attention and global-free methods side by side.
fig_risk_coverage_<protocol>.png 4.13: risk vs coverage on clean and on the mixed stream.
The proposed variant is the reliability head with the best calib_val_mca in Setting B (A3).
Usage: python scripts/m4_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

IN, OUT = Path('results/milestone4/fusion'), Path('results/milestone4')
FAM = yaml.safe_load(open('configs/m4.yaml'))['families']
PART = ['one_leg', 'one_arm', 'lower_body', 'upper_body', 'upper_body_one_leg']


def load(protocol):
    d = pd.concat([pd.read_csv(f) for f in IN.glob(f'{protocol}_*_seed42.csv')])
    d['setting'] = d.setting.where(d.setting == 'Ball', d.setting.str[0])
    keys = ['setting', 'method', 'rgb_condition', 'skeleton_condition']
    return d.groupby(keys, as_index=False).mean(numeric_only=True)


def proposed(d):
    b = d[(d.setting == 'B') & d.method.str.startswith('reliability')]
    return b.groupby('method').calib_val_mca.mean().idxmax()


def curve(d, method, fam):
    """(severity list, metric rows) for one family with the other modality clean, S0 = clean."""
    rgb = fam in FAM['rgb']
    g = d[(d.method == method) & ((d.rgb_condition.str.startswith(fam) & (d.skeleton_condition == 'clean')) if rgb
                                  else (d.skeleton_condition.str.startswith(fam) & (d.rgb_condition == 'clean')))]
    g = g.assign(sev=(g.rgb_condition if rgb else g.skeleton_condition).str.rsplit('_s', n=1).str[1].astype(int)).sort_values('sev')
    clean = d[(d.method == method) & (d.rgb_condition == 'clean') & (d.skeleton_condition == 'clean')]
    return pd.concat([clean.assign(sev=0), g])


def main():
    for protocol in ('xsub', 'xset'):
        d = load(protocol)
        prop = proposed(d)
        methods = [('attention', 'B', 'attention (B)'), ('entropy', 'A', 'entropy weights'), ('average', 'A', 'average'),
                   (prop, 'B', f'{prop} (B, proposed)')]
        fams = FAM['rgb'] + FAM['skeleton']

        fig, ax = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
        for a, fam in zip(ax.flat, fams):
            for m, s, label in methods:
                c = curve(d[d.setting == s], m, fam)
                a.plot(c.sev, 100 * c.accuracy, marker='o', ms=3, label=label)
            a.set_title(fam); a.grid(alpha=.3); a.set_xticks(range(6))
            a.set_xticklabels(['clean'] + (PART if fam == 'skeleton_part_dropout' else [f'S{i}' for i in range(1, 6)]),
                              rotation=30 if fam == 'skeleton_part_dropout' else 0, fontsize=8)
        ax[0, 0].set_ylabel('top-1 (%)'); ax[1, 0].set_ylabel('top-1 (%)'); ax[0, 0].legend(fontsize=8)
        fig.suptitle(f'{protocol}: accuracy vs severity (other modality clean)'); fig.tight_layout()
        fig.savefig(OUT / f'fig_severity_{protocol}.png', dpi=130); plt.close(fig)

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        for a, (m, s, label) in zip(ax, [methods[0], methods[1], methods[3]]):
            for fam in fams:
                c = curve(d[d.setting == s], m, fam)
                a.plot(c.sev, c.mean_w_rgb, marker='o', ms=3, ls='-' if fam in FAM['rgb'] else '--', label=fam)
            a.axhline(0.5, color='grey', lw=.8); a.set_title(label); a.grid(alpha=.3); a.set_xlabel('severity (0 = clean)')
        ax[0].set_ylabel('mean RGB weight'); ax[0].legend(fontsize=7)
        fig.suptitle(f'{protocol}: weight response (solid = RGB corrupted, should fall; dashed = skeleton corrupted, should rise;'
                     ' part dropout severity is categorical)', fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / f'fig_weight_response_{protocol}.png', dpi=130); plt.close(fig)

        cov = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
        for a, cond in zip(ax, ('clean', 'mixed')):
            for m, s, label in methods + [('rgb', 'A', 'rgb only'), ('skeleton', 'A', 'skeleton only')]:
                r = d[(d.setting == s) & (d.method == m) & (d.rgb_condition == cond)].iloc[0]
                a.plot(cov, [100 * r[f'risk@{c}'] for c in cov], marker='o', ms=3, label=f'{label} (AURC {r.aurc:.4f})')
            a.set_title(f'{cond} test'); a.set_xlabel('coverage'); a.set_ylabel('risk = 1 - accuracy (%)'); a.invert_xaxis()
            a.grid(alpha=.3); a.legend(fontsize=7)
        fig.suptitle(f'{protocol}: risk-coverage (samples ranked by fused confidence)'); fig.tight_layout()
        fig.savefig(OUT / f'fig_risk_coverage_{protocol}.png', dpi=130); plt.close(fig)
        print(protocol, 'proposed =', prop)


if __name__ == '__main__':
    main()
