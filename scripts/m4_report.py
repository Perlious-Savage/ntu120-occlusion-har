"""Milestone 4 tables from scripts/m4_fusion.py outputs (results/milestone4/fusion/*.csv).

main_table.csv / .md  (4.14 + review items B1-B4): per protocol, setting (A = clean training, B = mean over the
  three leave-one-family-out rotations) and method:
  clean accuracy / macro-F1 / ECE / NLL / Brier / AURC; mCA over the 30 single-modality conditions;
  for B also in-family and out-of-family mCA (the held-out families); dual, coupled-occlusion (interpolated)
  and A5-control means; mixed stream accuracy / ECE / AURC, and the coverage and risk reached on it with the
  threshold set for 90% coverage on clean calib validation; calib_val_mca (variant choice, A3).
  Seeds: mean over the seeds present (and SD of mCA / mixed accuracy when there are several).
ablation.md (4.15): A average, B calibrated average, C entropy weights, D learned reliability (target chosen on
  calib_val_mca); 'Full' (temporal) is added when the temporal model exists.
Usage: python scripts/m4_report.py
"""
from pathlib import Path

import pandas as pd
import yaml

IN, OUT = Path('results/milestone4/fusion'), Path('results/milestone4')
FAM = yaml.safe_load(open('configs/m4.yaml'))['families']


def family(c):
    return 'clean' if c in ('clean', 'mixed') else c.rsplit('_s', 1)[0]


def summarise(d):
    rf, sf = d.rgb_condition.map(family), d.skeleton_condition.map(family)
    single = ((rf == 'clean') & sf.isin(FAM['skeleton'])) | ((sf == 'clean') & rf.isin(FAM['rgb']))
    held = d.held_out.fillna('').iloc[0]
    out_fam = single & (rf.isin(held.split('+')) | sf.isin(held.split('+'))) if held else single & False
    q = lambda m: d[m].accuracy.mean()  # noqa: E731
    clean = d[(d.rgb_condition == 'clean') & (d.skeleton_condition == 'clean')].iloc[0]
    mixed = d[d.rgb_condition == 'mixed'].iloc[0]
    return {'clean_acc': clean.accuracy, 'clean_macro_f1': clean.macro_f1, 'clean_ece': clean.ece,
            'clean_nll': clean.nll, 'clean_brier': clean.brier, 'clean_aurc': clean.aurc,
            'mCA': q(single), 'in_family_mCA': q(single & ~out_fam) if held else float('nan'),
            'out_family_mCA': q(out_fam) if held else float('nan'),
            'dual': q((rf != 'clean') & (sf != 'clean') & (rf != 'mixed') & ~sf.str.startswith('skeleton_coupled')),
            'coupled_interp': q(sf == 'skeleton_coupled_interp'), 'coupled_zero': q(sf == 'skeleton_coupled_occlusion'),
            'controls': q(rf.eq('rgb_temporal_freeze') | sf.eq('skeleton_joint_interp')),
            'mixed_acc': mixed.accuracy, 'mixed_ece': mixed.ece, 'mixed_aurc': mixed.aurc,
            'mixed_cov@t0.9': mixed['coverage_at_t0.9'], 'mixed_risk@t0.9': mixed['risk_at_t0.9'],
            'calib_val_mca': clean.calib_val_mca}


def main():
    rows = []
    for f in sorted(IN.glob('*.csv')):
        d = pd.read_csv(f)
        for m, g in d.groupby('method', sort=False):
            rows.append({'protocol': g.protocol.iloc[0], 'setting': 'Ball' if g.setting.iloc[0] == 'Ball' else g.setting.iloc[0][0], 'rotation': g.setting.iloc[0],
                         'seed': g.seed.iloc[0], 'method': m, **summarise(g)})
    r = pd.DataFrame(rows)
    order = list(dict.fromkeys(r.method))
    per_seed = r.groupby(['protocol', 'setting', 'method', 'seed'], sort=False).mean(numeric_only=True)  # B: mean over rotations
    t = per_seed.groupby(level=[0, 1, 2], sort=False).mean()
    sd = per_seed.groupby(level=[0, 1, 2], sort=False)[['mCA', 'mixed_acc']].std().add_suffix('_sd')
    t = t.join(sd).join(per_seed.groupby(level=[0, 1, 2], sort=False).size().rename('n_seeds')).reset_index()
    t['method'] = pd.Categorical(t.method, order)
    t = t.sort_values(['protocol', 'setting', 'method'])
    t.to_csv(OUT / 'main_table.csv', index=False)

    pct = [c for c in t.columns if c not in ('protocol', 'setting', 'method', 'n_seeds') and 'ece' not in c
           and 'nll' not in c and 'brier' not in c and 'aurc' not in c and 'calib_val' not in c and not c.endswith('_sd')]
    show = t.copy()
    for c in pct + ['calib_val_mca', 'mCA_sd', 'mixed_acc_sd']:
        show[c] = 100 * show[c]
    lines = ['# Milestone 4 main comparison (frozen branches; seed-42 backbones)', '',
             'Accuracies in %. A = heads trained on clean validation (fit half); B = corruption-aware training, mean '
             'over three leave-one-family-out rotations (out-of-family = the held-out RGB and skeleton family). '
             'mixed = test stream with each sample clean / RGB / skeleton / dual corrupted (40/20/20/20). '
             'cov/risk@t0.9 = coverage and risk on the mixed stream at the threshold giving 90% coverage on clean '
             'calib validation. calib_val_mca chooses variants (never test).', '']
    for (p, s), g in show.groupby(['protocol', 'setting'], sort=False):
        lines += [f'## {p} - setting {s}', '', g.drop(columns=['protocol', 'setting']).round(
            {**{c: 2 for c in pct + ['calib_val_mca', 'mCA_sd', 'mixed_acc_sd']}, 'clean_ece': 4, 'clean_nll': 3,
             'clean_brier': 4, 'clean_aurc': 4, 'mixed_ece': 4, 'mixed_aurc': 4}).dropna(axis=1, how='all')
            .to_markdown(index=False), '']
    (OUT / 'main_table.md').write_text('\n'.join(lines))

    ab = ['# Ablation (4.15)', '', '| Protocol | Setting | Variant | Method | Clean | mCA | Mixed | Mixed ECE | Mixed AURC |',
          '|---|---|---|---|---|---|---|---|---|']
    for (p, s), g in t.groupby(['protocol', 'setting'], sort=False):
        g = g.set_index('method')
        rel = [m for m in g.index if str(m).startswith('reliability')]
        best = max(rel, key=lambda m: g.loc[m, 'calib_val_mca']) if rel else None
        for v, m in (('A (no calibration, fixed)', 'average'), ('B (+ calibration)', 'calibrated_average'),
                     ('C (+ dynamic entropy weights)', 'entropy'), ('D (+ learned reliability)', best)):
            if m in g.index:
                x = g.loc[m]
                ab.append(f'| {p} | {s} | {v} | {m} | {100 * x.clean_acc:.2f} | {100 * x.mCA:.2f} | '
                          f'{100 * x.mixed_acc:.2f} | {x.mixed_ece:.4f} | {x.mixed_aurc:.4f} |')
    ab += ['', 'Full (temporal reliability + consistency loss): pending (4.8-4.10).']
    (OUT / 'ablation.md').write_text('\n'.join(ab))
    print('\n'.join(lines[:3]))
    print((OUT / 'ablation.md').read_text())


if __name__ == '__main__':
    main()
