"""Milestone 4 steps 4.1-4.4 and review item B3 on the saved frozen-branch probabilities.

Per branch and protocol: one temperature fitted on the clean 'calib' half of validation (src/calibration.py,
val_half; never on test), then for every saved test condition:
  accuracy, ECE / NLL / Brier before and after temperature scaling (4.3),
  mean normalised entropy U and reliability R = 1 - U (4.1),
  error-detection AUROC of U WITHIN the condition (4.4, B3.1),
  corruption-presence AUROC of U (condition vs clean test samples; B3.1, kept separate),
  share of errors made with calibrated confidence >= 0.5, i.e. confidently wrong rather than uncertain (B3.2).
Outputs: results/milestone4/temperatures.csv, branch_uncertainty.csv, fig_branch_uncertainty_<protocol>.png
Usage: python scripts/m4_uncertainty.py [--features-dir outputs/m3_robustness]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.calibration import calibration_metrics, entropy, fit_temperature, scale, val_half  # noqa: E402

RUNS = {'rgb': 'r3d18_{p}_seed42', 'skeleton': 'ctrgcn_{p}_seed42'}
OUT = Path('results/milestone4')


def load(d):
    p = np.load(d / 'probabilities.npy').astype(np.float32)
    pred = pd.read_csv(d / 'predictions.csv')
    return p / p.sum(1, keepdims=True), pred.true_label.to_numpy(), pred.sample_id.to_numpy()


def split_name(c):
    return ('clean', 0) if c == 'clean' else (c.rsplit('_s', 1)[0], int(c.rsplit('_s', 1)[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features-dir', default='outputs/m3_robustness', type=Path)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    temps, rows = [], []
    for protocol in ('xsub', 'xset'):
        for branch, r in RUNS.items():
            run = args.features_dir / r.format(p=protocol)
            pv, yv, iv = load(run / 'val' / 'clean')
            calib = val_half(iv) == 1
            t = fit_temperature(pv[calib], yv[calib])
            temps.append({'protocol': protocol, 'branch': branch, 'temperature': t, 'n_calib': int(calib.sum()),
                          **{f'calib_{k}_raw': v for k, v in calibration_metrics(pv[calib], yv[calib]).items()},
                          **{f'calib_{k}_cal': v for k, v in calibration_metrics(scale(pv[calib], t), yv[calib]).items()}})
            print(f'{protocol} {branch}: T = {t:.3f}', flush=True)
            u_clean = None
            conds = sorted(d.name for d in (run / 'test').iterdir() if (d / 'probabilities.npy').exists())
            for c in ['clean'] + [c for c in conds if c != 'clean']:
                p, y, _ = load(run / 'test' / c)
                q = scale(p, t)
                u = entropy(q)
                wrong = q.argmax(1) != y
                if c == 'clean':
                    u_clean = u
                name, sev = split_name(c)
                row = {'protocol': protocol, 'branch': branch, 'corruption': name, 'severity': sev, 'n': len(y),
                       **{f'{k}_raw': v for k, v in calibration_metrics(p, y).items() if k != 'accuracy'},
                       **calibration_metrics(q, y),
                       'mean_U': float(u.mean()), 'mean_R': float(1 - u.mean()),
                       'mean_U_correct': float(u[~wrong].mean()), 'mean_U_wrong': float(u[wrong].mean()),
                       'error_auroc': float(roc_auc_score(wrong, u)) if 0 < wrong.sum() < len(y) else np.nan,
                       'corruption_auroc': np.nan if c == 'clean' else float(roc_auc_score(
                           np.r_[np.zeros(len(u_clean)), np.ones(len(u))], np.r_[u_clean, u])),
                       'confidently_wrong_share': float((q.max(1)[wrong] >= 0.5).mean())}
                rows.append(row)
    pd.DataFrame(temps).to_csv(OUT / 'temperatures.csv', index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'branch_uncertainty.csv', index=False)
    print(df.groupby(['protocol', 'branch'])[['ece_raw', 'ece', 'error_auroc', 'corruption_auroc',
                                              'confidently_wrong_share']].mean().round(3))

    for protocol in ('xsub', 'xset'):                     # 4.3: accuracy, uncertainty, ECE versus severity
        fig, ax = plt.subplots(3, 2, figsize=(11, 10), sharex=True)
        for j, branch in enumerate(RUNS):
            d = df[(df.protocol == protocol) & (df.branch == branch)]
            clean = d[d.corruption == 'clean'].iloc[0]
            for name in sorted(set(d.corruption) - {'clean'}):
                e = d[d.corruption == name].sort_values('severity')
                for i, k in enumerate(('accuracy', 'mean_U', 'ece')):
                    ax[i, j].plot([0, *e.severity], [clean[k], *e[k]], marker='o', ms=3, label=name)
            for i, k in enumerate(('top-1 accuracy', 'mean normalised entropy U', 'ECE (calibrated)')):
                ax[i, j].set_ylabel(k); ax[i, j].grid(alpha=.3)
            ax[0, j].set_title(f'{"R3D-18" if branch == "rgb" else "CTR-GCN"} {protocol}')
            ax[2, j].set_xlabel('severity (0 = clean)'); ax[0, j].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(OUT / f'fig_branch_uncertainty_{protocol}.png', dpi=130); plt.close(fig)


if __name__ == '__main__':
    main()
