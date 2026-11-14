"""Choose each fusion head's training setting on CORRUPTED VALIDATION data, never test (review item A1).

Settings (the three rows of fusion_ablation.csv):
  val_aux    heads trained on the validation split, aux loss    (outputs/m3_fusion)
  val_noaux  same without aux loss                               (outputs/m3_fusion_noaux)
  trainfeat  heads trained on training features                  (outputs/m3_fusion_trainfeat)
A head must not be scored on the samples it was fitted on, so the validation-trained settings are
scored OUT OF FOLD: the same 5 class-stratified folds as training, each fold head fitted on the clean
features of the other folds and stopped at the cross-validated epoch, then evaluated on its held-out
fold under all 31 single-modality conditions (clean + 30). Train-feature heads are scored directly
(validation only chose their stopping epoch). Score = validation mCA over the 30 corrupted conditions.

Needs outputs/m3_robustness/<branch>/val_corrupted/ (evaluate_corruptions.py --split val --out ...).
Writes results/milestone3/fusion_selection_val.csv (+ .md) and prints the chosen setting per head
next to its test numbers from <setting dir>/<method>_<protocol>_seed42/summary.csv.
Usage: python scripts/select_fusion_on_val.py [xsub] [xset]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_fusion import Features, cv_folds, fit, infer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.corruptions.base import SEVERITIES  # noqa: E402
from src.corruptions.registry import RGB, SKELETON  # noqa: E402
from src.models.fusion import FusionHead  # noqa: E402

SETTINGS = {'val_aux': 'outputs/m3_fusion', 'val_noaux': 'outputs/m3_fusion_noaux', 'trainfeat': 'outputs/m3_fusion_trainfeat'}
OUT = Path('results/milestone3')
CONDS = [('clean', 'clean')] + [(f'{c}_s{s}', 'clean') for c in RGB for s in SEVERITIES] + \
        [('clean', f'{c}_s{s}') for c in SKELETON for s in SEVERITIES]


def score(acc):
    """acc: {condition pair: accuracy} -> (clean, mCA over the 30 corrupted conditions)."""
    return acc[CONDS[0]], float(np.mean([acc[c] for c in CONDS[1:]]))


def main():
    protocols = sys.argv[1:] or ['xsub', 'xset']   # e.g. `xsub` while the X-Set heads are being re-fitted
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = []
    for p in protocols:
        for setting, d in SETTINGS.items():
            for method in ('average', 'concat', 'global', 'attention'):
                run = Path(d) / f'{method}_{p}_seed42'
                if not (run / 'config.yaml').exists():
                    continue
                cfg = yaml.safe_load(open(run / 'config.yaml'))
                val = Features(cfg, p, 'val_corrupted')
                lp = method == 'attention'
                data = {c: val.pair(*c, lp) for c in CONDS}
                y = data[CONDS[0]][4]
                if method == 'average':
                    acc = {c: float(((v[2] + v[3]).argmax(1) == v[4]).mean()) for c, v in data.items()}
                elif cfg.get('train_split', 'train') == 'train':     # old configs: training features, no aux
                    ck = torch.load(run / 'checkpoint.pt', map_location=device, weights_only=False)
                    fr, fs = data[CONDS[0]][:2]
                    head = FusionHead(method, fr.shape[1], fs.shape[1], cfg['latent_dim'], cfg['num_classes'], cfg['dropout']).to(device)
                    head.load_state_dict(ck['model_state_dict'])
                    acc = {c: float((infer(head, v[0], v[1], device)[0].argmax(1) == v[4]).mean()) for c, v in data.items()}
                else:                              # out of fold, same folds and stopping epoch as training
                    epoch = int(pd.read_csv(run / 'cv_accuracy.csv')['mean'].idxmax())
                    right = {c: np.zeros(len(y), bool) for c in CONDS}
                    for a, b in cv_folds(data[CONDS[0]][5], y, cfg['cv_folds'], cfg['seed']):
                        fr, fs = data[CONDS[0]][:2]
                        _, head, _ = fit(cfg, method, (fr[a], fs[a], y[a]), None, device, keep_epoch=epoch)
                        for c, v in data.items():
                            right[c][b] = infer(head, v[0][b], v[1][b], device)[0].argmax(1) == y[b]
                    acc = {c: float(r.mean()) for c, r in right.items()}
                clean, mca = score(acc)
                test = pd.read_csv(run / 'summary.csv')
                single = test[(test.rgb_corruption == 'clean') | (test.skeleton_corruption == 'clean')]
                single = single[~((single.rgb_corruption != 'clean') & (single.skeleton_corruption != 'clean'))]
                t_clean = float(single[(single.rgb_corruption == 'clean') & (single.skeleton_corruption == 'clean')].top1_accuracy.iloc[0])
                core = single[single.rgb_corruption.isin(['clean'] + RGB) & single.skeleton_corruption.isin(['clean'] + SKELETON)]
                t_mca = float(core[(core.rgb_corruption != 'clean') | (core.skeleton_corruption != 'clean')].top1_accuracy.mean())
                rows.append({'protocol': p, 'method': method, 'setting': setting, 'val_clean': clean, 'val_mCA': mca,
                             'test_clean': t_clean, 'test_mCA': t_mca})
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df['chosen_on_val'] = df.groupby(['protocol', 'method']).val_mCA.transform('max') == df.val_mCA
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f'fusion_selection_val{"_" + protocols[0] if len(protocols) == 1 else ""}.csv', index=False)
    md = ['# Fusion setting chosen on corrupted validation data (review item A1)', '',
          'Validation-trained settings scored out of fold. mCA = mean top-1 over the 30 single-modality '
          'corrupted conditions. **Bold** = chosen on validation; its test numbers are the ones to report.', '',
          '| Protocol | Method | Setting | Val clean | Val mCA | Test clean | Test mCA |', '|---|---|---|---|---|---|---|']
    for _, r in df.iterrows():
        b = '**' if r.chosen_on_val else ''
        md.append(f'| {r.protocol} | {r.method} | {b}{r.setting}{b} | {r.val_clean * 100:.2f} | {r.val_mCA * 100:.2f} | '
                  f'{r.test_clean * 100:.2f} | {r.test_mCA * 100:.2f} |')
    (OUT / f'fusion_selection_val{"_" + protocols[0] if len(protocols) == 1 else ""}.md').write_text('\n'.join(md), encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
