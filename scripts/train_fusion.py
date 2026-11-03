"""Train one fusion baseline on clean frozen branch features, then test it under corruption without
retraining (Milestone 3, steps 3.7-3.10).

Methods: average (mean of the two branch softmaxes, no training), concat, global, attention
(src/models/fusion.py). Training uses clean train_internal features; the epoch with the best clean
validation accuracy is kept. Test conditions are listed in configs/fusion.yaml.

Outputs in <output_dir>/<method>_<protocol>_seed<seed>/:
  config.yaml, checkpoint.pt, training_log.csv (trained methods)
  test/<rgb condition>+<skeleton condition>/predictions.csv, probabilities.npy (float16),
      weights.npy ((N, 2) RGB/skeleton weight, weighting methods), metrics.json
  summary.csv   one row per test condition
Usage: python scripts/train_fusion.py --protocol xsub --method attention [--config configs/fusion.yaml]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.corruptions.base import SEVERITIES  # noqa: E402
from src.corruptions.registry import RGB, SKELETON  # noqa: E402
from src.engine import classification_metrics, seed_everything  # noqa: E402
from src.models.fusion import FusionHead  # noqa: E402

METHODS = ('average', 'concat', 'global', 'attention')


class Features:
    """Saved branch outputs of one split: load(modality, condition) -> features, probs, labels, ids."""

    def __init__(self, cfg, protocol, split):
        self.dirs = {m: Path(cfg['features_dir']) / r.format(protocol=protocol) / split
                     for m, r in cfg['branch_runs'].items()}

    def load(self, modality, condition):
        d = self.dirs[modality] / condition
        pred = pd.read_csv(d / 'predictions.csv')
        return (np.load(d / 'features.npy').astype(np.float32), np.load(d / 'probabilities.npy').astype(np.float32),
                pred.true_label.to_numpy(), pred.sample_id.to_numpy())

    def pair(self, rgb_condition, sk_condition):
        fr, pr, y, ids = self.load('rgb', rgb_condition)
        fs, ps, y2, ids2 = self.load('skeleton', sk_condition)
        assert (ids == ids2).all() and (y == y2).all(), 'RGB and skeleton rows differ'
        return fr, fs, pr, ps, y, ids


def test_conditions(cfg):
    conds = [('clean', 'clean')]
    conds += [(f'{c}_s{s}', 'clean') for c in RGB for s in SEVERITIES]
    conds += [('clean', f'{c}_s{s}') for c in SKELETON for s in SEVERITIES]
    dual = cfg['dual_corruptions']
    conds += [(f'{rc}_s{a}', f'{sc}_s{b}') for rc, sc in dual['pairs'] for a, b in dual['severities']]
    return conds


@torch.no_grad()
def infer(head, fr, fs, device, bs=4096):
    head.eval()
    logits, weights = [], []
    for i in range(0, len(fr), bs):
        lg, w = head(torch.from_numpy(fr[i:i + bs]).to(device), torch.from_numpy(fs[i:i + bs]).to(device))
        logits.append(lg.float().cpu())
        weights.append(None if w is None else w.cpu())
    probs = torch.cat(logits).softmax(1).numpy()
    return probs, (None if weights[0] is None else torch.cat(weights).numpy())


def fit(cfg, method, tr, ev, device, keep_epoch=None):
    """Train one head for cfg['epochs'] epochs on tr = (f_rgb, f_sk, y), evaluating on ev after every
    epoch (ev may be None). Returns (log rows, head with the weights of keep_epoch, or of the best ev
    epoch, that epoch)."""
    seed_everything(cfg['seed'])
    fr, fs, y = tr
    head = FusionHead(method, fr.shape[1], fs.shape[1], cfg['latent_dim'], cfg['num_classes'], cfg['dropout']).to(device)
    o = cfg['optimizer']
    opt = torch.optim.AdamW(head.parameters(), lr=o['lr'], weight_decay=o['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs'] * int(np.ceil(len(y) / cfg['batch_size'])))
    fr, fs, yt = (torch.from_numpy(a).to(device) for a in (fr, fs, y))
    gen = torch.Generator(device='cpu').manual_seed(cfg['seed'])
    best, log = (-1.0, None, -1), []
    for epoch in range(cfg['epochs']):
        t0 = time.time()
        head.train()
        total = correct = 0
        for idx in torch.randperm(len(yt), generator=gen).split(cfg['batch_size']):
            idx = idx.to(device)
            logits, _ = head(fr[idx], fs[idx])
            loss = F.cross_entropy(logits, yt[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            total += loss.item() * len(idx); correct += (logits.argmax(1) == yt[idx]).sum().item()
        row = {'epoch': epoch, 'train_loss': total / len(yt), 'train_accuracy': correct / len(yt)}
        if ev is not None:
            vp, vw = infer(head, ev[0], ev[1], device)
            m = classification_metrics(vp, ev[2], cfg['num_classes'])
            row.update(val_loss=float(F.nll_loss(torch.log(torch.from_numpy(vp) + 1e-12), torch.from_numpy(ev[2]))),
                       val_accuracy=m['top1_accuracy'], val_macro_f1=m['macro_f1'],
                       val_mean_w_rgb=float('nan') if vw is None else float(vw[:, 0].mean()))
        row['epoch_seconds'] = time.time() - t0
        log.append(row)
        score = row.get('val_accuracy', 0.0)
        if (keep_epoch is None and score > best[0]) or epoch == keep_epoch:
            best = (score, {k: v.detach().clone() for k, v in head.state_dict().items()}, epoch)
    head.load_state_dict(best[1])
    return log, head, best[2]


def cv_folds(protocol, ids, y, k, seed):
    """X-Sub: folds grouped by subject (no subject on both sides); X-Set: stratified by class."""
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    if protocol == 'xsub':
        subjects = [int(s[9:12]) for s in ids]                       # S001C001P003R001A001 -> 3
        return list(GroupKFold(n_splits=k).split(ids, y, subjects))
    return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(ids, y))


def train(cfg, method, feats_train, feats_val, device, run):
    """Main protocol (train_split: val): heads learn from the held-out validation split, where the
    frozen branches make realistic errors. k-fold CV inside it picks the stopping epoch; the final
    head is trained on all of it with the same schedule and stopped at that epoch.
    Ablation (train_split: train): train on the branches' own training features, select on val."""
    vr, vs, _, _, vy, vids = feats_val.pair('clean', 'clean')
    if cfg['train_split'] == 'train':
        fr, fs, _, _, y, _ = feats_train.pair('clean', 'clean')
        log, head, epoch = fit(cfg, method, (fr, fs, y), (vr, vs, vy), device)
        print(f'best epoch {epoch} (0-based), val top-1 {log[epoch]["val_accuracy"]:.4f}', flush=True)
    else:
        folds = cv_folds(cfg['protocol'], vids, vy, cfg['cv_folds'], cfg['seed'])
        acc = []
        for f, (a, b) in enumerate(folds):
            flog, _, _ = fit(cfg, method, (vr[a], vs[a], vy[a]), (vr[b], vs[b], vy[b]), device)
            acc.append([r['val_accuracy'] for r in flog])
            print(f'fold {f}: {len(a)} train / {len(b)} held out, best held-out top-1 {max(acc[-1]):.4f} '
                  f'at epoch {int(np.argmax(acc[-1]))}', flush=True)
        acc = np.array(acc)
        epoch = int(acc.mean(0).argmax())
        pd.DataFrame(acc.T, columns=[f'fold{f}' for f in range(len(folds))]).assign(
            mean=acc.mean(0), epoch=range(acc.shape[1])).to_csv(run / 'cv_accuracy.csv', index=False)
        print(f'cross-validated stopping epoch {epoch} (0-based), mean held-out top-1 {acc.mean(0)[epoch]:.4f}', flush=True)
        log, head, _ = fit(cfg, method, (vr, vs, vy), None, device, keep_epoch=epoch)
    pd.DataFrame(log).to_csv(run / 'training_log.csv', index=False)
    torch.save({'epoch': epoch, 'method': method, 'model_state_dict': head.state_dict(), 'config': cfg},
               run / 'checkpoint.pt')
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/fusion.yaml')
    ap.add_argument('--protocol', required=True, choices=['xsub', 'xset'])
    ap.add_argument('--method', required=True, choices=METHODS)
    ap.add_argument('--train-split', choices=['val', 'train'], help='override config train_split (train = ablation)')
    ap.add_argument('--epochs', type=int, help='override config epochs')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    cfg.update(protocol=args.protocol, method=args.method)
    if args.train_split:
        cfg['train_split'] = args.train_split
    if args.epochs:
        cfg['epochs'] = args.epochs
    seed_everything(cfg['seed'])
    out = Path(cfg['output_dir'] + ('' if cfg['train_split'] == 'val' else '_trainfeat'))
    run = out / f'{args.method}_{args.protocol}_seed{cfg["seed"]}'
    run.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(run / 'config.yaml', 'w'), sort_keys=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    head = None
    if args.method != 'average':
        head = train(cfg, args.method, Features(cfg, args.protocol, 'train'), Features(cfg, args.protocol, 'val'), device, run)
    test, rows = Features(cfg, args.protocol, 'test'), []
    for rc, sc in test_conditions(cfg):
        fr, fs, pr, ps, y, ids = test.pair(rc, sc)
        if head is None:
            probs, w = (pr + ps) / 2, np.full((len(y), 2), 0.5, dtype=np.float32)
        else:
            probs, w = infer(head, fr, fs, device)
        d = run / 'test' / f'{rc}+{sc}'
        d.mkdir(parents=True, exist_ok=True)
        pred = probs.argmax(1)
        pd.DataFrame({'sample_id': ids, 'true_label': y, 'predicted_label': pred, 'confidence': probs.max(1),
                      'correct': pred == y}).to_csv(d / 'predictions.csv', index=False)
        np.save(d / 'probabilities.npy', probs.astype(np.float16))
        if w is not None:
            np.save(d / 'weights.npy', w.astype(np.float16))
        split = lambda c: (c.rsplit('_s', 1)[0], int(c.rsplit('_s', 1)[1])) if c != 'clean' else ('clean', 0)  # noqa: E731
        (rcn, rs), (scn, ss) = split(rc), split(sc)
        m = {'method': args.method, 'protocol': args.protocol, 'rgb_corruption': rcn, 'rgb_severity': rs,
             'skeleton_corruption': scn, 'skeleton_severity': ss, **classification_metrics(probs, y, cfg['num_classes']),
             'mean_w_rgb': float('nan') if w is None else float(w[:, 0].mean())}
        json.dump(m, open(d / 'metrics.json', 'w'), indent=2)
        rows.append(m)
        print(f'{rc:>24} + {sc:<28} top-1 {m["top1_accuracy"]:.4f}  w_rgb {m["mean_w_rgb"]:.3f}', flush=True)
    pd.DataFrame(rows).to_csv(run / 'summary.csv', index=False)
    print(f'-> {run}/summary.csv')


if __name__ == '__main__':
    main()
