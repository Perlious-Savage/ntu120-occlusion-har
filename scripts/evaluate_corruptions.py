"""Single-modality robustness of a frozen Milestone 2 baseline (Milestone 3, step 3.6).

Each sample is loaded once and every condition (clean + this modality's 3 corruptions x S1-S5 = 16)
is built from that same clean input, so videos are decoded once, not 16 times. Corruptions act on
the exact model input (RGB: uint8 16 x 112 x 112 centre crop; skeleton: 3 x 64 x 25 x 2 after the
official eval crop/resize) with seed sample_seed(sample_id, corruption, severity, cfg seed).

Outputs in --out (default outputs/m3_robustness/<run>/<split>):
  <condition>/probabilities.npy  (N, 120) float16 softmax, rows in predictions.csv order
  <condition>/features.npy       (N, 512 RGB | 256 skeleton) float16 branch features (for fusion)
  <condition>/predictions.csv    sample_id, true_label, predicted_label, confidence, correct
  <condition>/per_class_accuracy.csv, <condition>/metrics.json
  summary.csv                    one row per condition: top-1, top-5, macro-F1, robustness ratio
  shards/                        resumable partial results (a rerun skips finished shards)
Conditions are named clean or <corruption>_s<severity>.
Usage: python scripts/evaluate_corruptions.py --run outputs/r3d18_xsub_seed42 [--split test] [--clean-only]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.corruptions.base import SEVERITIES, sample_seed  # noqa: E402
from src.corruptions.registry import COUPLED, RGB, SKELETON, get  # noqa: E402
from src.corruptions.rgb_occlusion import boxes_in_clip, to_clip  # noqa: E402
from src.datasets.baseline_datasets import build_dataset  # noqa: E402
from src.datasets.rgb_loader import read_selected_frames  # noqa: E402
from src.engine import amp_dtype, classification_metrics  # noqa: E402
from src.models.baselines import build_model  # noqa: E402

SHARD = 4096


def condition_list(modality, clean_only, corruption=None):
    if corruption:                                    # one extra corruption, S1-S5 only
        return [(corruption, s) for s in SEVERITIES]
    names = RGB if modality == 'rgb' else SKELETON
    return [('clean', 0)] + ([] if clean_only else [(c, s) for c in names for s in SEVERITIES])


def condition_name(c, s):
    return 'clean' if s == 0 else f'{c}_s{s}'


class AllConditions(torch.utils.data.Dataset):
    """Wraps the Milestone 2 eval dataset; returns (K, ...) inputs, one per condition."""

    def __init__(self, base, conditions, seed, boxes=None, rgb=False, clip=None):
        """clip: RGB input geometry {'resize_hw', 'size'} for the coupled occlusion (skeleton runs)."""
        self.base, self.conditions, self.seed, self.boxes, self.rgb, self.clip = base, conditions, seed, boxes, rgb, clip
        self.coupled = any(c in COUPLED for c, _ in conditions)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        sid = self.base.df.sample_id.iloc[i]
        label = int(self.base.df.label.iloc[i])
        if self.rgb:
            b = self.base
            _, frames = read_selected_frames(b.root / b.df.rgb_path.iloc[i], b.pick, b.resize_hw)
            top, left = (frames.shape[1] - b.size) // 2, (frames.shape[2] - b.size) // 2
            x = np.ascontiguousarray(frames[:, top:top + b.size, left:left + b.size])   # (T, S, S, 3) uint8
            ctx = {'boxes': boxes_in_clip(self.boxes.loc[sid], b.resize_hw, b.size)}
        else:
            x, ctx = self.base[i][0].numpy(), {}
            if self.coupled:                          # the RGB occluder of this sample, and where its joints are
                ctx = {'boxes': boxes_in_clip(self.boxes.loc[sid], **self.clip), 'size': self.clip['size'],
                       'color': np.stack(to_clip(*self.base.eval_color(i), **self.clip))}
        out = np.stack([get(c).apply(x, s, sample_seed(sid, get(c).seed_as or c, s, self.seed), **ctx) if s else x
                        for c, s in self.conditions])
        return torch.from_numpy(out), label, sid


def to_model_input(x, rgb):
    """(B, K, ...) batch -> (B*K, ...) float model input."""
    x = x.flatten(0, 1)
    return x.permute(0, 4, 1, 2, 3).float().div_(255) if rgb else x


@torch.no_grad()
def run_shard(model, dl, k, rgb, device, dtype, use_amp):
    probs, feats, labels, ids = [], [], [], []
    for x, y, sid in dl:
        x = to_model_input(x.to(device, non_blocking=True), rgb)
        with torch.autocast('cuda', dtype=dtype, enabled=use_amp):
            f, logits = model(x, return_features=True)
        b = len(y)
        probs.append(logits.float().softmax(1).view(b, k, -1).half().cpu())
        feats.append(f.float().view(b, k, -1).half().cpu())
        labels.append(y)
        ids += list(sid)
    return torch.cat(probs).numpy(), torch.cat(feats).numpy(), torch.cat(labels).numpy(), ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, type=Path)
    ap.add_argument('--split', default='test', help='test, val or train (the run config split) or a split CSV')
    ap.add_argument('--clean-only', action='store_true', help='only the clean condition (e.g. train/val features)')
    ap.add_argument('--corruption', choices=COUPLED, help='evaluate only this extra corruption (S1-S5), kept apart from the main shards')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--batch-size', type=int, help='samples per batch (each holds 16 conditions)')
    ap.add_argument('--workers', type=int)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.run / 'config.yaml'))
    rgb = cfg['model']['name'] == 'r3d18'
    out = args.out or Path('outputs/m3_robustness') / args.run.name / Path(args.split).stem
    shard_dir = out / ('shards' + (f'_{args.corruption}' if args.corruption else ''))
    shard_dir.mkdir(parents=True, exist_ok=True)
    conditions = condition_list('rgb' if rgb else 'skeleton', args.clean_only, args.corruption)
    k = len(conditions)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    ck = torch.load(args.run / 'best_val_accuracy.pt', map_location=device, weights_only=False)
    model = build_model({**cfg, 'model': {**cfg['model'], 'pretrained': False}}).to(device).eval()
    model.load_state_dict(ck['model_state_dict'])
    use_amp, dtype = cfg.get('amp', True) and device.type == 'cuda', amp_dtype()

    split_csv = args.split if args.split.endswith('.csv') else cfg['splits'][args.split]
    base = build_dataset(cfg, split_csv, train=False)
    boxes = pd.read_csv(Path(cfg['data_root']) / 'metadata/person_boxes.csv').set_index('sample_id')
    data_cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / 'configs/data.yaml'))
    clip = {'resize_hw': tuple(data_cfg['rgb_resize_hw']), 'size': data_cfg['rgb_size']}
    ds = AllConditions(base, conditions, cfg['seed'], boxes, rgb, clip)
    bs = args.batch_size or max(1, (128 if rgb else 1024) // k)
    workers = cfg['num_workers'] if args.workers is None else args.workers
    shards = range(0, len(ds), SHARD)
    print(f'{args.run.name} {args.split}: {len(ds)} samples x {k} conditions, checkpoint epoch {ck["epoch"]}, '
          f'{len(shards)} shards of {SHARD}', flush=True)
    for j, start in enumerate(shards):
        path = shard_dir / f'{j:03d}.npz'
        if path.exists():
            continue
        sub = torch.utils.data.Subset(ds, range(start, min(start + SHARD, len(ds))))
        dl = torch.utils.data.DataLoader(sub, batch_size=bs, num_workers=workers, pin_memory=True)
        p, f, y, ids = run_shard(model, dl, k, rgb, device, dtype, use_amp)
        np.savez(str(path) + '.tmp.npz', probs=p, feats=f, labels=y, ids=np.array(ids))
        Path(str(path) + '.tmp.npz').rename(path)                         # a reclaimed session never leaves a half shard
        print(f'shard {j + 1}/{len(shards)} done, clean top-1 {(p[:, 0].argmax(1) == y).mean():.4f}', flush=True)

    parts = [np.load(shard_dir / f'{j:03d}.npz') for j in range(len(shards))]
    probs = np.concatenate([q['probs'] for q in parts]); feats = np.concatenate([q['feats'] for q in parts])
    labels = np.concatenate([q['labels'] for q in parts]); ids = np.concatenate([q['ids'] for q in parts])
    assert list(ids) == list(base.df.sample_id), 'shards out of order'
    n, rows = cfg['num_classes'], []
    for ci, (c, s) in enumerate(conditions):
        d = out / condition_name(c, s)
        d.mkdir(exist_ok=True)
        p = probs[:, ci].astype(np.float32)
        p /= p.sum(1, keepdims=True)                                     # renormalise after float16 rounding
        pred = p.argmax(1)
        np.save(d / 'probabilities.npy', probs[:, ci]); np.save(d / 'features.npy', feats[:, ci])
        pd.DataFrame({'sample_id': ids, 'true_label': labels, 'predicted_label': pred, 'confidence': p.max(1),
                      'correct': pred == labels}).to_csv(d / 'predictions.csv', index=False)
        support = np.bincount(labels, minlength=n)
        hits = np.bincount(labels[pred == labels], minlength=n)
        pd.DataFrame({'label': range(n), 'action_id': np.arange(n) + 1, 'support': support,
                      'accuracy': hits / np.maximum(support, 1)}).to_csv(d / 'per_class_accuracy.csv', index=False)
        m = {'run': args.run.name, 'split': args.split, 'corruption': c, 'severity': s, 'num_samples': len(labels),
             **classification_metrics(p, labels, n)}
        json.dump(m, open(d / 'metrics.json', 'w'), indent=2)
        rows.append(m)
    df = pd.DataFrame(rows)
    clean = json.load(open(out / 'clean/metrics.json'))['top1_accuracy']
    df['robustness_ratio'] = df.top1_accuracy / clean
    df.to_csv(out / (f'summary_{args.corruption}.csv' if args.corruption else 'summary.csv'), index=False)
    print(df[['corruption', 'severity', 'top1_accuracy', 'top5_accuracy', 'macro_f1', 'robustness_ratio']]
          .to_string(index=False, float_format='%.4f'))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
