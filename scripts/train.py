"""Train a Milestone 2 baseline from a config. Resumes automatically from <out>/last.pt.

Usage:
  python scripts/train.py --config configs/r3d18_xsub.yaml
  python scripts/train.py --config configs/ctrgcn_xsub.yaml --overfit 64 --epochs 60   # tiny-subset overfit test
"""
import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.baseline_datasets import build_dataset  # noqa: E402
from src.engine import (amp_dtype, classification_metrics, lr_factor, predict, seed_everything,  # noqa: E402
                        seed_worker, train_one_epoch)
from src.models.baselines import build_model  # noqa: E402

LOG_FIELDS = ['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'val_macro_f1', 'lr', 'epoch_seconds']


def loader(ds, cfg, shuffle, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=cfg['batch_size'], shuffle=shuffle, generator=g,
                                       num_workers=cfg['num_workers'], worker_init_fn=seed_worker,
                                       pin_memory=True, persistent_workers=cfg['num_workers'] > 0,
                                       drop_last=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--overfit', type=int, default=0, help='train and validate on N training samples, no augmentation')
    ap.add_argument('--epochs', type=int, help='override epochs (e.g. for the overfit test)')
    ap.add_argument('--workers', type=int, help='override num_workers')
    ap.add_argument('--batch-size', type=int, help='override batch_size (small GPUs; recorded in the saved config)')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.epochs:
        cfg['epochs'] = args.epochs
    if args.batch_size:
        cfg['batch_size'] = args.batch_size
    if args.workers is not None:
        cfg['num_workers'] = args.workers
    seed = cfg['seed']
    seed_everything(seed)

    run = f'{cfg["name"]}_seed{seed}' + (f'_overfit{args.overfit}' if args.overfit else '')
    out = Path(cfg['output_dir']) / run
    out.mkdir(parents=True, exist_ok=True)

    if args.overfit:
        subset = pd.read_csv(cfg['splits']['train']).sample(args.overfit, random_state=seed)
        subset.to_csv(out / 'overfit_subset.csv', index=False)
        train_ds = build_dataset(cfg, out / 'overfit_subset.csv', train=False)  # no augmentation
        val_ds = build_dataset(cfg, out / 'overfit_subset.csv', train=False)
        cfg['scheduler'] = {'name': 'constant'}
    else:
        train_ds = build_dataset(cfg, cfg['splits']['train'], train=True)
        val_ds = build_dataset(cfg, cfg['splits']['val'], train=False)
    yaml.safe_dump(cfg, open(out / 'config.yaml', 'w'), sort_keys=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp, dtype = cfg.get('amp', True) and device.type == 'cuda', amp_dtype()
    model = build_model(cfg).to(device)
    o = cfg['optimizer']
    optimizer = torch.optim.SGD(model.parameters(), lr=o['lr'], momentum=o['momentum'],
                                nesterov=o['nesterov'], weight_decay=o['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor(cfg['scheduler'], cfg['epochs']))
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and dtype == torch.float16)

    start, best = 0, {'val_accuracy': -1.0, 'val_macro_f1': -1.0}
    if (out / 'last.pt').exists():
        ck = torch.load(out / 'last.pt', map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        scheduler.load_state_dict(ck['scheduler_state_dict'])
        start, best = ck['epoch'] + 1, ck['best_metric']
        print(f'resumed from {out / "last.pt"} at epoch {start}')
    elif (out / 'training_log.csv').exists():
        (out / 'training_log.csv').unlink()  # stale log without a checkpoint

    train_dl, val_dl = loader(train_ds, cfg, True, seed), loader(val_ds, cfg, False, seed)
    writer = SummaryWriter(out / 'tensorboard')
    print(f'{run}: {len(train_ds)} train / {len(val_ds)} val samples, {cfg["epochs"]} epochs, '
          f'device {device}, amp {use_amp} ({dtype}), seed {seed}', flush=True)

    for epoch in range(start, cfg['epochs']):
        t0 = time.time()
        lr = optimizer.param_groups[0]['lr']
        tr_loss, tr_acc = train_one_epoch(model, train_dl, optimizer, scaler, device, dtype, use_amp)
        scheduler.step()
        probs, labels, _, val_loss = predict(model, val_dl, device, dtype, use_amp)
        m = classification_metrics(probs, labels, cfg['num_classes'])
        row = dict(epoch=epoch, train_loss=tr_loss, train_accuracy=tr_acc, val_loss=val_loss,
                   val_accuracy=m['top1_accuracy'], val_macro_f1=m['macro_f1'], lr=lr, epoch_seconds=time.time() - t0)

        new_log = not (out / 'training_log.csv').exists()
        with open(out / 'training_log.csv', 'a', newline='') as f:
            w = csv.DictWriter(f, LOG_FIELDS)
            if new_log:
                w.writeheader()
            w.writerow(row)
        for k, v in row.items():
            if k != 'epoch':
                writer.add_scalar(k, v, epoch)

        improved = {k: row[k] > best[k] for k in best}
        best = {k: max(best[k], row[k]) for k in best}
        ck = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
              'scheduler_state_dict': scheduler.state_dict(), 'best_metric': best, 'config': cfg, 'seed': seed}
        torch.save(ck, out / 'last.pt.tmp')
        (out / 'last.pt.tmp').replace(out / 'last.pt')  # atomic: a disconnect never leaves a half-written last.pt
        if improved['val_accuracy']:
            shutil.copy(out / 'last.pt', out / 'best_val_accuracy.pt')
        if improved['val_macro_f1']:
            shutil.copy(out / 'last.pt', out / 'best_val_f1.pt')
        print(f'epoch {epoch:3d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | val loss {val_loss:.3f} '
              f'acc {row["val_accuracy"]:.3f} f1 {row["val_macro_f1"]:.3f} | lr {lr:.4g} | {row["epoch_seconds"]:.0f}s'
              + (' *' if improved['val_accuracy'] else ''), flush=True)
    writer.close()
    print(f'done. best val accuracy {best["val_accuracy"]:.4f}, best val macro-F1 {best["val_macro_f1"]:.4f} -> {out}')


if __name__ == '__main__':
    main()
