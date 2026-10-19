"""Evaluate a trained baseline on its official test split (Milestone 2, steps 11-12).

Primary results (best_val_accuracy.pt) are written to the run folder:
  test_metrics.json  top-1, top-5, macro-F1 for both best checkpoints, best epochs
  predictions.csv    sample_id, true_label, predicted_label, confidence, correct
  probabilities.npy  (N, 120) softmax, rows in predictions.csv order
  per_class_accuracy.csv, confusion_matrix.npy, confusion_matrix.png, learning_curves.png
Usage: python scripts/evaluate.py --run outputs/r3d18_xsub_seed42
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.baseline_datasets import build_dataset  # noqa: E402
from src.engine import amp_dtype, classification_metrics, predict  # noqa: E402
from src.models.baselines import build_model  # noqa: E402


def evaluate_checkpoint(path, cfg, dl, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model({**cfg, 'model': {**cfg['model'], 'pretrained': False}}).to(device)
    model.load_state_dict(ck['model_state_dict'])
    use_amp = cfg.get('amp', True) and device.type == 'cuda'
    probs, labels, ids, loss = predict(model, dl, device, amp_dtype(), use_amp)
    return ck['epoch'], probs, labels, ids, loss


def plot_curves(log, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(log.epoch, log.train_loss, label='train'); ax[0].plot(log.epoch, log.val_loss, label='val')
    ax[1].plot(log.epoch, log.train_accuracy, label='train'); ax[1].plot(log.epoch, log.val_accuracy, label='val')
    ax[2].plot(log.epoch, log.val_macro_f1, label='val macro-F1'); ax[2].plot(log.epoch, log.lr / log.lr.max(), '--', label='lr (scaled)')
    for a, t in zip(ax, ['loss', 'accuracy', 'val macro-F1']):
        a.set_title(t); a.set_xlabel('epoch'); a.legend(); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, type=Path)
    ap.add_argument('--workers', type=int)
    args = ap.parse_args()
    run = args.run
    cfg = yaml.safe_load(open(run / 'config.yaml'))
    if args.workers is not None:
        cfg['num_workers'] = args.workers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = build_dataset(cfg, cfg['splits']['test'], train=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg['batch_size'], shuffle=False,
                                     num_workers=cfg['num_workers'], pin_memory=True)
    n = cfg['num_classes']

    results = {'run': run.name, 'test_split': cfg['splits']['test'], 'num_test_samples': len(ds)}
    for name in ['best_val_accuracy', 'best_val_f1']:
        epoch, probs, labels, ids, loss = evaluate_checkpoint(run / f'{name}.pt', cfg, dl, device)
        results[name] = {'best_epoch': epoch, 'test_loss': loss, **classification_metrics(probs, labels, n)}
        print(f'{name:18} (epoch {epoch}): ' + ', '.join(f'{k} {v:.4f}' for k, v in results[name].items()
                                                         if k not in ('best_epoch',)), flush=True)
        if name != 'best_val_accuracy':
            continue
        pred = probs.argmax(1)
        pd.DataFrame({'sample_id': ids, 'true_label': labels, 'predicted_label': pred,
                      'confidence': probs.max(1), 'correct': pred == labels}).to_csv(run / 'predictions.csv', index=False)
        np.save(run / 'probabilities.npy', probs.astype(np.float32))
        cm = confusion_matrix(labels, pred, labels=list(range(n)))
        np.save(run / 'confusion_matrix.npy', cm)
        support = cm.sum(1)
        pd.DataFrame({'label': range(n), 'action_id': np.arange(n) + 1, 'support': support,
                      'accuracy': np.where(support > 0, np.diag(cm) / np.maximum(support, 1), np.nan)}
                     ).to_csv(run / 'per_class_accuracy.csv', index=False)
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm / np.maximum(support[:, None], 1), cmap='viridis', vmin=0, vmax=1)
        ax.set_xlabel('predicted label'); ax.set_ylabel('true label')
        ax.set_title(f'{run.name}: row-normalised confusion matrix (test, best val accuracy epoch {epoch})')
        fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout()
        fig.savefig(run / 'confusion_matrix.png', dpi=100); plt.close(fig)

    json.dump(results, open(run / 'test_metrics.json', 'w'), indent=2)
    if (run / 'training_log.csv').exists():
        plot_curves(pd.read_csv(run / 'training_log.csv'), run / 'learning_curves.png')
    print(f'-> {run}/test_metrics.json, predictions.csv, probabilities.npy, confusion matrix, per-class accuracy')


if __name__ == '__main__':
    main()
