"""Training / evaluation loops shared by scripts/train.py and scripts/evaluate.py."""
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    """Give every DataLoader worker its own NumPy stream, derived from the run seed (augmentation uses NumPy)."""
    np.random.seed(torch.initial_seed() % 2**32)


def amp_dtype():
    """bfloat16 where supported (Ampere+ e.g. RTX 40xx, A100, L4); float16 + GradScaler otherwise (T4)."""
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def lr_factor(sched, epochs):
    """Per-epoch LR multiplier: linear warm-up, then step decay (CTR-GCN recipe) or cosine."""
    warm = sched.get('warmup_epochs', 0)

    def f(epoch):
        if epoch < warm:
            return (epoch + 1) / warm
        if sched['name'] == 'step':
            return sched['gamma'] ** sum(epoch >= m for m in sched['milestones'])
        if sched['name'] == 'cosine':
            return 0.5 * (1 + math.cos(math.pi * (epoch - warm) / max(1, epochs - warm)))
        return 1.0  # 'constant'
    return f


def train_one_epoch(model, loader, optimizer, scaler, device, dtype, use_amp):
    model.train()
    total_loss = correct = n = 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=dtype, enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def predict(model, loader, device, dtype, use_amp):
    """-> probabilities (N, C) float32, labels (N,), sample_ids (list), mean loss."""
    model.eval()
    probs, labels, ids, total_loss = [], [], [], 0.0
    for x, y, sid in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=dtype, enabled=use_amp):
            logits = model(x)
        logits = logits.float()
        total_loss += F.cross_entropy(logits, y, reduction='sum').item()
        probs.append(logits.softmax(1).cpu())
        labels.append(y.cpu())
        ids += list(sid)
    probs, labels = torch.cat(probs).numpy(), torch.cat(labels).numpy()
    return probs, labels, ids, total_loss / len(labels)


def classification_metrics(probs, labels, num_classes):
    pred = probs.argmax(1)
    top5 = np.argsort(-probs, axis=1)[:, :5]
    return {
        'top1_accuracy': float((pred == labels).mean()),
        'top5_accuracy': float((top5 == labels[:, None]).any(1).mean()),
        'macro_f1': float(f1_score(labels, pred, average='macro', labels=list(range(num_classes)), zero_division=0)),
    }
