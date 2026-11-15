"""Uncertainty and calibration helpers (Milestone 4, steps 4.1-4.4).

Branch outputs are saved as softmax probabilities, so temperature scaling works on log p: softmax(log p / T)
equals softmax(z / T) for the original logits z (the log-sum-exp constant cancels).
"""
import zlib

import numpy as np
from scipy.optimize import minimize_scalar

EPS = 1e-7


def val_half(ids):
    """Review item A3: split the validation split once into two disjoint halves grouped by performance
    (all camera views of a take on one side). 0 = 'fit' (fusion / reliability heads), 1 = 'calib'
    (temperatures, variant choice). Deterministic, the same for every method."""
    return np.array([zlib.crc32((s[:4] + s[8:]).encode()) % 2 for s in ids])


def scale(p, t):
    z = np.log(np.clip(p, EPS, 1)) / t
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(p, y):
    """One temperature per branch: minimise NLL on (clean) validation probabilities."""
    lp = np.log(np.clip(p, EPS, 1))

    def nll(log_t):
        z = lp / np.exp(log_t)
        z -= z.max(1, keepdims=True)
        return float(np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(y)), y]))
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method='bounded').x))


def entropy(p):
    """Normalised entropy U in [0, 1] (step 4.1); reliability R = 1 - U."""
    return -(p * np.log(np.clip(p, EPS, 1))).sum(1) / np.log(p.shape[1])


def ece(p, y, bins=15):
    conf, correct = p.max(1), p.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], EPS, 1)).mean())


def brier(p, y):
    q = p.copy()
    q[np.arange(len(y)), y] -= 1
    return float((q ** 2).sum(1).mean())


def calibration_metrics(p, y):
    return {'accuracy': float((p.argmax(1) == y).mean()), 'ece': ece(p, y), 'nll': nll(p, y), 'brier': brier(p, y)}


def risk_coverage(score, correct, coverages=(1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6)):
    """score: higher = more certain. Returns ({coverage: risk}, AURC) with samples ranked by score
    (step 4.13); AURC is the mean risk over all coverage levels 1/N .. 1."""
    order = np.argsort(-score, kind='stable')
    risk = np.cumsum(~correct[order]) / np.arange(1, len(order) + 1)
    at = {c: float(risk[max(int(round(c * len(order))), 1) - 1]) for c in coverages}
    return at, float(risk.mean())


if __name__ == '__main__':                                 # self-check
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, 4000)
    z = rng.normal(0, 1, (4000, 10)); z[np.arange(4000), y] += 2
    p = np.exp(3 * z); p /= p.sum(1, keepdims=True)        # Bayes posterior is softmax(2z): true T = 1.5
    t = fit_temperature(p, y)
    assert 1.35 < t < 1.65, t
    assert ece(scale(p, t), y) < ece(p, y)
    assert np.allclose(scale(p, 1.0), p, atol=1e-5)
    assert 0 <= entropy(p).min() and entropy(np.full((1, 10), .1))[0] > 0.999
    at, aurc = risk_coverage(p.max(1), p.argmax(1) == y)
    assert at[0.6] <= at[1.0] and 0 < aurc < at[1.0]
    h = val_half(['S001C002P003R001A010', 'S001C003P003R001A010'])
    assert h[0] == h[1]
    print('ok', round(t, 3))
