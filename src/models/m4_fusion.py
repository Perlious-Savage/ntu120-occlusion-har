"""Milestone 4 fusion methods on frozen branch outputs (steps 4.5-4.7 and review item B1).

Every method maps (f_rgb, f_sk, lq_rgb, lq_sk) -> (log-probabilities, weights (B, 2) or None, extra loss), where
f are the frozen branch features and lq the TEMPERATURE-SCALED branch log-probabilities (src/calibration.py).

Not trained:  rgb, skeleton, average (raw softmax mean, the M3 comparator), calibrated_average (0.5 / 0.5 on
              lq: ablation variant B), maxconf (the more confident branch),
              entropy (4.5 logit level: w_m = R_m / (R_rgb + R_sk), R = 1 - normalised entropy).
Trained:      entropy_feat (4.5 feature level: entropy weights mix projected features, then a classifier),
              attention (M3 head), moddrop (concat head trained with modality dropout), tmc (evidential
              Dempster-Shafer fusion, Han et al.), reliability (4.6/4.7, proposed: one reliability net per branch,
              Linear -> GELU -> Dropout -> Linear -> Sigmoid; w_m = r_m / (r_rgb + r_sk) on lq).
Reliability targets (4.7): 'tcp' = calibrated p(y_true); 'correct' = 1[branch argmax == y]. The plan's
target B, exp(-cross-entropy), equals p(y_true) for hard labels, so it would duplicate target A; the binary
correctness target is used as the second target instead.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import FusionHead, projection

K = 120


def norm_entropy(lq):
    return -(lq.exp() * lq).sum(1) / math.log(lq.shape[1])


def mix(w, lr, ls):
    return F.log_softmax(w[:, :1] * lr + w[:, 1:] * ls, 1)


class Fixed(nn.Module):
    trainable = False

    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def forward(self, fr, fs, lr, ls, y=None, raw=None):
        b = len(lr)
        if self.kind == 'rgb':
            return lr, torch.tensor([[1., 0.]], device=lr.device).expand(b, 2), 0
        if self.kind == 'skeleton':
            return ls, torch.tensor([[0., 1.]], device=lr.device).expand(b, 2), 0
        if self.kind == 'average':                         # raw (uncalibrated) softmax mean, as in Milestone 3
            pr, ps = raw
            return torch.log((pr + ps) / 2 + 1e-12), torch.full((b, 2), .5, device=lr.device), 0
        if self.kind == 'calibrated_average':               # ablation B: calibration, fixed 0.5 / 0.5 on lq
            w = torch.full((b, 2), .5, device=lr.device)
            return mix(w, lr, ls), w, 0
        if self.kind == 'maxconf':
            pick = (lr.max(1).values >= ls.max(1).values).float()[:, None]
            w = torch.cat([pick, 1 - pick], 1)
            return w[:, :1] * lr + w[:, 1:] * ls, w, 0
        if self.kind == 'entropy':
            r = torch.stack([1 - norm_entropy(lr), 1 - norm_entropy(ls)], 1).clamp_min(1e-6)
            w = r / r.sum(1, keepdim=True)
            return mix(w, lr, ls), w, 0
        raise ValueError(self.kind)


class EntropyFeat(nn.Module):
    trainable = True

    def __init__(self, d=256, dropout=0.5):
        super().__init__()
        self.pr, self.ps, self.cls = projection(512, d, dropout), projection(256, d, dropout), nn.Linear(d, K)

    def forward(self, fr, fs, lr, ls, y=None, raw=None):
        r = torch.stack([1 - norm_entropy(lr), 1 - norm_entropy(ls)], 1).clamp_min(1e-6)
        w = r / r.sum(1, keepdim=True)
        return F.log_softmax(self.cls(w[:, :1] * self.pr(fr) + w[:, 1:] * self.ps(fs)), 1), w, 0


class Attention(nn.Module):
    trainable = True

    def __init__(self, d=256, dropout=0.5):
        super().__init__()
        self.head = FusionHead('attention', 512 + K, 256 + K, d, K, dropout)

    def forward(self, fr, fs, lr, ls, y=None, raw=None):
        logits, w = self.head(torch.cat([fr, lr], 1), torch.cat([fs, ls], 1))
        return F.log_softmax(logits, 1), w, 0


class ModDrop(nn.Module):
    """Concat head; in training each sample loses its RGB or its skeleton features with probability p each."""
    trainable = True

    def __init__(self, d=256, dropout=0.5, p=0.25):
        super().__init__()
        self.head, self.p = FusionHead('concat', 512, 256, d, K, dropout), p

    def forward(self, fr, fs, lr, ls, y=None, raw=None):
        if self.training:
            u = torch.rand(len(fr), 1, device=fr.device)
            fr, fs = fr * (u >= self.p), fs * ((u < self.p) | (u >= 2 * self.p))
        logits, _ = self.head(fr, fs)
        return F.log_softmax(logits, 1), None, 0


class TMC(nn.Module):
    """Trusted multi-view classification: per-branch Dirichlet evidence, reduced Dempster-Shafer combination,
    loss = expected CE + annealed KL for each branch and the combination (Han et al., ICLR 2021)."""
    trainable = True

    def __init__(self, d=256, dropout=0.5):
        super().__init__()
        self.er = nn.Sequential(projection(512, d, dropout), nn.Linear(d, K), nn.Softplus())
        self.es = nn.Sequential(projection(256, d, dropout), nn.Linear(d, K), nn.Softplus())
        self.anneal = 1.0                                  # set by the trainer: KL weight ramps 0 -> 1 over 10% of training

    @staticmethod
    def combine(a1, a2):
        s1, s2 = a1.sum(1, keepdim=True), a2.sum(1, keepdim=True)
        b1, b2, u1, u2 = (a1 - 1) / s1, (a2 - 1) / s2, K / s1, K / s2
        c = b1.sum(1, keepdim=True) * b2.sum(1, keepdim=True) - (b1 * b2).sum(1, keepdim=True)
        b, u = (b1 * b2 + b1 * u2 + b2 * u1) / (1 - c), u1 * u2 / (1 - c)
        s = K / u
        return b * s + 1, (u1, u2)

    def loss(self, a, y):
        s = a.sum(1)
        ce = torch.digamma(s) - torch.digamma(a.gather(1, y[:, None])[:, 0])
        at = a.scatter(1, y[:, None], 1.0)                # remove the evidence of the true class for the KL term
        st = at.sum(1)
        kl = (torch.lgamma(st) - math.lgamma(K) - torch.lgamma(at).sum(1)
              + ((at - 1) * (torch.digamma(at) - torch.digamma(st)[:, None])).sum(1))
        return (ce + self.anneal * kl).mean()

    def forward(self, fr, fs, lr, ls, y=None, raw=None):
        a1, a2 = self.er(fr) + 1, self.es(fs) + 1
        a, (u1, u2) = self.combine(a1, a2)
        w = torch.cat([1 - u1, 1 - u2], 1)
        w = w / w.sum(1, keepdim=True).clamp_min(1e-6)
        extra = 0 if y is None else self.loss(a1, y) + self.loss(a2, y) + self.loss(a, y)
        return torch.log(a / a.sum(1, keepdim=True)), w, extra


def reliability_net(d_in, d=256, dropout=0.5):
    return nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1), nn.Sigmoid())


class Reliability(nn.Module):
    """Proposed (4.6/4.7): r_m = net_m([f_m ; lq_m]); w = r / sum(r); fused = log_softmax(sum w_m lq_m).
    Loss = CE(fused) [added by the trainer] + lam * MSE(r_m, target_m) for each branch."""
    trainable = True

    def __init__(self, target='tcp', lam=1.0, d=256, dropout=0.5):
        super().__init__()
        self.nr, self.ns, self.target, self.lam = reliability_net(512 + K, d, dropout), reliability_net(256 + K, d, dropout), target, lam

    def forward(self, fr, fs, lr, ls, y=None, raw=None):
        r = torch.cat([self.nr(torch.cat([fr, lr], 1)), self.ns(torch.cat([fs, ls], 1))], 1)
        w = r.clamp_min(1e-4) / r.clamp_min(1e-4).sum(1, keepdim=True)
        extra = 0
        if y is not None:
            if self.target == 'tcp':
                t = torch.stack([lr.gather(1, y[:, None])[:, 0].exp(), ls.gather(1, y[:, None])[:, 0].exp()], 1)
            else:
                t = torch.stack([(lr.argmax(1) == y).float(), (ls.argmax(1) == y).float()], 1)
            extra = self.lam * F.mse_loss(r, t)
        return mix(w, lr, ls), w, extra


def build(name):
    if name in ('rgb', 'skeleton', 'average', 'calibrated_average', 'maxconf', 'entropy'):
        return Fixed(name)
    return {'entropy_feat': EntropyFeat, 'attention': Attention, 'moddrop': ModDrop, 'tmc': TMC,
            'reliability_tcp': lambda: Reliability('tcp'), 'reliability_correct': lambda: Reliability('correct')}[name]()


if __name__ == '__main__':                                 # self-check: shapes, weights sum to 1, losses finite
    torch.manual_seed(0)
    b = 8
    fr, fs = torch.randn(b, 512), torch.randn(b, 256)
    lr, ls = F.log_softmax(torch.randn(b, K), 1), F.log_softmax(torch.randn(b, K), 1)
    y = torch.randint(0, K, (b,))
    raw = (lr.exp(), ls.exp())
    for n in ('rgb', 'skeleton', 'average', 'calibrated_average', 'maxconf', 'entropy', 'entropy_feat', 'attention', 'moddrop', 'tmc',
              'reliability_tcp', 'reliability_correct'):
        m = build(n).train()
        lp, w, extra = m(fr, fs, lr, ls, y, raw)
        assert lp.shape == (b, K) and torch.allclose(lp.exp().sum(1), torch.ones(b), atol=1e-4), n
        assert w is None or torch.allclose(w.sum(1), torch.ones(b), atol=1e-4), n
        loss = F.nll_loss(lp, y) + extra
        assert torch.isfinite(loss), n
        if m.trainable:
            loss.backward()
    a1, a2 = torch.full((1, K), 1.0), torch.full((1, K), 1.0)   # no evidence + no evidence -> uniform, u = 1
    a, (u1, _) = TMC.combine(a1, a2)
    assert torch.allclose(a, torch.ones(1, K)) and torch.allclose(u1, torch.ones(1, 1))
    print('ok')
