"""Conventional fusion baselines on frozen branch features (Milestone 3, step 3.7).

Both branches are projected to the same latent size D (fixed across experiments):
    h = Dropout(GELU(LayerNorm(Linear(f))))
concat     logits = W [h_rgb ; h_sk]
global     logits = W (a h_rgb + (1 - a) h_sk),  a = sigmoid(learned scalar)
attention  logits = W (w_rgb h_rgb + w_sk h_sk), (w_rgb, w_sk) = softmax(MLP([h_rgb ; h_sk])) per sample
Probability averaging needs no training (scripts/train_fusion.py averages the saved branch softmaxes).
forward() returns (logits, weights) with weights (B, 2) = (w_rgb, w_sk), or None for concat; with
aux=True it also returns per-branch auxiliary logits (U_rgb h_rgb, U_sk h_sk). Training adds their
cross-entropy (config aux_weight) so both projections stay discriminative: without it, attention
fusion saturates onto RGB early (weight 1.0) and the skeleton projection stops receiving gradient.
"""
import torch
import torch.nn as nn


def projection(d_in, d, dropout):
    return nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))


class FusionHead(nn.Module):
    def __init__(self, method, d_rgb=512, d_sk=256, d=256, num_classes=120, dropout=0.5):
        super().__init__()
        if method not in ('concat', 'global', 'attention'):
            raise ValueError(f'unknown fusion method {method!r}')
        self.method = method
        self.proj_rgb, self.proj_sk = projection(d_rgb, d, dropout), projection(d_sk, d, dropout)
        self.cls = nn.Linear(2 * d if method == 'concat' else d, num_classes)
        self.aux_rgb, self.aux_sk = nn.Linear(d, num_classes), nn.Linear(d, num_classes)
        if method == 'global':
            self.alpha = nn.Parameter(torch.zeros(1))       # sigmoid(0) = 0.5: start balanced
        if method == 'attention':
            self.att = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 2))

    def forward(self, f_rgb, f_sk, aux=False):
        h_rgb, h_sk = self.proj_rgb(f_rgb), self.proj_sk(f_sk)
        if self.method == 'concat':
            logits, w = self.cls(torch.cat([h_rgb, h_sk], 1)), None
        else:
            if self.method == 'global':
                w = torch.cat([torch.sigmoid(self.alpha), 1 - torch.sigmoid(self.alpha)]).expand(len(h_rgb), 2)
            else:
                w = self.att(torch.cat([h_rgb, h_sk], 1)).softmax(1)
            logits = self.cls(w[:, :1] * h_rgb + w[:, 1:] * h_sk)
        return (logits, w, (self.aux_rgb(h_rgb), self.aux_sk(h_sk))) if aux else (logits, w)
