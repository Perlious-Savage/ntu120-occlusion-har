"""Conventional fusion baselines on frozen branch features (Milestone 3, step 3.7).

Both branches are projected to the same latent size D (fixed across experiments):
    h = Dropout(GELU(LayerNorm(Linear(f))))
concat     logits = W [h_rgb ; h_sk]
global     logits = W (a h_rgb + (1 - a) h_sk),  a = sigmoid(learned scalar)
attention  logits = w_rgb log p_rgb + w_sk log p_sk,  (w_rgb, w_sk) = softmax(MLP([h_rgb ; h_sk])) per sample,
           where p are the FROZEN branch softmaxes. Its inputs carry log p as their last num_classes
           columns (scripts/train_fusion.py appends them). Average fusion is the special case of fixed
           weights 0.5 / 0.5 (on probabilities).
Probability averaging needs no training (scripts/train_fusion.py averages the saved branch softmaxes).
forward() returns (logits, weights) with weights (B, 2) = (w_rgb, w_sk), or None for concat; with
aux=True it also returns per-branch logits (U_rgb h_rgb, U_sk h_sk) for the auxiliary losses
(config aux_weight) of concat and global.

Why attention weighs the frozen predictions (5-fold CV on the validation split only,
scripts/compare_attention_variants.py): heads that mix trainable branch outputs are fitted on the
same samples the gate learns from, where the refitted RGB head is almost always right, so the gate
saturates onto RGB (weight 1.000). That happened in every fold when mixing projected features, and in
3 of 5 X-Set folds when mixing refitted branch logits, also with zero-initialised or detached gates.
The frozen backbones never saw the validation split, so their errors there are real.
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
        self.method, self.num_classes = method, num_classes
        if method == 'attention':                           # inputs = [features ; log p]
            d_rgb, d_sk = d_rgb - num_classes, d_sk - num_classes
        self.proj_rgb, self.proj_sk = projection(d_rgb, d, dropout), projection(d_sk, d, dropout)
        if method == 'attention':
            self.att = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 2))
            nn.init.zeros_(self.att[-1].weight); nn.init.zeros_(self.att[-1].bias)   # start at 0.5 / 0.5
            return
        self.cls = nn.Linear(2 * d if method == 'concat' else d, num_classes)
        self.aux_rgb, self.aux_sk = nn.Linear(d, num_classes), nn.Linear(d, num_classes)
        if method == 'global':
            self.alpha = nn.Parameter(torch.zeros(1))       # sigmoid(0) = 0.5: start balanced

    def forward(self, f_rgb, f_sk, aux=False):
        if self.method == 'attention':
            c = self.num_classes
            lp_rgb, lp_sk = f_rgb[:, -c:], f_sk[:, -c:]
            w = self.att(torch.cat([self.proj_rgb(f_rgb[:, :-c]), self.proj_sk(f_sk[:, :-c])], 1)).softmax(1)
            logits = w[:, :1] * lp_rgb + w[:, 1:] * lp_sk
            return (logits, w, None) if aux else (logits, w)
        h_rgb, h_sk = self.proj_rgb(f_rgb), self.proj_sk(f_sk)
        if self.method == 'concat':
            logits, w = self.cls(torch.cat([h_rgb, h_sk], 1)), None
        else:
            w = torch.cat([torch.sigmoid(self.alpha), 1 - torch.sigmoid(self.alpha)]).expand(len(h_rgb), 2)
            logits = self.cls(w[:, :1] * h_rgb + w[:, 1:] * h_sk)
        return (logits, w, (self.aux_rgb(h_rgb), self.aux_sk(h_sk))) if aux else (logits, w)
