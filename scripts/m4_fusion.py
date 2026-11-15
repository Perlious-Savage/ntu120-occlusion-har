"""Milestone 4 fusion on frozen branch outputs: train every method in Setting A (clean) and Setting B
(corruption-aware, leave-one-family-out rotations B0-B2), then test on the Milestone 3 benchmark
(configs/fusion.yaml conditions) and on the mixed-condition stream (B4).

One process per protocol loads all branch outputs once (GPU-resident, float16).
Output: <output_dir>/<protocol>_<setting>_seed<seed>.csv, one row per method x test condition:
  accuracy, macro_f1, ece, nll, brier, aurc, risk@coverage, coverage/risk at thresholds set on clean
  calib validation, mean_w_rgb.
Usage: python scripts/m4_fusion.py --protocol xsub [--settings A B0 B1 B2 Ball] [--seeds 42 1 2 3 4] [--methods ...]
(seeds other than config seed rerun only the trained methods)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from src.calibration import brier, ece, fit_temperature, nll, risk_coverage, val_half  # noqa: E402
from src.corruptions.base import SEVERITIES  # noqa: E402
from src.models.m4_fusion import build, norm_entropy  # noqa: E402
from train_fusion import test_conditions  # noqa: E402

K = 120


class Bank:
    """All conditions of one split and modality stacked: f (C, N, d) and p (C, N, K) float16 on the device."""

    def __init__(self, run_dir, conds, dev):
        f, p = [], []
        for c in conds:
            d = run_dir / c
            f.append(torch.from_numpy(np.load(d / 'features.npy')).half())
            p.append(torch.from_numpy(np.load(d / 'probabilities.npy')).half())
            if c == conds[0]:
                pred = pd.read_csv(d / 'predictions.csv')
                self.y, self.ids = pred.true_label.to_numpy(), pred.sample_id.to_numpy()
        self.f, self.p = torch.stack(f).to(dev), torch.stack(p).to(dev)
        self.index = {c: i for i, c in enumerate(conds)}


class Data:
    def __init__(self, rgb, sk, t, dev):
        self.rgb, self.sk, self.t = rgb, sk, t
        assert (rgb.ids == sk.ids).all()
        self.y = torch.from_numpy(rgb.y).to(dev)

    def gather(self, cr, cs, rows):
        out = []
        for b, c, t in ((self.rgb, cr, self.t[0]), (self.sk, cs, self.t[1])):
            p = b.p[c, rows].float()
            p = p / p.sum(1, keepdim=True)
            out.append((b.f[c, rows].float(), F.log_softmax(torch.log(p.clamp_min(1e-7)) / t, 1), p))
        (fr, lr, pr), (fs, ls, ps) = out
        return fr, fs, lr, ls, (pr, ps)


@torch.no_grad()
def infer(m, data, cr, cs, rows, bs=8192):
    m.eval()
    lps, ws = [], []
    for i in range(0, len(rows), bs):
        fr, fs, lr, ls, raw = data.gather(cr[i:i + bs], cs[i:i + bs], rows[i:i + bs])
        lp, w, _ = m(fr, fs, lr, ls, raw=raw)
        lps.append(lp.float()); ws.append(None if w is None else w.float())
    return torch.cat(lps), (None if ws[0] is None else torch.cat(ws))


def train(name, data, rows, sampler, cfg, seed, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    m = build(name).to(dev)
    if not m.trainable:
        return m
    o, bs, epochs = cfg['optimizer'], cfg['batch_size'], cfg['epochs']
    opt = torch.optim.AdamW(m.parameters(), lr=o['lr'], weight_decay=o['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * int(np.ceil(len(rows) / bs)))
    gen = np.random.default_rng(seed)
    m.log = []
    for ep in range(epochs):
        m.train()
        if hasattr(m, 'anneal'):
            m.anneal = min(1.0, ep / (0.1 * epochs))       # TMC: KL annealing (Han et al.: 50 of 500 epochs)
        cr, cs = sampler(len(rows), gen)
        perm = torch.from_numpy(gen.permutation(len(rows))).to(dev)
        tot = ce = right = 0.0
        for idx in perm.split(bs):
            r = rows[idx]
            fr, fs, lr, ls, raw = data.gather(cr[idx], cs[idx], r)
            lp, _, extra = m(fr, fs, lr, ls, y=data.y[r], raw=raw)
            c = F.nll_loss(lp, data.y[r])
            loss = c + extra
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(idx); ce += c.item() * len(idx); right += (lp.argmax(1) == data.y[r]).sum().item()
        m.log.append({'epoch': ep, 'loss': tot / len(rows), 'fusion_ce': ce / len(rows),
                      'extra_loss': (tot - ce) / len(rows), 'train_accuracy': right / len(rows)})
    return m


def save_outputs(d, m, name, cfg, setting, seed, t, test, conds, stream, rows):
    """Step 4.23 for one final run: config.yaml, checkpoint.pt, metrics.json, training_log.csv, predictions.csv and
    probabilities.npy (clean test and the mixed stream), weights.csv.gz (every test condition, one row per sample;
    segment_id 0 = whole sequence until the temporal model, 4.8). U = normalised entropy of the calibrated branch."""
    import json
    d.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump({**cfg, 'method': name, 'setting': setting, 'seed': seed, 'temperatures': t}, open(d / 'config.yaml', 'w'), sort_keys=False)
    if m.trainable:
        torch.save({'model_state_dict': m.state_dict(), 'method': name, 'setting': setting}, d / 'checkpoint.pt')
        pd.DataFrame(m.log).to_csv(d / 'training_log.csv', index=False)
    json.dump(rows, open(d / 'metrics.json', 'w'), indent=1)
    n, ids, y = len(test.y), test.rgb.ids, test.rgb.y
    allr = torch.arange(n, device=test.y.device)
    parts, preds = [], []
    for rc, sc in conds + [('mixed', 'mixed')]:
        if rc == 'mixed':
            cr, cs = stream
        else:
            cr = torch.full((n,), test.rgb.index[rc], device=allr.device); cs = torch.full((n,), test.sk.index[sc], device=allr.device)
        lp, w = infer(m, test, cr, cs, allr)
        with torch.no_grad():
            u = [torch.cat([norm_entropy(test.gather(cr[i:i + 8192], cs[i:i + 8192], allr[i:i + 8192])[k])
                            for i in range(0, n, 8192)]).cpu().numpy() for k in (2, 3)]
        p = lp.exp().cpu().numpy(); pred = p.argmax(1)
        if rc == 'mixed':
            names_r = np.array(sorted(test.rgb.index, key=test.rgb.index.get))[cr.cpu().numpy()]
            names_s = np.array(sorted(test.sk.index, key=test.sk.index.get))[cs.cpu().numpy()]
        else:
            names_r, names_s = np.full(n, rc), np.full(n, sc)
        wr = np.full(n, np.nan) if w is None else w[:, 0].cpu().numpy()
        df = pd.DataFrame({'sample_id': ids, 'segment_id': 0, 'stream': 'mixed' if rc == 'mixed' else 'fixed',
                           'rgb_condition': names_r, 'skeleton_condition': names_s, 'true_label': y, 'predicted_label': pred,
                           'confidence': p.max(1), 'rgb_uncertainty': u[0], 'skeleton_uncertainty': u[1],
                           'rgb_weight': wr, 'skeleton_weight': 1 - wr})
        parts.append(df)
        if rc in ('clean', 'mixed'):
            preds.append(df.assign(correct=df.predicted_label == df.true_label))
            np.save(d / f'probabilities_{rc}.npy', p.astype(np.float16))
    pd.concat(preds).to_csv(d / 'predictions.csv', index=False)
    pd.concat(parts).to_csv(d / 'weights.csv.gz', index=False, float_format='%.4f')


def metrics(lp, w, y, taus, coverages):
    p = lp.exp()
    conf, pred = p.max(1)
    p, y_, conf, pred = p.cpu().numpy(), y.cpu().numpy(), conf.cpu().numpy(), pred.cpu().numpy()
    correct = pred == y_
    at, aurc = risk_coverage(conf, correct, coverages)
    row = {'accuracy': float(correct.mean()),
           'macro_f1': float(f1_score(y_, pred, average='macro', labels=list(range(K)), zero_division=0)),
           'ece': ece(p, y_), 'nll': nll(p, y_), 'brier': brier(p, y_), 'aurc': aurc,
           **{f'risk@{c}': v for c, v in at.items()},
           'mean_w_rgb': float('nan') if w is None else float(w[:, 0].mean()),
           'std_w_rgb': float('nan') if w is None else float(w[:, 0].std())}
    for c, tau in taus.items():
        keep = conf >= tau
        row[f'coverage_at_t{c}'] = float(keep.mean())
        row[f'risk_at_t{c}'] = float((~correct[keep]).mean()) if keep.any() else float('nan')
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/m4.yaml')
    ap.add_argument('--protocol', required=True, choices=['xsub', 'xset'])
    ap.add_argument('--settings', nargs='+', default=['A', 'B0', 'B1', 'B2'])
    ap.add_argument('--seeds', nargs='+', type=int)
    ap.add_argument('--methods', nargs='+')
    ap.add_argument('--save-dir', type=Path, help='also write the 4.23 per-sample outputs of every run (final runs only)')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    fcfg = yaml.safe_load(open(ROOT / 'configs/fusion.yaml'))
    methods, seeds = args.methods or cfg['methods'], args.seeds or cfg['seeds']
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base = {m: Path(cfg['features_dir']) / r.format(protocol=args.protocol) for m, r in cfg['branch_runs'].items()}
    fam = cfg['families']
    t0 = time.time()

    # validation: clean + the three standard families per modality
    vconds = {m: ['clean'] + [f'{c}_s{s}' for c in fam[m] for s in SEVERITIES] for m in fam}
    vr, vs = (Bank(base[m] / 'val_corrupted', vconds[m], dev) for m in ('rgb', 'skeleton'))
    half = val_half(vr.ids)
    t = []
    for b in (vr, vs):                                     # one temperature per branch, clean calib half
        p = b.p[0].float().cpu().numpy(); p /= p.sum(1, keepdims=True)
        t.append(fit_temperature(p[half == 1], b.y[half == 1]))
    val = Data(vr, vs, t, dev)
    fit_rows = torch.from_numpy(np.flatnonzero(half == 0)).to(dev)
    calib_rows = torch.from_numpy(np.flatnonzero(half == 1)).to(dev)

    conds = test_conditions(fcfg)
    tr = Bank(base['rgb'] / 'test', ['clean'] + sorted({rc for rc, _ in conds} - {'clean'}), dev)
    ts = Bank(base['skeleton'] / 'test', ['clean'] + sorted({sc for _, sc in conds} - {'clean'}), dev)
    test = Data(tr, ts, t, dev)
    n = len(test.y)
    print(f'{args.protocol}: T = {t[0]:.3f} / {t[1]:.3f}; fit {len(fit_rows)}, calib {len(calib_rows)}, '
          f'test {n} x {len(conds)} conditions; loaded in {time.time() - t0:.0f} s', flush=True)

    # mixed stream (B4)
    g = np.random.default_rng(cfg['mixed_stream']['seed'])
    state = g.choice(4, n, p=cfg['mixed_stream']['prior'])
    pick = lambda m: np.array([tr.index[f'{c}_s{s}'] if m == 'rgb' else ts.index[f'{c}_s{s}']  # noqa: E731
                               for c, s in zip(g.choice(fam[m], n), g.choice(SEVERITIES, n))])
    mr, ms = pick('rgb'), pick('skeleton')
    stream_r = torch.from_numpy(np.where((state == 1) | (state == 3), mr, 0)).to(dev)
    stream_s = torch.from_numpy(np.where((state == 2) | (state == 3), ms, 0)).to(dev)
    all_rows = torch.arange(n, device=dev)

    out = Path(cfg['output_dir']); out.mkdir(parents=True, exist_ok=True)
    for setting in args.settings:
        if setting == 'A':
            sampler = lambda k, gen: (torch.zeros(k, dtype=torch.long, device=dev),) * 2  # noqa: E731
            held = None
        else:                                              # B0-B2: leave one family per modality out; Ball: none
            held = None if setting == 'Ball' else (fam['rgb'][int(setting[1])], fam['skeleton'][int(setting[1])])
            ok = {m: [vr.index[c] if m == 'rgb' else vs.index[c] for c in vconds[m][1:]
                      if held is None or not c.startswith(held[m != 'rgb'])] for m in fam}

            def sampler(k, gen, ok=ok):
                st = gen.choice(3, k, p=cfg['setting_b_prior'])
                cr = np.where(st == 1, gen.choice(ok['rgb'], k), 0)
                cs = np.where(st == 2, gen.choice(ok['skeleton'], k), 0)
                return torch.from_numpy(cr).to(dev), torch.from_numpy(cs).to(dev)
        for seed in seeds:
            if (out / f'{args.protocol}_{setting}_seed{seed}.csv').exists() and not args.save_dir:
                print(f'{setting} seed {seed}: done earlier, skipped', flush=True)   # resume after a lost session
                continue
            rows = []                                      # untrained methods do not depend on the head seed
            for name in methods if seed == cfg['seed'] or args.methods else [x for x in methods if x in cfg['trained_methods']]:
                t1 = time.time()
                m = train(name, val, fit_rows, sampler, cfg, seed, dev)
                z = torch.zeros(len(calib_rows), dtype=torch.long, device=dev)
                lp, _ = infer(m, val, z, z, calib_rows)
                conf = lp.exp().max(1).values.cpu().numpy()
                taus = {c: float(np.quantile(conf, 1 - c)) for c in cfg['selective_targets']}
                # variant choice (A3): mean top-1 on the calib half over the 30 single-modality validation conditions
                accs = []
                for mod, b in (('rgb', vr), ('skeleton', vs)):
                    for i in range(1, len(b.index)):
                        ci = torch.full((len(calib_rows),), i, device=dev)
                        lpv, _ = infer(m, val, *((ci, z) if mod == 'rgb' else (z, ci)), calib_rows)
                        accs.append(float((lpv.argmax(1) == val.y[calib_rows]).float().mean()))
                calib_mca = float(np.mean(accs))
                for rc, sc in conds + [('mixed', 'mixed')]:
                    if rc == 'mixed':
                        cr, cs = stream_r, stream_s
                    else:
                        cr = torch.full((n,), tr.index[rc], device=dev); cs = torch.full((n,), ts.index[sc], device=dev)
                    lp, w = infer(m, test, cr, cs, all_rows)
                    rows.append({'protocol': args.protocol, 'setting': setting, 'seed': seed, 'method': name,
                                 'held_out': '' if held is None else '+'.join(held),
                                 'rgb_condition': rc, 'skeleton_condition': sc, 'calib_val_mca': calib_mca,
                                 **metrics(lp, w, test.y, taus, cfg['coverages'])})
                r = {x['rgb_condition'] + '|' + x['skeleton_condition']: x['accuracy'] for x in rows if x['method'] == name}
                if args.save_dir:
                    save_outputs(args.save_dir / f'{name}_{args.protocol}_{setting}_seed{seed}', m, name, cfg, setting, seed, t,
                                 test, conds, (stream_r, stream_s), [x for x in rows if x['method'] == name])
                print(f'{setting} seed {seed} {name:>20}: clean {r["clean|clean"]:.4f}  mixed {r["mixed|mixed"]:.4f}  '
                      f'({time.time() - t1:.0f} s)', flush=True)
            pd.DataFrame(rows).to_csv(out / f'{args.protocol}_{setting}_seed{seed}.csv', index=False)


if __name__ == '__main__':
    main()
