"""Supervisor progress report (PDF): Milestones 1-4 status, results and the decisions needed.
Reads the committed result files only. Usage: python scripts/make_progress_report.py [--out docs/progress_report.pdf]
"""
import argparse
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

R3, R4 = Path('results/milestone3'), Path('results/milestone4')
TRAINED = set(__import__('yaml').safe_load(open('configs/m4.yaml'))['trained_methods'])
ss = getSampleStyleSheet()
H1, H2, H3 = ss['Heading1'], ss['Heading2'], ss['Heading3']
B = ParagraphStyle('b', parent=ss['BodyText'], fontSize=9.5, leading=13)
SM = ParagraphStyle('sm', parent=B, fontSize=8, leading=10)
CELL = ParagraphStyle('cell', parent=B, fontSize=7.6, leading=9.2)
CAP = ParagraphStyle('cap', parent=SM, textColor=colors.HexColor('#444444'))
W = 17.5 * cm


def P(t, s=B):
    return Paragraph(t, s)


def table(rows, widths=None, head=1, hl=()):
    data = [[P(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=head)
    st = [('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#999999')),
          ('BACKGROUND', (0, 0), (-1, head - 1), colors.HexColor('#e3e9f2')),
          ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('TOPPADDING', (0, 0), (-1, -1), 1.5),
          ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5)]
    st += [('BACKGROUND', (0, r), (-1, r), colors.HexColor('#fff4d6')) for r in hl]
    t.setStyle(TableStyle(st))
    return t


def fig(path, width=W, caption=None):
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(str(path)).getSize()
    out = [Image(str(path), width=width, height=width * h / w)]
    if caption:
        out.append(P(caption, CAP))
    return KeepTogether(out + [Spacer(1, 6)])


def pct(x, d=1):
    return '' if pd.isna(x) else f'{100 * x:.{d}f}'


def bullets(items):
    return [P(f'&bull; {i}') for i in items]


NICE = {'rgb': 'RGB only (R3D-18)', 'skeleton': 'Skeleton only (CTR-GCN)', 'average': 'Average (M3 baseline)',
        'calibrated_average': 'Calibrated average', 'maxconf': 'Max-confidence selector',
        'entropy': 'Entropy-weighted (4.5, logit)', 'entropy_feat': 'Entropy-weighted (4.5, feature)',
        'attention': 'Attention (M3 head)', 'moddrop': 'Modality-dropout head', 'tmc': 'TMC (evidential)',
        'reliability_tcp': 'Proposed: reliability, target p(y)', 'reliability_correct': 'Proposed: reliability, target 1[correct]',
        'reliability_cross_tcp': 'Proposed + cross-modal, p(y)', 'reliability_cross_correct': 'Proposed + cross-modal, 1[correct]'}


def m4_main(protocol, t):
    """Setting B for trained heads (corruption-aware, mean over the leave-one-family-out rotations), untrained
    methods have no training; every number is a mean over the head seeds present (+- SD of mCA / mixed)."""
    rows = [['Method', 'Clean', 'mCA (30 cond.)', 'Out-of-family', 'Coupled occl.', 'Mixed stream', 'Mixed ECE',
             'Mixed AURC', 'Clean ECE', 'Seeds']]
    g = t[(t.protocol == protocol) & (t.setting == 'B')]
    best = g.loc[g.mCA.idxmax(), 'method']
    hl = []
    for i, r in enumerate(g.itertuples(), start=1):
        sd = lambda v: '' if pd.isna(v) or r.n_seeds < 2 else f' &plusmn; {100 * v:.2f}'  # noqa: E731
        trained = r.method in TRAINED
        rows.append([NICE.get(r.method, r.method), pct(r.clean_acc, 2), pct(r.mCA, 2) + sd(r.mCA_sd),
                     pct(r.out_family_mCA, 2) if trained else '(no training)', pct(r.coupled_interp, 2),
                     pct(r.mixed_acc, 2) + sd(r.mixed_acc_sd), f'{r.mixed_ece:.3f}', f'{r.mixed_aurc:.4f}',
                     f'{r.clean_ece:.3f}', int(r.n_seeds)])
        if r.method == best or str(r.method).startswith('reliability'):
            hl.append(i)
    return table(rows, [3.9 * cm, 1.2 * cm, 2.2 * cm, 1.8 * cm, 1.4 * cm, 2.2 * cm, 1.3 * cm, 1.3 * cm, 1.2 * cm, 1.0 * cm],
                 hl=hl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='docs/progress_report.pdf')
    args = ap.parse_args()
    s = []
    s += [P('Occlusion-Robust RGB-Skeleton Human Action Recognition on NTU RGB+D 120', ss['Title']),
          P('Progress report: Milestones 1-4, results, and decisions needed from the supervisor', H3),
          P('Gagan Saideep (2023A7PS0269U) &nbsp;&middot;&nbsp; 16 November 2026 &nbsp;&middot;&nbsp; '
            'code: github.com/Perlious-Savage/ntu120-occlusion-har'), Spacer(1, 8)]

    s += [P('1. Summary', H1),
          P('Milestones 1-3 are complete, including a revision of Milestone 3 that answers an internal review '
            '(selection on validation only, leak-free cross-validation, seed variance and significance tests, '
            'information-matched control corruptions, and coupled / natural occlusion). Milestone 4 is about '
            'two-thirds done: uncertainty, calibration, uncertainty-guided and learned-reliability fusion, '
            'corruption-aware training, selective prediction, the main comparison table, the ablation and most '
            'analysis figures exist for both protocols. The temporal part (steps 4.8-4.10, 4.19) is waiting for '
            'decision Q1 below.'), Spacer(1, 4),
          table([['Milestone', 'Status', 'Key outcome'],
                 ['M1 Data pipeline', 'Done', '113,945 usable samples; official X-Sub / X-Set splits; RGB and skeleton frames verified one-to-one'],
                 ['M2 Single-modality baselines', 'Done', 'R3D-18 85.75 / 87.17%, CTR-GCN 83.54 / 84.93% (X-Sub / X-Set top-1)'],
                 ['M3 Corruption benchmark + fusion baselines', 'Done (revised)', '6 corruptions x 5 severities + 4 extra conditions; '
                  'no learned fusion head beats probability averaging beyond seed noise'],
                 ['M4 Uncertainty-aware dynamic fusion', 'In progress', 'Learned reliability with corruption-aware training: '
                  'best mCA on both protocols (see Section 5); temporal model pending Q1']],
                [4.2 * cm, 2.2 * cm, 11.1 * cm]), Spacer(1, 6)]

    s += [P('2. Decisions needed', H1),
          P('Each question has the context needed to answer it; my recommendation is given in italics.')]
    qs = [('Q1. Temporal segments (step 4.8: "start with 8 temporal segments").',
           'The frozen R3D-18 receives 16 frames; its feature maps keep 4 time steps after layer 3 and 2 after layer 4 '
           '(measured). CTR-GCN keeps 16 steps for its 64-frame window. Eight RGB segments are therefore not available '
           'from the frozen backbone; they would need R3D-18 re-run on 8 sliding windows of each video (a new test-time '
           'protocol, about 8x the RGB compute, and the Milestone 2 RGB model was never trained on such short windows).',
           '<i>Use 4 segments: RGB segment features from layer 3 (256-d, 4 steps), skeleton from its last block pooled to '
           '4 steps, segments defined on the original video timeline.</i>'),
          ('Q2. Second reliability target (step 4.7: "Target B: exp(-cross-entropy loss)").',
           'For a single correct label, cross-entropy = -log p(y_true), so exp(-CE) = p(y_true), which is Target A. '
           'The two targets would be identical.',
           '<i>Use the binary target 1[branch prediction is correct] as Target B (implemented; results in Section 5).</i>'),
          ('Q3. Setting B, corruption-aware training (step 4.12).',
           'The General Rules forbid changing the accepted Milestone 2 backbones, so Setting B trains the fusion and '
           'reliability heads on corrupted outputs of the frozen backbones (validation clips), not the backbones '
           'themselves. Following the review, families are held out in rotation and every trained baseline gets the '
           'same corruption-aware training.',
           '<i>Keep head-level corruption-aware training; report backbone fine-tuning as future work.</i>'),
          ('Q4. Cross-modal reliability (an addition to step 4.6).',
           'Under realistic corruptions (frozen frames, coupled occlusion) a branch is often confidently wrong, which '
           'its own uncertainty cannot reveal (Section 5.1). A variant lets each reliability network also see the other '
           "branch's prediction, so disagreement can lower trust. It is selected on validation like any other variant.",
           '<i>Allow it as the proposed method if it wins on validation; otherwise report it as an ablation.</i>')]
    for q, ctx, rec in qs:
        s += [KeepTogether([P(f'<b>{q}</b>'), P(ctx), P(rec), Spacer(1, 4)])]

    s += [PageBreak(), P('3. Milestones 1 and 2', H1),
          table([['Item', 'Result'],
                 ['Samples', '114,480 IDs, all with RGB and skeleton; 113,945 usable after removing NTU\'s 535 missing-skeleton samples'],
                 ['X-Sub', 'train 63,026 / test 50,919 (internal train 56,594 / val 6,432)'],
                 ['X-Set', 'train 54,468 / test 59,477 (internal train 49,021 / val 5,447, grouped by performance)'],
                 ['Synchronisation', 'RGB and skeleton frame counts identical for every sample (15-300 frames, mean 71.6)'],
                 ['Skeleton denoising', 'official noise-body rules; 3,296 samples cleaned']], [3.2 * cm, 14.3 * cm]),
          Spacer(1, 6),
          table([['Model', 'Protocol', 'Top-1', 'Top-5', 'Macro-F1'],
                 ['R3D-18 (RGB)', 'X-Sub', '85.75', '98.02', '0.867'], ['R3D-18 (RGB)', 'X-Set', '87.17', '98.37', '0.870'],
                 ['CTR-GCN (joint)', 'X-Sub', '83.54', '96.79', '0.838'], ['CTR-GCN (joint)', 'X-Set', '84.93', '97.02', '0.851']],
                [4 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm]),
          P('Clean test accuracy (%), seed 42, best-validation checkpoint. RGB wins on object/appearance actions, '
            'skeleton on motion-defined ones; both are weakest on fine hand gestures.', CAP)]

    s += [P('4. Milestone 3: corruption benchmark and fusion baselines', H1),
          P('Six corruptions with five severities each, applied to one modality while the other stays clean: RGB '
            'occlusion, blur, temporal mask; skeleton joint dropout, body-part dropout, coordinate noise. Extra '
            'conditions added after review: information-matched controls (frozen frames instead of black frames; '
            'interpolated instead of zeroed joints) and coupled occlusion (one occluder hides both the pixels and the '
            'joints behind it). Fusion heads are trained on clean data and tested without retraining.')]
    m = pd.read_csv(R3 / 'mca.csv')
    rows = [['Method', 'X-Sub clean', 'X-Sub mCA', 'X-Set clean', 'X-Set mCA']]
    for meth in m.method.unique():
        a, b = (m[(m.protocol == p) & (m.method == meth)].iloc[0] for p in ('xsub', 'xset'))
        rows.append([meth.replace('_', ' '), pct(a.clean, 2), pct(a.mCA_all, 2), pct(b.clean, 2), pct(b.mCA_all, 2)])
    s += [table(rows, [4 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm]),
          P('Seed-42 heads in the setting chosen on corrupted validation. mCA = mean top-1 over the 30 single-modality '
            'conditions. Over 5 head seeds, X-Sub attention gives 85.86 &plusmn; 0.43 vs 85.54 for averaging.', CAP),
          fig(R3 / 'fig_accuracy_vs_rgb_severity.png', 14 * cm, 'Milestone 3: single-branch and fusion accuracy vs RGB corruption severity.'),
          P('<b>Main findings of the revised Milestone 3</b>')]
    s += bullets(['No learned fusion head reliably beats parameter-free probability averaging (5 seeds, paired bootstrap and '
                  'McNemar tests with Holm correction).',
                  'Most of the skeleton "collapse" is an artifact of zero-valued joints: dropping 40% of joints costs CTR-GCN '
                  '71 points when zeroed but 4 points when interpolated; two black frames cost R3D-18 40 points, two frozen '
                  'frames 2-3.',
                  'Coupled occlusion is genuine information loss (CTR-GCN 50% at S5 even with interpolation); fusion stays '
                  'above both branches.',
                  'Natural occlusion (Kinect inferred joints): skeleton accuracy drops 7-9 points from low- to high-occlusion '
                  'samples of the same class, RGB about 2.',
                  'A per-sample oracle choosing the right branch reaches 88.9 / 90.2% mean accuracy, 3.4-3.5 points above '
                  'averaging: the headroom Milestone 4 targets.'])

    s += [PageBreak(), P('5. Milestone 4: uncertainty-aware dynamic fusion (in progress)', H1),
          P('All Milestone 4 models use the frozen Milestone 2 branches. Validation is split once into two halves grouped '
            'by performance: one trains fusion/reliability heads, the other fits temperatures, thresholds and chooses '
            'variants. The test set is never used for any choice.')]
    steps = [('4.1 Entropy uncertainty baseline', 'Done'), ('4.2 Temperature calibration (validation only)', 'Done'),
             ('4.3 ECE / NLL / Brier, clean + all severities', 'Done'), ('4.4 Uncertainty vs branch errors (AUROC)', 'Done'),
             ('4.5 Uncertainty-guided fusion, logit + feature level', 'Done'), ('4.6 Learned reliability head', 'Done'),
             ('4.7 Two reliability targets', 'Done (Q2)'), ('4.8-4.10 Temporal reliability, fusion, consistency loss', 'Waiting for Q1'),
             ('4.11 Configurable full objective', 'Done for the non-temporal terms'),
             ('4.12 Setting A (clean) vs Setting B (corruption-aware)', 'Done (Q3)'),
             ('4.13 Selective prediction, risk-coverage, AURC', 'Done'), ('4.14 Main comparison table', 'Done (non-temporal)'),
             ('4.15 Ablation A-D', 'Done; "Full" waits for 4.8-4.10'), ('4.16 Severity curves X-Sub / X-Set', 'Done'),
             ('4.17 Per-class modality weights (top 20)', 'Next (outputs being generated)'), ('4.18 Weight response to corruption', 'Done'),
             ('4.19 Temporal visualisation', 'Waiting for Q1'), ('4.20 Seeds (mean &plusmn; SD)', 'Done: 5 head seeds'),
             ('4.21 Error analysis (30 failure cases)', 'Next'), ('4.22 Complexity (params, FLOPs, latency)', 'Next'),
             ('4.23 Saved outputs per run', 'Implemented')]
    s += [table([['Plan step', 'Status']] + [list(x) for x in steps], [11 * cm, 6.5 * cm])]

    bu = pd.read_csv(R4 / 'branch_uncertainty.csv')
    tp = pd.read_csv(R4 / 'temperatures.csv')
    s += [P('5.1 Calibration and uncertainty (4.1-4.4)', H2),
          table([['Protocol', 'Branch', 'Temperature', 'Val ECE before', 'Val ECE after', 'Val NLL before', 'Val NLL after']] +
                [[r.protocol, r.branch, f'{r.temperature:.3f}', f'{r.calib_ece_raw:.3f}', f'{r.calib_ece_cal:.3f}',
                  f'{r.calib_nll_raw:.3f}', f'{r.calib_nll_cal:.3f}'] for r in tp.itertuples()],
                [2 * cm, 2 * cm, 2.3 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm]), Spacer(1, 4)]
    x = bu[(bu.protocol == 'xsub') & (bu.severity.isin([3, 5]))]
    rows = [['Corruption (X-Sub)', 'Sev.', 'Top-1', 'ECE (cal.)', 'Error AUROC', 'Corruption AUROC', 'Confidently wrong']]
    for r in x.sort_values(['branch', 'corruption', 'severity']).itertuples():
        rows.append([r.corruption, r.severity, pct(r.accuracy), f'{r.ece:.3f}', f'{r.error_auroc:.3f}',
                     f'{r.corruption_auroc:.3f}', pct(r.confidently_wrong_share, 0) + '%'])
    s += [table(rows, [5 * cm, 1 * cm, 1.6 * cm, 1.9 * cm, 2.1 * cm, 2.7 * cm, 2.6 * cm]),
          P('Error AUROC: how well the branch\'s calibrated entropy separates its own errors from correct predictions '
            'within one condition. Corruption AUROC: how well it separates corrupted from clean inputs. Confidently wrong '
            '= share of errors made with calibrated confidence &ge; 0.5.', CAP),
          P('<b>Finding.</b> Temperature scaling fixes clean calibration (ECE 0.05-0.08 to 0.02-0.03). Sentinel '
            'corruptions (black frames, zeroed joints) are easy to detect from entropy (corruption AUROC 0.92-0.99), '
            'but the realistic ones are not: under frozen frames, coupled occlusion and interpolated joints, '
            '38-50% of the errors are confidently wrong. Entropy alone cannot flag them, which motivates the learned '
            'and cross-modal reliability (Q4).'),
          fig(R4 / 'fig_branch_uncertainty_xsub.png', 13 * cm, 'X-Sub: accuracy, mean uncertainty and calibrated ECE vs severity per branch.')]

    t = pd.read_csv(R4 / 'main_table.csv')
    s += [PageBreak(), P('5.2 Main comparison (4.14)', H2),
          P('Trained heads use Setting B (corruption-aware, mean over three leave-one-family-out rotations); '
            '"out-of-family" is accuracy on the held-out RGB and skeleton family, which the head never saw corrupted. '
            'Mixed stream: every test sample independently clean / RGB / skeleton / both corrupted (40/20/20/20%, '
            'random family and severity). Highlighted: the proposed variants and the best mCA. Accuracies in %.')]
    for p in ('xsub', 'xset'):
        s += [P(f'<b>{"X-Sub" if p == "xsub" else "X-Set"}</b>'), m4_main(p, t), Spacer(1, 6)]
    ab = (R4 / 'ablation.md').read_text().splitlines()
    rows = [[c.strip() for c in line.strip('|').split('|')] for line in ab if line.startswith('|') and not line.startswith('|---')]
    s += [P('5.3 Ablation (4.15)', H2), table(rows, [1.6 * cm, 1.4 * cm, 4.2 * cm, 3.3 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm, 1.6 * cm, 1.6 * cm]),
          P('A = plain average; B = + per-branch temperature; C = + dynamic entropy weights; D = + learned reliability '
            '(target chosen on validation). Setting A trains on clean data, Setting B corruption-aware.', CAP)]

    s += [PageBreak(), P('5.4 Mechanism and selective prediction (4.13, 4.16, 4.18)', H2),
          fig(R4 / 'fig_weight_response_xsub.png', W, 'Weight response (X-Sub): the RGB weight falls when RGB is corrupted '
              '(solid) and rises when the skeleton is corrupted (dashed), as step 4.18 requires.'),
          fig(R4 / 'fig_severity_xset.png', W, 'Accuracy vs severity (X-Set) for attention, entropy-weighted, average and '
              'the proposed fusion.'),
          fig(R4 / 'fig_risk_coverage_xset.png', 15 * cm, 'Risk-coverage (X-Set) on clean data and on the mixed stream.')]

    s += [P('6. Next steps', H1)]
    s += bullets(['After Q1: extract segment-level features (4 or 8 segments), train temporal reliability with and without the '
                  'consistency loss (4.8-4.10), complete the "Full" ablation row and the temporal visualisations with '
                  'non-sentinel bursts (frozen frames, coupled occluder entering and leaving).',
                  'Per-class modality weights (4.17), error analysis of at least 30 failure cases (4.21), complexity table (4.22).',
                  'Package final tables and figures for the paper; README with exact commands.'])
    s += [P('Appendix: deviations from the Milestone 4 guide', H2)]
    s += bullets(['4.6: a LayerNorm in front of the reliability MLP (without it the sigmoid saturated on the raw frozen features '
                  'and both weights stayed exactly 0.5).',
                  '4.7: Target B replaced by 1[correct] (Q2).',
                  '4.8: 4 instead of 8 segments, pending Q1.',
                  '4.11: frozen backbones make the RGB / skeleton auxiliary losses constant; trained loss = fusion CE + '
                  'lambda x reliability loss (+ temporal consistency once 4.10 exists).',
                  '4.12: corruption-aware training at head level (Q3), with leave-one-family-out evaluation and matched training '
                  'for all trained baselines.',
                  'Additions from the internal review: extra baselines (max-confidence selector, TMC, modality dropout), the '
                  'mixed-condition test stream, out-of-family columns, coverage reached under shift.'])

    doc = SimpleDocTemplate(args.out, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                            bottomMargin=1.6 * cm, title='NTU120 occlusion-robust HAR: progress report',
                            author='Gagan Saideep')
    doc.build(s, onLaterPages=lambda c, d: (c.setFont('Helvetica', 7.5), c.drawRightString(A4[0] - 1.8 * cm, 1 * cm, f'page {d.page}')))
    print('->', args.out)


if __name__ == '__main__':
    main()
