# Milestone 3 revision: response to the review panel (Group A + occlusion evidence)

Status of every Group A item and B5, with the measured numbers. Tables are in `results/milestone3/`.
All results are single test passes of frozen backbones (seed 42); fusion heads add 4 more seeds (A4).

## Summary of what changed in the conclusions

1. **No learned fusion head reliably beats plain probability averaging.** With settings chosen on
   validation instead of test (A1), leak-free CV folds (A2) and 5 head seeds (A4): on X-Sub attention
   reaches 85.86 +- 0.43 mCA against 85.54 for averaging (within noise), concat/global about 82; on
   X-Set averaging is best (86.67) and attention is unstable across seeds (83.66 +- 1.97).
2. **The skeleton collapse is mostly a zero-value artifact.** Dropping 40% of joints costs CTR-GCN 71-80
   points when the joints are set to zero, but only 3-4 points when they are interpolated from neighbouring
   frames (A5). Two black frames cost R3D-18 40 points; two frozen frames cost 2-3 (A5).
3. **Real occlusion does hurt, and fusion still helps.** Under coupled occlusion (one box hides the pixels
   and the joints behind it) fusion falls to about 70% at S5 but stays above both single branches; on the
   natural Kinect-occlusion split the skeleton loses 7-9 points between low- and high-occlusion samples of
   the same class while RGB loses about 2 (B5).
4. **There is real headroom for Milestone 4.** A per-sample oracle that picks whichever branch is right
   reaches 88.9% (X-Sub) / 90.2% (X-Set) mean accuracy over the 30 conditions, 3.4-3.5 points above
   averaging; current learned heads recover at most 2.5% of that gap, and none beats averaging beyond
   seed noise. This is the opening for Milestone 4.

## A1. Selection on validation, not test

- `evaluate_corruptions.py --split val --out .../val_corrupted` builds a corrupted validation set
  (clean + 30 conditions) for all four backbones.
- `scripts/select_fusion_on_val.py` scores every training setting of every head on it; heads trained on
  validation data are scored out of fold with the same grouped folds and stopping epoch as training.
- `scripts/m3_report.py` and `natural_occlusion.py` now report each head in the setting chosen there
  (`fusion_selection_val_<protocol>.md`); test results choose nothing. The README correction of the
  earlier "test set is never used" claim stands.

| Protocol | Head | Setting chosen on validation | Test clean | Test mCA |
|---|---|---|---|---|
| X-Sub | average | (no training) | 89.49 | 85.54 |
| X-Sub | concat | validation + aux | 88.46 | 81.92 |
| X-Sub | global | validation, no aux | 88.41 | 82.41 |
| X-Sub | attention | validation | 90.68 | 86.40 |
| X-Set | average | (no training) | 91.41 | 86.67 |
| X-Set | concat | training features | 91.88 | 85.81 |
| X-Set | global | training features | 91.75 | 86.20 |
| X-Set | attention | validation | 92.57 | 85.57 |

Validation mCA differs by less than 1 point between settings for most heads, so the choice is close.
These are seed-42 numbers; over 5 head seeds (`seed_variance.md`) the X-Sub attention head gives
85.86 +- 0.43 mCA (seed 42 was the best draw) and the X-Set validation-trained attention head
83.66 +- 1.97. Averaging has no trainable parameters and no seed variance.

## A2. Leak-free validation

- X-Set internal validation holds out whole performances (all three camera views of one take);
  both X-Set backbones were retrained on it (R3D-18 87.17%, CTR-GCN 84.93% test top-1; earlier
  87.52% / 86.23% runs kept as `*_m2split` until the Drive clean-up).
- Fusion CV folds are now grouped by performance (`StratifiedGroupKFold`). With per-sample folds every
  held-out validation sample had a sibling view in the fold's training part: out-of-fold accuracy of the
  X-Sub concat head was 96.6% (stopping epoch 75) against 92.9% (epoch 32) with grouped folds and 88.5%
  on test. A gap remains because the 7 X-Sub validation subjects appear in several folds; subject-grouped
  folds are impossible (NTU-120 ties actions to subject groups). Stated as a limitation.

## A3. Validation data budget (for M4)

The corrupted validation set now carries: CV for fusion heads, setting selection, and (in M4) temperature
fitting and reliability-head training. Plan for M4: split the validation performances once into two
disjoint halves (grouped by performance) - one fits reliability/fusion heads, the other fits temperatures
and chooses variants - and apply the same split to every baseline in the M4 table.

## A4. The diagnostic with uncertainty

Seed variance (5 head seeds; mCA mean +- SD, conditions more than 0.5 points below healthy, mean):

| Head (setting) | X-Sub mCA | X-Sub below | X-Set mCA | X-Set below |
|---|---|---|---|---|
| Concat (validation + aux) | 82.03 +- 0.25 | 19.4 | 83.92 +- 0.21 | 17.4 |
| Concat (training features) | 84.30 +- 0.16 | 14.2 | 85.98 +- 0.15 | 13.6 |
| Global (validation + aux) | 81.86 +- 0.18 | 19.2 | 84.22 +- 0.21 | 17.6 |
| Global (training features) | 84.43 +- 0.20 | 12.6 | 86.28 +- 0.16 | 13.6 |
| Attention (validation) | 85.86 +- 0.43 | 8.8 | 83.66 +- 1.97 | 15.0 |
| Attention (training features) | 82.34 +- 0.77 | 8.2 | 82.39 +- 0.79 | 9.6 |
| Average (no training) | 85.54 | - | 86.67 | - |

`scripts/m3_stats.py` (pre-declared setting for every head, so one rule for all): per condition the gap
Acc(fusion) - Acc(healthy) with a 95% paired bootstrap CI and an exact McNemar test, Holm-corrected over
120 cells; `scripts/seed_summary.py` adds 4 more head seeds (`seed_variance.md`).

Conditions (of 30) with fusion significantly below the healthy modality by more than 0.5 points:

| Head | X-Sub | X-Set | Worst gap X-Sub | Worst gap X-Set |
|---|---|---|---|---|
| Average | 6 | 10 | -1.8 (temporal mask S4) | -3.9 (joint dropout S4) |
| Concat | 18 | 17 | -11.7 (temporal mask S5) | -10.1 (joint dropout S5) |
| Global | 16 | 17 | -16.0 (temporal mask S5) | -12.4 (temporal mask S5) |
| Attention | 6 | 12 | -3.0 (noise S5) | -14.9 (joint dropout S4) |

Headroom (mean top-1 over the 30 single-modality conditions, %):

| | Average | Healthy (oracle per condition) | Max-confidence selector | Either branch right (oracle per sample) |
|---|---|---|---|---|
| X-Sub | 85.54 | 84.65 | 85.17 | 88.94 |
| X-Set | 86.67 | 86.05 | 86.33 | 90.20 |

Averaging beats the per-condition oracle on average, so "below the healthy modality" mostly flags the
worst conditions; the per-sample oracle is the meaningful ceiling (+3.4 / +3.5 points).

## A5. Information-matched controls

Same frames / joints as the original corruption, no sentinel values (`rgb_temporal_freeze`,
`skeleton_joint_interp`; tests in `tests/test_corruptions.py`).

| Top-1 (%) | S1 | S3 | S5 |
|---|---|---|---|
| R3D-18 X-Sub, black frames | 45.8 | 35.1 | 12.4 |
| R3D-18 X-Sub, frozen frames | 83.6 | 61.2 | 34.3 |
| CTR-GCN X-Sub, joints zeroed | 65.3 | 34.7 | 12.0 |
| CTR-GCN X-Sub, joints interpolated | 83.3 | 82.1 | 79.5 |
| Attention fusion X-Sub, frozen frames | 90.1 | 86.1 | 79.4 |
| Attention fusion X-Sub, joints interpolated | 90.6 | 90.5 | 90.2 |

Consequence: the zero-fill and black-frame conditions measure sentinel fragility. The paper reports the
benchmark in two layers (sentinel / information-matched), and M4 claims must hold on the second layer.

## A6. Related work, M4 claim, title

Done earlier: `docs/related_work.md` (70 sources). Added since: DRAF-Net (Shen et al., Sensors 2026,
quality-aware RGB-skeleton fusion on Drive&Act; a validation-weighted average of frozen logits beat all
four trained heads there too). Title keeps "occlusion" because B5 is now in place.

## B5. Occlusion evidence

**Coupled occlusion** (RGB box S1-S5 and every joint whose colour-camera projection lies behind it;
hidden joints either zeroed or, as Kinect does, interpolated in time from the frames where they are visible):

| Top-1 (%) | S1 | S3 | S5 |
|---|---|---|---|
| R3D-18 X-Sub (the box alone) | 83.3 | 75.7 | 66.5 |
| CTR-GCN X-Sub, hidden joints zeroed | 76.8 | 60.1 | 41.3 |
| CTR-GCN X-Sub, hidden joints interpolated | 80.4 | 68.5 | 50.0 |
| Average fusion X-Sub (interpolated) | 87.5 | 80.4 | 69.7 |
| Attention fusion X-Sub (interpolated) | 89.1 | 82.9 | 73.5 |
| Average fusion X-Set (interpolated) | 89.6 | 81.7 | 69.6 |
| Attention fusion X-Set (interpolated) | 91.0 | 84.0 | 72.9 |

Unlike random joint dropout (interpolation recovers almost everything), interpolation recovers only 4-9
points here: the box is static, so a joint behind it is usually hidden in every frame and there is
nothing to interpolate from. Coupled occlusion is therefore genuine information loss in both modalities,
and fusion stays above both branches. It is also the one condition family where attention fusion beats
averaging clearly (by 3-4 points at S3-S5 on both protocols; seed 42 only on these conditions).

**Natural occlusion** (Kinect "inferred" joints, 15.8% of joint observations; within each class, samples
above vs below the class median inferred share):

| Top-1 (%) | X-Sub low | X-Sub high | X-Set low | X-Set high |
|---|---|---|---|---|
| RGB only | 87.72 | 85.97 | 87.96 | 86.29 |
| Skeleton only | 88.37 | 81.13 | 89.37 | 80.43 |
| Average fusion | 91.77 | 89.20 | 92.78 | 89.98 |
| Attention fusion | 92.80 | 90.25 | 93.84 | 91.23 |

## Storage

The Drive work store is capped; per-condition fusion probabilities, resumable shards and seed-run
per-sample files are no longer written (README, "What is stored on Drive").

## Still open

- Head-seed variance on the coupled, control and natural-occlusion conditions (the seed runs were
  evaluated on the 30 standard conditions only).
- Group B items are M4 design constraints; Groups C/D are paper-stage.
