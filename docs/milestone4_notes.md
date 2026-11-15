# Milestone 4: implementation notes, deviations from the plan, open questions

Status, design decisions and every place the implementation differs from the Milestone 4 guide. Tables are
in `results/milestone4/`.

## What is implemented (frozen Milestone 2 branches, as the General Rules require)

| Plan step | Where | Notes |
|---|---|---|
| 4.1 entropy uncertainty, R = 1 - U | `src/calibration.py` | U = entropy / log(120) |
| 4.2 temperature per branch, validation only | `src/calibration.py`, `scripts/m4_uncertainty.py` | fitted on the clean 'calib' half of validation (A3) |
| 4.3 ECE / NLL / Brier, clean and every severity | `results/milestone4/branch_uncertainty.csv`, `fig_branch_uncertainty_*.png` | |
| 4.4 uncertainty vs branch errors (AUROC) | same | within each condition, plus a separate corruption-presence AUROC (B3) |
| 4.5 uncertainty-guided fusion, logit and feature level | `src/models/m4_fusion.py` (`entropy`, `entropy_feat`) | |
| 4.6 learned reliability head | `Reliability` | see deviations 2 and 5 |
| 4.7 two reliability targets | `reliability_tcp`, `reliability_correct` | see deviation 1 |
| 4.12 Setting A / Setting B | `scripts/m4_fusion.py` | B = leave-one-family-out rotations B0-B2 (review B2) and all families (Ball, final model) |
| 4.13 selective prediction, risk-coverage, AURC | `src/calibration.py` `risk_coverage`, `fig_risk_coverage_*.png` | thresholds set on clean validation; coverage reached under shift reported (B3.3) |
| 4.14 main table | `scripts/m4_report.py` -> `main_table.md` | plus review baselines (B1) and the mixed stream (B4) |
| 4.15 ablation A-D | `ablation.md` | 'Full' waits for the temporal model |
| 4.16 severity curves | `fig_severity_*.png` | |
| 4.18 weight response | `fig_weight_response_*.png` | attention as the reference (B8); part dropout categorical |
| 4.20 seeds | `configs/m4.yaml` seeds 42, 1-4 | trained heads only; untrained methods have no seed |
| 4.23 saved outputs | `m4_fusion.py --save-dir` | weights.csv.gz holds the uncertainty columns too |

## Deviations from the guide (to confirm with the supervisor)

1. **4.7 Target B.** exp(-cross-entropy) equals p(y_true) for hard labels, so it would repeat Target A. The
   second target is the binary "branch prediction correct" instead (same MSE loss).
2. **4.6 reliability network.** A LayerNorm was added in front of Linear -> GELU -> Dropout -> Linear ->
   Sigmoid. Without it the sigmoid saturated on the raw frozen features and both weights stayed 0.5 / 0.5.
3. **4.11 full objective.** The backbones are frozen (General Rules), so the RGB and skeleton auxiliary
   classification losses are constants. The trained loss is fusion CE + lambda * reliability loss (+ temporal
   consistency once 4.10 exists).
4. **4.12 Setting B** trains the fusion / reliability heads on corrupted VALIDATION outputs of the frozen
   branches (the 'fit' half), not by re-training the backbones on corrupted training clips. Every trained
   baseline gets the same corruption-aware training (review B2.3).
5. **Addition: cross-modal reliability** (`reliability_cross_*`): each branch's reliability net also sees the
   other branch's prediction, so disagreement can flag confidently-wrong failures (which entropy misses under
   frozen frames and coupled occlusion, `branch_uncertainty.csv`). Kept as a variant chosen on validation.
6. **4.8 temporal segments: 4 instead of 8 (pending approval).** Measured temporal outputs of the frozen
   backbones on one clip: R3D-18 layer3 = 4 steps, layer4 = 2 steps (16 input frames); CTR-GCN = 16 steps
   (64-frame window). Eight RGB segments would need re-running R3D-18 on sliding windows of each video (a new
   test-time protocol and about 8x the RGB compute). Proposal: 4 segments; RGB segment features from layer3
   (256-d, 4 steps), skeleton from the last block pooled to 4 steps.
7. **Timeline alignment (review B6).** RGB samples 16 frames uniformly over all frames; the skeleton branch
   uses the central 95% of the valid frames resized to 64. Segment k is defined as the fraction [k/S, (k+1)/S)
   of the original video; the skeleton crop shifts the skeleton steps by at most 2.5% of the video length,
   which is stated rather than corrected.

## Questions for the supervisor

1. Is 4 temporal segments acceptable (deviation 6), or must it be 8 with sliding-window RGB re-runs?
2. Is the binary-correctness target acceptable as Target B (deviation 1)?
3. Is head-level corruption-aware training on frozen branches acceptable for Setting B (deviation 4), or
   should the backbones also be fine-tuned on corrupted clips (would change the Milestone 2 baselines)?
4. Cross-modal reliability (deviation 5): it ties with the plain reliability head on validation (X-Sub 87.09 vs
   87.12, X-Set 89.14 vs 89.14 mCA), so the plain head is the proposed method and the cross-modal one an ablation.
