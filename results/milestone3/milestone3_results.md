# Milestone 3 results

Top-1 accuracy (%) on the official test splits. Frozen Milestone 2 backbones (seed 42). Fusion heads are trained on clean features only and tested under corruption without retraining. Each is reported in its strongest setting: concat and global on the training features (epoch chosen on validation), attention on the held-out validation split (5-fold CV picks the stopping epoch). The other settings are in `fusion_ablation.csv`. Corruption definitions: `docs/corruption_protocol.md`.

## XSUB

### Single-modality robustness (3.6)

| model | Clean | S1 | S2 | S3 | S4 | S5 | mean S1-S5 | robustness ratio |
|---|---|---|---|---|---|---|---|---|
| R3D-18: RGB occlusion | 85.75 | 83.26 | 79.86 | 75.74 | 71.62 | 66.47 | 75.39 | 0.879 |
| R3D-18: RGB blur | 85.75 | 84.13 | 81.11 | 74.22 | 56.27 | 34.27 | 66.00 | 0.770 |
| R3D-18: RGB temporal mask | 85.75 | 45.76 | 53.27 | 35.07 | 24.57 | 12.35 | 34.21 | 0.399 |
| CTR-GCN: Joint dropout | 83.54 | 65.29 | 55.15 | 34.71 | 20.50 | 12.02 | 37.53 | 0.449 |
| CTR-GCN: Body-part dropout | 83.54 | 80.76 | 60.44 | 65.40 | 6.36 | 3.96 | 43.38 | 0.519 |
| CTR-GCN: Coordinate noise | 83.54 | 58.49 | 52.18 | 41.41 | 19.78 | 11.65 | 36.70 | 0.439 |

### Clean accuracy, mean corruption accuracy (mCA) and robustness ratio

mCA = mean top-1 over S1-S5 of the three corruptions of one modality (the other modality clean); "all" averages all 30 single-modality conditions. Robustness ratio = mCA_all / clean.

| Method | Clean | mCA (RGB corrupted) | mCA (skeleton corrupted) | mCA (all) | Robustness ratio |
|---|---|---|---|---|---|
| RGB only (R3D-18) | 85.75 | 58.53 | 85.75 | 72.14 | 0.841 |
| Skeleton only (CTR-GCN) | 83.54 | 83.54 | 39.21 | 61.37 | 0.735 |
| Average fusion | 89.49 | 85.17 | 85.91 | 85.54 | 0.956 |
| Concat fusion | 89.88 | 83.41 | 84.99 | 84.20 | 0.937 |
| Global-weight fusion | 89.84 | 83.24 | 85.96 | 84.60 | 0.942 |
| Attention fusion | 89.93 | 83.61 | 86.09 | 84.85 | 0.944 |

### Diagnostic: fusion vs the healthy modality (3.9)

Healthy = the uncorrupted modality alone. **Bold** = fusion below the healthy modality, i.e. Acc(fusion) < Acc(healthy). Conditions below healthy (of 30): Average fusion 15, Concat fusion 16, Global-weight fusion 15, Attention fusion 13.

| Condition | RGB only (R3D-18) | Skeleton only (CTR-GCN) | Average fusion | Concat fusion | Global-weight fusion | Attention fusion |
|---|---|---|---|---|---|---|
| RGB occlusion S1 | 83.26 | 83.54 | 88.60 | 89.03 | 88.98 | 89.01 |
| RGB occlusion S2 | 79.86 | 83.54 | 87.54 | 87.83 | 87.59 | 87.54 |
| RGB occlusion S3 | 75.74 | 83.54 | 86.68 | 86.69 | 86.43 | 86.26 |
| RGB occlusion S4 | 71.62 | 83.54 | 85.93 | 85.66 | 85.32 | 85.12 |
| RGB occlusion S5 | 66.47 | 83.54 | 85.19 | 84.39 | 84.06 | 83.55 |
| RGB blur S1 | 84.13 | 83.54 | 88.70 | 89.28 | 89.16 | 89.32 |
| RGB blur S2 | 81.11 | 83.54 | 87.75 | 88.15 | 88.01 | 88.25 |
| RGB blur S3 | 74.22 | 83.54 | 86.09 | 86.11 | 85.98 | 86.52 |
| RGB blur S4 | 56.27 | 83.54 | 84.17 | **82.11** | **81.48** | **83.21** |
| RGB blur S5 | 34.27 | 83.54 | **83.51** | **77.51** | **76.36** | **79.48** |
| RGB temporal mask S1 | 45.76 | 83.54 | **82.52** | **81.13** | **80.69** | **81.18** |
| RGB temporal mask S2 | 53.27 | 83.54 | 84.22 | **82.78** | **83.34** | 84.05 |
| RGB temporal mask S3 | 35.07 | 83.54 | **82.68** | **79.42** | **79.81** | **80.84** |
| RGB temporal mask S4 | 24.57 | 83.54 | **81.74** | **76.92** | **77.53** | **76.55** |
| RGB temporal mask S5 | 12.35 | 83.54 | **82.24** | **74.05** | **73.85** | **73.33** |
| Joint dropout S1 | 85.75 | 65.29 | 86.63 | 87.26 | 87.46 | 87.31 |
| Joint dropout S2 | 85.75 | 55.15 | 86.00 | 86.26 | 86.68 | 86.46 |
| Joint dropout S3 | 85.75 | 34.71 | **85.25** | **84.17** | **85.32** | **85.65** |
| Joint dropout S4 | 85.75 | 20.50 | **85.07** | **83.11** | **84.65** | **85.26** |
| Joint dropout S5 | 85.75 | 12.02 | **84.69** | **82.14** | **83.96** | **85.22** |
| Body-part dropout S1 | 85.75 | 80.76 | 89.01 | 89.54 | 89.53 | 89.50 |
| Body-part dropout S2 | 85.75 | 60.44 | 86.31 | 86.41 | 87.18 | 86.29 |
| Body-part dropout S3 | 85.75 | 65.40 | 87.74 | 88.22 | 88.53 | 87.90 |
| Body-part dropout S4 | 85.75 | 6.36 | **85.35** | **81.99** | **83.85** | **85.55** |
| Body-part dropout S5 | 85.75 | 3.96 | **85.58** | **82.27** | **83.93** | **85.70** |
| Coordinate noise S1 | 85.75 | 58.49 | **85.61** | 86.89 | 87.19 | 87.31 |
| Coordinate noise S2 | 85.75 | 52.18 | **85.34** | **85.72** | 86.35 | 86.66 |
| Coordinate noise S3 | 85.75 | 41.41 | **85.32** | **84.51** | **85.53** | 85.86 |
| Coordinate noise S4 | 85.75 | 19.78 | **85.31** | **83.40** | **84.74** | **84.13** |
| Coordinate noise S5 | 85.75 | 11.65 | **85.43** | **83.00** | **84.57** | **82.49** |

### Dual corruption (3.10)

Both modalities corrupted independently.

| Condition | RGB only (R3D-18) | Skeleton only (CTR-GCN) | Average fusion | Concat fusion | Global-weight fusion | Attention fusion |
|---|---|---|---|---|---|---|
| RGB blur S1 + Coordinate noise S4 | 84.13 | 19.78 | 83.60 | 81.50 | 83.18 | 82.18 |
| RGB blur S2 + Coordinate noise S2 | 81.11 | 52.18 | 81.32 | 82.86 | 83.55 | 83.64 |
| RGB blur S3 + Coordinate noise S3 | 74.22 | 41.41 | 74.61 | 75.64 | 76.85 | 77.30 |
| RGB blur S4 + Coordinate noise S1 | 56.27 | 58.49 | 68.89 | 74.73 | 74.77 | 74.82 |
| RGB blur S4 + Coordinate noise S4 | 56.27 | 19.78 | 54.78 | 55.28 | 57.20 | 55.61 |
| RGB occlusion S1 + Body-part dropout S4 | 83.26 | 6.36 | 82.70 | 79.21 | 80.76 | 83.01 |
| RGB occlusion S2 + Body-part dropout S2 | 79.86 | 60.44 | 82.10 | 82.43 | 83.12 | 82.43 |
| RGB occlusion S3 + Body-part dropout S3 | 75.74 | 65.40 | 82.82 | 83.61 | 83.92 | 83.02 |
| RGB occlusion S4 + Body-part dropout S1 | 71.62 | 80.76 | 84.77 | 84.82 | 84.53 | 84.15 |
| RGB occlusion S4 + Body-part dropout S4 | 71.62 | 6.36 | 70.30 | 65.70 | 67.46 | 71.23 |
| RGB temporal mask S1 + Joint dropout S4 | 45.76 | 20.50 | 47.77 | 52.43 | 53.97 | 54.10 |
| RGB temporal mask S2 + Joint dropout S2 | 53.27 | 55.15 | 66.69 | 70.18 | 70.76 | 71.51 |
| RGB temporal mask S3 + Joint dropout S3 | 35.07 | 34.71 | 47.00 | 51.95 | 52.03 | 52.97 |
| RGB temporal mask S4 + Joint dropout S1 | 24.57 | 65.29 | 64.23 | 63.11 | 63.10 | 64.23 |
| RGB temporal mask S4 + Joint dropout S4 | 24.57 | 20.50 | 31.64 | 36.08 | 37.58 | 35.18 |

### Class-level modality analysis (3.12)

Per-class clean accuracy; dominant = gap of at least 5 percentage points (about 2 standard errors of a per-class difference at ~400-500 test samples per class). Counts: balanced 78, RGB-dominant 29, skeleton-dominant 13. Full list: `class_modality_xsub.csv`.

| Action | Name | Test samples | RGB acc | Skeleton acc | RGB - skeleton (pts) | Class type |
|---|---|---|---|---|---|---|
| 30 | typing on a keyboard | 275 | 91.64 | 64.73 | +26.9 | RGB-dominant |
| 75 | cutting nails | 569 | 82.60 | 62.57 | +20.0 | RGB-dominant |
| 11 | reading | 273 | 75.46 | 58.24 | +17.2 | RGB-dominant |
| 94 | throw up cap/hat | 573 | 97.03 | 82.55 | +14.5 | RGB-dominant |
| 17 | take off a shoe | 274 | 83.21 | 69.34 | +13.9 | RGB-dominant |
| 106 | hit other person with something | 575 | 73.74 | 61.39 | +12.3 | RGB-dominant |
| 85 | apply cream on face | 574 | 95.12 | 82.93 | +12.2 | RGB-dominant |
| 12 | writing | 272 | 65.44 | 53.31 | +12.1 | RGB-dominant |
| 5 | drop | 275 | 98.18 | 86.55 | +11.6 | RGB-dominant |
| 73 | staple book | 571 | 38.88 | 28.02 | +10.9 | RGB-dominant |
| 82 | fold paper | 575 | 60.00 | 65.74 | -5.7 | skeleton-dominant |
| 76 | cutting paper (using scissors) | 573 | 49.91 | 56.02 | -6.1 | skeleton-dominant |
| 49 | use a fan (with hand or paper)/feeling warm | 275 | 86.55 | 92.73 | -6.2 | skeleton-dominant |
| 90 | take something out of a bag | 576 | 80.38 | 86.81 | -6.4 | skeleton-dominant |
| 18 | wear on glasses | 273 | 87.18 | 93.77 | -6.6 | skeleton-dominant |
| 78 | open bottle | 573 | 65.45 | 73.12 | -7.7 | skeleton-dominant |
| 34 | rub two hands together | 276 | 79.71 | 88.04 | -8.3 | skeleton-dominant |
| 77 | snapping fingers | 574 | 59.41 | 68.47 | -9.1 | skeleton-dominant |
| 81 | toss a coin | 573 | 78.71 | 88.31 | -9.6 | skeleton-dominant |
| 104 | stretch oneself | 576 | 81.42 | 92.71 | -11.3 | skeleton-dominant |

## XSET

### Single-modality robustness (3.6)

| model | Clean | S1 | S2 | S3 | S4 | S5 | mean S1-S5 | robustness ratio |
|---|---|---|---|---|---|---|---|---|
| R3D-18: RGB occlusion | 87.52 | 84.78 | 81.10 | 76.30 | 71.50 | 65.92 | 75.92 | 0.867 |
| R3D-18: RGB blur | 87.52 | 85.71 | 82.60 | 74.75 | 54.25 | 32.36 | 65.94 | 0.753 |
| R3D-18: RGB temporal mask | 87.52 | 51.26 | 48.48 | 25.97 | 21.03 | 10.87 | 31.52 | 0.360 |
| CTR-GCN: Joint dropout | 86.23 | 68.91 | 57.79 | 32.04 | 13.70 | 5.67 | 35.62 | 0.413 |
| CTR-GCN: Body-part dropout | 86.23 | 83.79 | 62.08 | 65.66 | 5.42 | 2.77 | 43.94 | 0.510 |
| CTR-GCN: Coordinate noise | 86.23 | 57.68 | 49.66 | 29.43 | 10.24 | 5.10 | 30.42 | 0.353 |

### Clean accuracy, mean corruption accuracy (mCA) and robustness ratio

mCA = mean top-1 over S1-S5 of the three corruptions of one modality (the other modality clean); "all" averages all 30 single-modality conditions. Robustness ratio = mCA_all / clean.

| Method | Clean | mCA (RGB corrupted) | mCA (skeleton corrupted) | mCA (all) | Robustness ratio |
|---|---|---|---|---|---|
| RGB only (R3D-18) | 87.52 | 57.79 | 87.52 | 72.65 | 0.830 |
| Skeleton only (CTR-GCN) | 86.23 | 86.23 | 36.66 | 61.45 | 0.713 |
| Average fusion | 91.91 | 87.38 | 87.59 | 87.49 | 0.952 |
| Concat fusion | 92.18 | 85.99 | 86.80 | 86.40 | 0.937 |
| Global-weight fusion | 92.08 | 85.47 | 87.84 | 86.65 | 0.941 |
| Attention fusion | 92.86 | 87.76 | 87.78 | 87.77 | 0.945 |

### Diagnostic: fusion vs the healthy modality (3.9)

Healthy = the uncorrupted modality alone. **Bold** = fusion below the healthy modality, i.e. Acc(fusion) < Acc(healthy). Conditions below healthy (of 30): Average fusion 14, Concat fusion 15, Global-weight fusion 16, Attention fusion 10.

| Condition | RGB only (R3D-18) | Skeleton only (CTR-GCN) | Average fusion | Concat fusion | Global-weight fusion | Attention fusion |
|---|---|---|---|---|---|---|
| RGB occlusion S1 | 84.78 | 86.23 | 91.17 | 91.29 | 91.13 | 92.23 |
| RGB occlusion S2 | 81.10 | 86.23 | 90.31 | 90.12 | 89.93 | 91.40 |
| RGB occlusion S3 | 76.30 | 86.23 | 89.48 | 89.25 | 88.66 | 90.72 |
| RGB occlusion S4 | 71.50 | 86.23 | 88.79 | 87.99 | 87.29 | 89.83 |
| RGB occlusion S5 | 65.92 | 86.23 | 87.96 | 86.97 | **86.03** | 89.07 |
| RGB blur S1 | 85.71 | 86.23 | 91.34 | 91.52 | 91.49 | 92.43 |
| RGB blur S2 | 82.60 | 86.23 | 90.53 | 90.57 | 90.40 | 91.75 |
| RGB blur S3 | 74.75 | 86.23 | 89.06 | 88.27 | 88.04 | 90.31 |
| RGB blur S4 | 54.25 | 86.23 | 86.96 | **84.09** | **83.63** | 87.84 |
| RGB blur S5 | 32.36 | 86.23 | **86.21** | **80.27** | **78.91** | **86.10** |
| RGB temporal mask S1 | 51.26 | 86.23 | 86.83 | **85.25** | **85.27** | 87.26 |
| RGB temporal mask S2 | 48.48 | 86.23 | **85.15** | **84.99** | **84.71** | 86.48 |
| RGB temporal mask S3 | 25.97 | 86.23 | **81.04** | **81.26** | **79.84** | **81.14** |
| RGB temporal mask S4 | 21.03 | 86.23 | **82.55** | **80.47** | **79.64** | **80.57** |
| RGB temporal mask S5 | 10.87 | 86.23 | **83.36** | **77.55** | **77.15** | **79.24** |
| Joint dropout S1 | 87.52 | 68.91 | 88.71 | 89.64 | 89.87 | 90.11 |
| Joint dropout S2 | 87.52 | 57.79 | 87.76 | 88.35 | 88.74 | 88.96 |
| Joint dropout S3 | 87.52 | 32.04 | **87.14** | **86.36** | **87.27** | 87.69 |
| Joint dropout S4 | 87.52 | 13.70 | **86.88** | **84.39** | **86.12** | **86.80** |
| Joint dropout S5 | 87.52 | 5.67 | **86.70** | **83.31** | **85.60** | **86.49** |
| Body-part dropout S1 | 87.52 | 83.79 | 91.43 | 91.81 | 91.85 | 92.57 |
| Body-part dropout S2 | 87.52 | 62.08 | 87.72 | 88.80 | 89.32 | 89.53 |
| Body-part dropout S3 | 87.52 | 65.66 | 89.54 | 90.37 | 90.65 | 91.20 |
| Body-part dropout S4 | 87.52 | 5.42 | **86.51** | **84.28** | **85.76** | **81.45** |
| Body-part dropout S5 | 87.52 | 2.77 | **87.02** | **84.42** | **85.69** | **83.02** |
| Coordinate noise S1 | 87.52 | 57.68 | 87.64 | 88.91 | 89.08 | 89.19 |
| Coordinate noise S2 | 87.52 | 49.66 | **87.33** | 87.52 | 88.27 | 88.75 |
| Coordinate noise S3 | 87.52 | 29.43 | **86.94** | **85.86** | **87.09** | 87.73 |
| Coordinate noise S4 | 87.52 | 10.24 | **86.36** | **84.24** | **86.22** | **86.86** |
| Coordinate noise S5 | 87.52 | 5.10 | **86.15** | **83.76** | **86.01** | **86.43** |

### Dual corruption (3.10)

Both modalities corrupted independently.

| Condition | RGB only (R3D-18) | Skeleton only (CTR-GCN) | Average fusion | Concat fusion | Global-weight fusion | Attention fusion |
|---|---|---|---|---|---|---|
| RGB blur S1 + Coordinate noise S4 | 85.71 | 10.24 | 84.53 | 82.54 | 84.80 | 85.60 |
| RGB blur S2 + Coordinate noise S2 | 82.60 | 49.66 | 83.44 | 84.99 | 85.63 | 86.42 |
| RGB blur S3 + Coordinate noise S3 | 74.75 | 29.43 | 74.90 | 75.69 | 77.44 | 79.08 |
| RGB blur S4 + Coordinate noise S1 | 54.25 | 57.68 | 69.43 | 76.46 | 75.71 | 77.07 |
| RGB blur S4 + Coordinate noise S4 | 54.25 | 10.24 | 47.90 | 49.99 | 52.67 | 54.69 |
| RGB occlusion S1 + Body-part dropout S4 | 84.78 | 5.42 | 83.58 | 80.90 | 82.69 | 77.99 |
| RGB occlusion S2 + Body-part dropout S2 | 81.10 | 62.08 | 83.60 | 84.61 | 85.04 | 85.83 |
| RGB occlusion S3 + Body-part dropout S3 | 76.30 | 65.66 | 84.23 | 85.16 | 85.33 | 87.26 |
| RGB occlusion S4 + Body-part dropout S1 | 71.50 | 83.79 | 87.67 | 87.10 | 86.49 | 88.88 |
| RGB occlusion S4 + Body-part dropout S4 | 71.50 | 5.42 | 69.21 | 64.57 | 67.09 | 61.25 |
| RGB temporal mask S1 + Joint dropout S4 | 51.26 | 13.70 | 51.17 | 51.67 | 54.83 | 56.56 |
| RGB temporal mask S2 + Joint dropout S2 | 48.48 | 57.79 | 64.74 | 70.02 | 70.29 | 71.52 |
| RGB temporal mask S3 + Joint dropout S3 | 25.97 | 32.04 | 36.60 | 45.96 | 44.64 | 45.84 |
| RGB temporal mask S4 + Joint dropout S1 | 21.03 | 68.91 | 64.53 | 65.12 | 63.68 | 65.37 |
| RGB temporal mask S4 + Joint dropout S4 | 21.03 | 13.70 | 24.21 | 28.69 | 29.67 | 30.67 |

### Class-level modality analysis (3.12)

Per-class clean accuracy; dominant = gap of at least 5 percentage points (about 2 standard errors of a per-class difference at ~400-500 test samples per class). Counts: balanced 89, RGB-dominant 19, skeleton-dominant 12. Full list: `class_modality_xset.csv`.

| Action | Name | Test samples | RGB acc | Skeleton acc | RGB - skeleton (pts) | Class type |
|---|---|---|---|---|---|---|
| 11 | reading | 500 | 81.00 | 61.60 | +19.4 | RGB-dominant |
| 94 | throw up cap/hat | 487 | 96.10 | 80.08 | +16.0 | RGB-dominant |
| 75 | cutting nails | 481 | 77.55 | 62.99 | +14.6 | RGB-dominant |
| 17 | take off a shoe | 502 | 86.06 | 71.51 | +14.5 | RGB-dominant |
| 12 | writing | 498 | 65.46 | 52.81 | +12.7 | RGB-dominant |
| 106 | hit other person with something | 491 | 80.24 | 68.23 | +12.0 | RGB-dominant |
| 5 | drop | 502 | 98.61 | 87.45 | +11.2 | RGB-dominant |
| 30 | typing on a keyboard | 502 | 83.47 | 72.51 | +11.0 | RGB-dominant |
| 28 | make a phone call/answer phone | 501 | 93.01 | 83.03 | +10.0 | RGB-dominant |
| 85 | apply cream on face | 489 | 93.46 | 84.46 | +9.0 | RGB-dominant |
| 72 | make victory sign | 490 | 46.12 | 51.63 | -5.5 | skeleton-dominant |
| 45 | touch chest (stomachache/heart pain) | 504 | 77.38 | 83.93 | -6.5 | skeleton-dominant |
| 83 | ball up paper | 491 | 67.82 | 74.95 | -7.1 | skeleton-dominant |
| 74 | counting money | 486 | 60.08 | 67.90 | -7.8 | skeleton-dominant |
| 81 | toss a coin | 489 | 77.91 | 86.91 | -9.0 | skeleton-dominant |
| 86 | apply cream on hand back | 489 | 74.64 | 83.64 | -9.0 | skeleton-dominant |
| 76 | cutting paper (using scissors) | 488 | 46.93 | 56.35 | -9.4 | skeleton-dominant |
| 10 | clapping | 500 | 75.40 | 87.60 | -12.2 | skeleton-dominant |
| 77 | snapping fingers | 489 | 61.15 | 76.28 | -15.1 | skeleton-dominant |
| 82 | fold paper | 491 | 61.51 | 83.30 | -21.8 | skeleton-dominant |

## Classes with the same dominance on both protocols

typing on a keyboard (RGB-dominant), cutting nails (RGB-dominant), reading (RGB-dominant), throw up cap/hat (RGB-dominant), take off a shoe (RGB-dominant), hit other person with something (RGB-dominant), apply cream on face (RGB-dominant), writing (RGB-dominant), drop (RGB-dominant), blow nose (RGB-dominant), kicking other person (RGB-dominant), whisper in other person's ear (RGB-dominant), play magic cube (skeleton-dominant), fold paper (skeleton-dominant), cutting paper (using scissors) (skeleton-dominant), snapping fingers (skeleton-dominant), toss a coin (skeleton-dominant)
