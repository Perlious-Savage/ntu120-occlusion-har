# Milestone 3 results

Top-1 accuracy (%) on the official test splits. Frozen Milestone 2 backbones (seed 42). Fusion heads are trained on clean features only and tested under corruption without retraining. Each is reported in the training setting chosen on corrupted validation data (`fusion_selection_val_<protocol>.md`; test results choose nothing). CV folds are grouped by performance. Corruption definitions: `docs/corruption_protocol.md`.

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
| Concat fusion | 88.46 | 81.46 | 82.38 | 81.92 | 0.926 |
| Global-weight fusion | 88.41 | 81.16 | 83.65 | 82.41 | 0.932 |
| Attention fusion | 90.68 | 86.49 | 86.30 | 86.40 | 0.953 |

### Diagnostic: fusion vs the healthy modality (3.9)

Healthy = the uncorrupted modality alone. **Bold** = fusion below the healthy modality, i.e. Acc(fusion) < Acc(healthy). Conditions below healthy (of 30): Average fusion 15, Concat fusion 21, Global-weight fusion 19, Attention fusion 10.

| Condition | RGB only (R3D-18) | Skeleton only (CTR-GCN) | Average fusion | Concat fusion | Global-weight fusion | Attention fusion |
|---|---|---|---|---|---|---|
| RGB occlusion S1 | 83.26 | 83.54 | 88.60 | 87.63 | 87.48 | 90.14 |
| RGB occlusion S2 | 79.86 | 83.54 | 87.54 | 86.48 | 86.33 | 89.21 |
| RGB occlusion S3 | 75.74 | 83.54 | 86.68 | 85.11 | 84.78 | 88.46 |
| RGB occlusion S4 | 71.62 | 83.54 | 85.93 | 84.08 | 83.75 | 87.76 |
| RGB occlusion S5 | 66.47 | 83.54 | 85.19 | **82.77** | **82.00** | 87.08 |
| RGB blur S1 | 84.13 | 83.54 | 88.70 | 87.79 | 87.72 | 90.26 |
| RGB blur S2 | 81.11 | 83.54 | 87.75 | 86.60 | 86.48 | 89.46 |
| RGB blur S3 | 74.22 | 83.54 | 86.09 | 84.39 | 84.26 | 88.23 |
| RGB blur S4 | 56.27 | 83.54 | 84.17 | **79.99** | **79.39** | 86.20 |
| RGB blur S5 | 34.27 | 83.54 | **83.51** | **75.14** | **73.48** | 84.70 |
| RGB temporal mask S1 | 45.76 | 83.54 | **82.52** | **78.30** | **78.42** | 84.52 |
| RGB temporal mask S2 | 53.27 | 83.54 | 84.22 | **80.26** | **80.62** | 85.89 |
| RGB temporal mask S3 | 35.07 | 83.54 | **82.68** | **76.79** | **76.80** | **83.32** |
| RGB temporal mask S4 | 24.57 | 83.54 | **81.74** | **74.67** | **74.53** | **81.47** |
| RGB temporal mask S5 | 12.35 | 83.54 | **82.24** | **71.85** | **71.39** | **80.66** |
| Joint dropout S1 | 85.75 | 65.29 | 86.63 | **85.14** | 85.84 | 87.67 |
| Joint dropout S2 | 85.75 | 55.15 | 86.00 | **83.69** | **84.82** | 86.71 |
| Joint dropout S3 | 85.75 | 34.71 | **85.25** | **81.24** | **83.20** | **85.55** |
| Joint dropout S4 | 85.75 | 20.50 | **85.07** | **79.49** | **81.92** | **85.13** |
| Joint dropout S5 | 85.75 | 12.02 | **84.69** | **78.25** | **81.03** | **84.79** |
| Body-part dropout S1 | 85.75 | 80.76 | 89.01 | 88.11 | 88.11 | 90.33 |
| Body-part dropout S2 | 85.75 | 60.44 | 86.31 | **84.26** | **85.22** | 87.55 |
| Body-part dropout S3 | 85.75 | 65.40 | 87.74 | 86.07 | 86.72 | 89.28 |
| Body-part dropout S4 | 85.75 | 6.36 | **85.35** | **78.14** | **80.22** | **84.94** |
| Body-part dropout S5 | 85.75 | 3.96 | **85.58** | **78.18** | **80.40** | **85.14** |
| Coordinate noise S1 | 85.75 | 58.49 | **85.61** | **85.39** | 85.76 | 87.81 |
| Coordinate noise S2 | 85.75 | 52.18 | **85.34** | **84.22** | **84.61** | 86.97 |
| Coordinate noise S3 | 85.75 | 41.41 | **85.32** | **82.69** | **83.37** | 85.83 |
| Coordinate noise S4 | 85.75 | 19.78 | **85.31** | **80.84** | **81.97** | **84.08** |
| Coordinate noise S5 | 85.75 | 11.65 | **85.43** | **80.01** | **81.60** | **82.75** |

### Dual corruption (3.10) and coupled occlusion

Both modalities corrupted: independently, or by one occluder that hides the RGB pixels and the joints behind it (coupled).

| Condition | RGB only (R3D-18) | Skeleton only (CTR-GCN) | Average fusion | Concat fusion | Global-weight fusion | Attention fusion |
|---|---|---|---|---|---|---|
| RGB blur S1 + Coordinate noise S4 | 84.13 | 19.78 | 83.60 | 78.85 | 80.22 | 82.13 |
| RGB blur S2 + Coordinate noise S2 | 81.11 | 52.18 | 81.32 | 81.33 | 81.88 | 84.43 |
| RGB blur S3 + Coordinate noise S3 | 74.22 | 41.41 | 74.61 | 73.57 | 74.47 | 77.72 |
| RGB blur S4 + Coordinate noise S1 | 56.27 | 58.49 | 68.89 | 73.49 | 73.58 | 78.58 |
| RGB blur S4 + Coordinate noise S4 | 56.27 | 19.78 | 54.78 | 52.73 | 53.45 | 53.79 |
| nan | 83.26 |  | 87.54 | 86.52 | 86.72 | 89.11 |
| RGB occlusion S1 + Coupled occlusion (joints behind the box) S1 | 83.26 |  | 86.71 | 85.60 | 85.95 | 88.40 |
| RGB occlusion S1 + Body-part dropout S4 | 83.26 | 6.36 | 82.70 | 75.08 | 76.88 | 82.29 |
| nan | 79.86 |  | 84.17 | 83.25 | 83.62 | 86.35 |
| RGB occlusion S2 + Coupled occlusion (joints behind the box) S2 | 79.86 |  | 83.03 | 82.01 | 82.46 | 85.30 |
| RGB occlusion S2 + Body-part dropout S2 | 79.86 | 60.44 | 82.10 | 80.28 | 81.21 | 83.94 |
| nan | 75.74 |  | 80.35 | 79.31 | 79.64 | 82.94 |
| RGB occlusion S3 + Coupled occlusion (joints behind the box) S3 | 75.74 |  | 78.59 | 77.74 | 78.58 | 81.68 |
| RGB occlusion S3 + Body-part dropout S3 | 75.74 | 65.40 | 82.82 | 81.20 | 81.70 | 85.78 |
| nan | 71.62 |  | 75.43 | 74.40 | 74.98 | 78.76 |
| RGB occlusion S4 + Coupled occlusion (joints behind the box) S4 | 71.62 |  | 73.29 | 72.39 | 73.62 | 77.47 |
| RGB occlusion S4 + Body-part dropout S1 | 71.62 | 80.76 | 84.77 | 83.18 | 82.90 | 86.79 |
| RGB occlusion S4 + Body-part dropout S4 | 71.62 | 6.36 | 70.30 | 61.07 | 63.00 | 69.44 |
| nan | 66.47 |  | 69.65 | 68.46 | 69.26 | 73.55 |
| RGB occlusion S5 + Coupled occlusion (joints behind the box) S5 | 66.47 |  | 67.36 | 66.67 | 67.97 | 72.34 |
| RGB temporal mask S1 + Joint dropout S4 | 45.76 | 20.50 | 47.77 | 49.12 | 51.95 | 58.00 |
| RGB temporal mask S2 + Joint dropout S2 | 53.27 | 55.15 | 66.69 | 66.54 | 68.26 | 72.97 |
| RGB temporal mask S3 + Joint dropout S3 | 35.07 | 34.71 | 47.00 | 49.73 | 51.15 | 56.30 |
| RGB temporal mask S4 + Joint dropout S1 | 24.57 | 65.29 | 64.23 | 61.86 | 62.04 | 68.37 |
| RGB temporal mask S4 + Joint dropout S4 | 24.57 | 20.50 | 31.64 | 34.77 | 37.07 | 40.67 |

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
| R3D-18: RGB occlusion | 87.17 | 84.73 | 80.91 | 76.32 | 71.47 | 66.24 | 75.93 | 0.871 |
| R3D-18: RGB blur | 87.17 | 85.59 | 82.60 | 75.44 | 57.82 | 35.01 | 67.29 | 0.772 |
| R3D-18: RGB temporal mask | 87.17 | 45.83 | 45.37 | 28.30 | 20.83 | 10.74 | 30.21 | 0.347 |
| CTR-GCN: Joint dropout | 84.93 | 61.34 | 38.96 | 14.59 | 7.45 | 4.44 | 25.36 | 0.299 |
| CTR-GCN: Body-part dropout | 84.93 | 82.00 | 62.37 | 63.21 | 7.43 | 3.41 | 43.68 | 0.514 |
| CTR-GCN: Coordinate noise | 84.93 | 61.05 | 50.81 | 27.97 | 9.49 | 5.20 | 30.90 | 0.364 |

### Clean accuracy, mean corruption accuracy (mCA) and robustness ratio

mCA = mean top-1 over S1-S5 of the three corruptions of one modality (the other modality clean); "all" averages all 30 single-modality conditions. Robustness ratio = mCA_all / clean.

| Method | Clean | mCA (RGB corrupted) | mCA (skeleton corrupted) | mCA (all) | Robustness ratio |
|---|---|---|---|---|---|
| RGB only (R3D-18) | 87.17 | 57.81 | 87.17 | 72.49 | 0.832 |
| Skeleton only (CTR-GCN) | 84.93 | 84.93 | 33.31 | 59.12 | 0.696 |
| Average fusion | 91.41 | 86.78 | 86.56 | 86.67 | 0.948 |
| Concat fusion | 91.88 | 85.77 | 85.84 | 85.81 | 0.934 |
| Global-weight fusion | 91.75 | 85.39 | 87.01 | 86.20 | 0.940 |
| Attention fusion | 92.57 | 87.45 | 83.68 | 85.57 | 0.924 |

### Diagnostic: fusion vs the healthy modality (3.9)

Healthy = the uncorrupted modality alone. **Bold** = fusion below the healthy modality, i.e. Acc(fusion) < Acc(healthy). Conditions below healthy (of 30): Average fusion 16, Concat fusion 16, Global-weight fusion 15, Attention fusion 14.

| Condition | RGB only (R3D-18) | Skeleton only (CTR-GCN) | Average fusion | Concat fusion | Global-weight fusion | Attention fusion |
|---|---|---|---|---|---|---|
| RGB occlusion S1 | 84.73 | 84.93 | 90.53 | 90.96 | 90.89 | 91.90 |
| RGB occlusion S2 | 80.91 | 84.93 | 89.60 | 89.83 | 89.66 | 91.15 |
| RGB occlusion S3 | 76.32 | 84.93 | 88.56 | 88.69 | 88.51 | 90.32 |
| RGB occlusion S4 | 71.47 | 84.93 | 87.67 | 87.43 | 87.07 | 89.17 |
| RGB occlusion S5 | 66.24 | 84.93 | 86.93 | 86.28 | 85.84 | 88.47 |
| RGB blur S1 | 85.59 | 84.93 | 90.72 | 91.27 | 91.05 | 92.06 |
| RGB blur S2 | 82.60 | 84.93 | 89.78 | 90.26 | 90.05 | 91.15 |
| RGB blur S3 | 75.44 | 84.93 | 87.93 | 88.11 | 87.80 | 89.52 |
| RGB blur S4 | 57.82 | 84.93 | 85.47 | **84.04** | **83.61** | 86.63 |
| RGB blur S5 | 35.01 | 84.93 | **84.24** | **79.26** | **78.74** | **84.39** |
| RGB temporal mask S1 | 45.83 | 84.93 | **84.55** | **84.28** | **83.81** | 85.79 |
| RGB temporal mask S2 | 45.37 | 84.93 | **84.26** | **84.71** | **84.38** | 86.03 |
| RGB temporal mask S3 | 28.30 | 84.93 | **83.77** | **81.90** | **81.20** | **82.69** |
| RGB temporal mask S4 | 20.83 | 84.93 | **83.68** | **81.09** | **80.20** | **81.69** |
| RGB temporal mask S5 | 10.74 | 84.93 | **84.01** | **78.49** | **78.03** | **80.85** |
| Joint dropout S1 | 87.17 | 61.34 | 87.70 | 88.89 | 88.92 | 88.41 |
| Joint dropout S2 | 87.17 | 38.96 | **85.59** | **86.93** | 87.34 | **84.53** |
| Joint dropout S3 | 87.17 | 14.59 | **83.38** | **83.99** | **85.39** | **75.87** |
| Joint dropout S4 | 87.17 | 7.45 | **83.30** | **82.08** | **84.47** | **72.32** |
| Joint dropout S5 | 87.17 | 4.44 | **83.60** | **81.16** | **84.02** | **72.96** |
| Body-part dropout S1 | 87.17 | 82.00 | 91.00 | 91.49 | 91.37 | 92.18 |
| Body-part dropout S2 | 87.17 | 62.37 | 87.63 | 88.43 | 88.99 | 88.99 |
| Body-part dropout S3 | 87.17 | 63.21 | 89.23 | 90.16 | 90.19 | 90.65 |
| Body-part dropout S4 | 87.17 | 7.43 | **85.81** | **82.27** | **85.10** | **78.33** |
| Body-part dropout S5 | 87.17 | 3.41 | **86.49** | **82.34** | **84.77** | **79.97** |
| Coordinate noise S1 | 87.17 | 61.05 | 87.61 | 88.74 | 89.05 | 88.37 |
| Coordinate noise S2 | 87.17 | 50.81 | **86.99** | 87.45 | 87.96 | **87.11** |
| Coordinate noise S3 | 87.17 | 27.97 | **86.67** | **85.58** | **86.55** | **86.00** |
| Coordinate noise S4 | 87.17 | 9.49 | **86.61** | **84.18** | **85.57** | **84.84** |
| Coordinate noise S5 | 87.17 | 5.20 | **86.76** | **83.94** | **85.40** | **84.70** |

### Dual corruption (3.10) and coupled occlusion

Both modalities corrupted: independently, or by one occluder that hides the RGB pixels and the joints behind it (coupled).

| Condition | RGB only (R3D-18) | Skeleton only (CTR-GCN) | Average fusion | Concat fusion | Global-weight fusion | Attention fusion |
|---|---|---|---|---|---|---|
| RGB blur S1 + Coordinate noise S4 | 85.59 | 9.49 | 84.97 | 82.74 | 83.90 | 83.53 |
| RGB blur S2 + Coordinate noise S2 | 82.60 | 50.81 | 83.17 | 84.64 | 85.12 | 85.04 |
| RGB blur S3 + Coordinate noise S3 | 75.44 | 27.97 | 75.05 | 76.25 | 76.77 | 77.75 |
| RGB blur S4 + Coordinate noise S1 | 57.82 | 61.05 | 72.09 | 77.15 | 76.92 | 78.91 |
| RGB blur S4 + Coordinate noise S4 | 57.82 | 9.49 | 54.93 | 56.08 | 56.92 | 55.31 |
| nan | 84.73 |  | 89.55 | 90.22 | 90.09 | 90.99 |
| RGB occlusion S1 + Coupled occlusion (joints behind the box) S1 | 84.73 |  | 88.75 | 89.38 | 89.34 | 90.15 |
| RGB occlusion S1 + Body-part dropout S4 | 84.73 | 7.43 | 83.15 | 78.93 | 82.27 | 75.48 |
| nan | 80.91 |  | 86.14 | 87.06 | 87.01 | 87.91 |
| RGB occlusion S2 + Coupled occlusion (joints behind the box) S2 | 80.91 |  | 85.02 | 85.72 | 85.74 | 86.75 |
| RGB occlusion S2 + Body-part dropout S2 | 80.91 | 62.37 | 83.42 | 84.17 | 84.88 | 85.19 |
| nan | 76.32 |  | 81.70 | 82.95 | 83.00 | 83.97 |
| RGB occlusion S3 + Coupled occlusion (joints behind the box) S3 | 76.32 |  | 79.92 | 81.21 | 81.45 | 82.30 |
| RGB occlusion S3 + Body-part dropout S3 | 76.32 | 63.21 | 83.94 | 85.06 | 85.01 | 86.77 |
| nan | 71.47 |  | 75.92 | 77.74 | 77.92 | 78.73 |
| RGB occlusion S4 + Coupled occlusion (joints behind the box) S4 | 71.47 |  | 73.85 | 74.99 | 75.79 | 76.47 |
| RGB occlusion S4 + Body-part dropout S1 | 71.47 | 82.00 | 86.44 | 86.46 | 86.13 | 88.22 |
| RGB occlusion S4 + Body-part dropout S4 | 71.47 | 7.43 | 68.94 | 63.91 | 68.41 | 61.15 |
| nan | 66.24 |  | 69.56 | 71.71 | 72.05 | 72.93 |
| RGB occlusion S5 + Coupled occlusion (joints behind the box) S5 | 66.24 |  | 67.01 | 68.00 | 69.07 | 69.85 |
| RGB temporal mask S1 + Joint dropout S4 | 45.83 | 7.45 | 37.56 | 40.29 | 43.48 | 41.12 |
| RGB temporal mask S2 + Joint dropout S2 | 45.37 | 38.96 | 53.64 | 63.31 | 60.70 | 64.52 |
| RGB temporal mask S3 + Joint dropout S3 | 28.30 | 14.59 | 27.44 | 36.43 | 35.32 | 37.25 |
| RGB temporal mask S4 + Joint dropout S1 | 20.83 | 61.34 | 61.57 | 62.81 | 57.99 | 64.39 |
| RGB temporal mask S4 + Joint dropout S4 | 20.83 | 7.45 | 17.88 | 23.06 | 24.04 | 23.74 |

### Class-level modality analysis (3.12)

Per-class clean accuracy; dominant = gap of at least 5 percentage points (about 2 standard errors of a per-class difference at ~400-500 test samples per class). Counts: balanced 83, RGB-dominant 26, skeleton-dominant 11. Full list: `class_modality_xset.csv`.

| Action | Name | Test samples | RGB acc | Skeleton acc | RGB - skeleton (pts) | Class type |
|---|---|---|---|---|---|---|
| 11 | reading | 500 | 79.20 | 56.20 | +23.0 | RGB-dominant |
| 30 | typing on a keyboard | 502 | 88.05 | 66.93 | +21.1 | RGB-dominant |
| 29 | playing with phone/tablet | 502 | 78.88 | 59.76 | +19.1 | RGB-dominant |
| 106 | hit other person with something | 491 | 81.26 | 63.14 | +18.1 | RGB-dominant |
| 5 | drop | 502 | 97.81 | 83.67 | +14.1 | RGB-dominant |
| 94 | throw up cap/hat | 487 | 95.28 | 81.72 | +13.6 | RGB-dominant |
| 75 | cutting nails | 481 | 81.29 | 67.78 | +13.5 | RGB-dominant |
| 7 | throw | 502 | 96.81 | 83.67 | +13.1 | RGB-dominant |
| 34 | rub two hands together | 503 | 83.10 | 69.98 | +13.1 | RGB-dominant |
| 51 | kicking other person | 502 | 98.80 | 86.06 | +12.7 | RGB-dominant |
| 45 | touch chest (stomachache/heart pain) | 504 | 74.80 | 79.96 | -5.2 | skeleton-dominant |
| 69 | thumb up | 490 | 56.94 | 63.88 | -6.9 | skeleton-dominant |
| 77 | snapping fingers | 489 | 62.58 | 70.96 | -8.4 | skeleton-dominant |
| 78 | open bottle | 488 | 68.44 | 76.84 | -8.4 | skeleton-dominant |
| 72 | make victory sign | 490 | 43.67 | 52.65 | -9.0 | skeleton-dominant |
| 83 | ball up paper | 491 | 66.60 | 75.97 | -9.4 | skeleton-dominant |
| 84 | play magic cube | 489 | 60.33 | 71.37 | -11.0 | skeleton-dominant |
| 81 | toss a coin | 489 | 77.71 | 88.96 | -11.2 | skeleton-dominant |
| 82 | fold paper | 491 | 61.30 | 72.71 | -11.4 | skeleton-dominant |
| 10 | clapping | 500 | 74.00 | 87.60 | -13.6 | skeleton-dominant |

## Classes with the same dominance on both protocols

typing on a keyboard (RGB-dominant), cutting nails (RGB-dominant), reading (RGB-dominant), throw up cap/hat (RGB-dominant), take off a shoe (RGB-dominant), hit other person with something (RGB-dominant), drop (RGB-dominant), tear up paper (RGB-dominant), flick hair (RGB-dominant), exchange things with other person (RGB-dominant), kicking other person (RGB-dominant), play magic cube (skeleton-dominant), fold paper (skeleton-dominant), open bottle (skeleton-dominant), snapping fingers (skeleton-dominant), toss a coin (skeleton-dominant)
