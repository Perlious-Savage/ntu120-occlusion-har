# Milestone 3 results

Top-1 accuracy (%) on the official test splits. Frozen Milestone 2 backbones (seed 42); fusion heads trained on clean features of the held-out validation split (5-fold CV picks the stopping epoch) and tested under corruption without retraining. Corruption definitions: `docs/corruption_protocol.md`.

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
| Concat fusion | 88.22 | 81.37 | 82.51 | 81.94 | 0.929 |
| Global-weight fusion | 87.99 | 78.92 | 83.78 | 81.35 | 0.925 |
| Attention fusion | 89.93 | 83.61 | 86.09 | 84.85 | 0.944 |

### Diagnostic: fusion vs the healthy modality (3.9)

Healthy = the uncorrupted modality alone. **Bold** = fusion below the healthy modality, i.e. Acc(fusion) < Acc(healthy). Conditions below healthy (of 30): Average fusion 15, Concat fusion 21, Global-weight fusion 24, Attention fusion 13.

| Condition | RGB only (R3D-18) | Skeleton only (CTR-GCN) | Average fusion | Concat fusion | Global-weight fusion | Attention fusion |
|---|---|---|---|---|---|---|
| RGB occlusion S1 | 83.26 | 83.54 | 88.60 | 87.39 | 86.97 | 89.01 |
| RGB occlusion S2 | 79.86 | 83.54 | 87.54 | 86.18 | 85.55 | 87.54 |
| RGB occlusion S3 | 75.74 | 83.54 | 86.68 | 84.84 | **83.52** | 86.26 |
| RGB occlusion S4 | 71.62 | 83.54 | 85.93 | 83.82 | **81.90** | 85.12 |
| RGB occlusion S5 | 66.47 | 83.54 | 85.19 | **82.57** | **80.25** | 83.55 |
| RGB blur S1 | 84.13 | 83.54 | 88.70 | 87.53 | 87.10 | 89.32 |
| RGB blur S2 | 81.11 | 83.54 | 87.75 | 86.34 | 85.79 | 88.25 |
| RGB blur S3 | 74.22 | 83.54 | 86.09 | 84.01 | **83.36** | 86.52 |
| RGB blur S4 | 56.27 | 83.54 | 84.17 | **79.78** | **78.26** | **83.21** |
| RGB blur S5 | 34.27 | 83.54 | **83.51** | **74.99** | **72.33** | **79.48** |
| RGB temporal mask S1 | 45.76 | 83.54 | **82.52** | **78.29** | **74.60** | **81.18** |
| RGB temporal mask S2 | 53.27 | 83.54 | 84.22 | **80.10** | **77.51** | 84.05 |
| RGB temporal mask S3 | 35.07 | 83.54 | **82.68** | **77.33** | **72.31** | **80.84** |
| RGB temporal mask S4 | 24.57 | 83.54 | **81.74** | **75.24** | **69.39** | **76.55** |
| RGB temporal mask S5 | 12.35 | 83.54 | **82.24** | **72.23** | **64.95** | **73.33** |
| Joint dropout S1 | 85.75 | 65.29 | 86.63 | **85.11** | **85.59** | 87.31 |
| Joint dropout S2 | 85.75 | 55.15 | 86.00 | **83.82** | **84.62** | 86.46 |
| Joint dropout S3 | 85.75 | 34.71 | **85.25** | **81.52** | **83.12** | **85.65** |
| Joint dropout S4 | 85.75 | 20.50 | **85.07** | **79.77** | **82.11** | **85.26** |
| Joint dropout S5 | 85.75 | 12.02 | **84.69** | **78.59** | **81.30** | **85.22** |
| Body-part dropout S1 | 85.75 | 80.76 | 89.01 | 87.90 | 87.83 | 89.50 |
| Body-part dropout S2 | 85.75 | 60.44 | 86.31 | **84.36** | **85.14** | 86.29 |
| Body-part dropout S3 | 85.75 | 65.40 | 87.74 | 86.03 | 86.48 | 87.90 |
| Body-part dropout S4 | 85.75 | 6.36 | **85.35** | **78.56** | **81.32** | **85.55** |
| Body-part dropout S5 | 85.75 | 3.96 | **85.58** | **78.63** | **81.23** | **85.70** |
| Coordinate noise S1 | 85.75 | 58.49 | **85.61** | **85.35** | **85.49** | 87.31 |
| Coordinate noise S2 | 85.75 | 52.18 | **85.34** | **84.32** | **84.77** | 86.66 |
| Coordinate noise S3 | 85.75 | 41.41 | **85.32** | **82.69** | **83.51** | 85.86 |
| Coordinate noise S4 | 85.75 | 19.78 | **85.31** | **80.90** | **82.27** | **84.13** |
| Coordinate noise S5 | 85.75 | 11.65 | **85.43** | **80.08** | **81.95** | **82.49** |

### Dual corruption (3.10)

Both modalities corrupted independently.

| Condition | RGB only (R3D-18) | Skeleton only (CTR-GCN) | Average fusion | Concat fusion | Global-weight fusion | Attention fusion |
|---|---|---|---|---|---|---|
| RGB blur S1 + Coordinate noise S4 | 84.13 | 19.78 | 83.60 | 78.76 | 80.31 | 82.18 |
| RGB blur S2 + Coordinate noise S2 | 81.11 | 52.18 | 81.32 | 81.52 | 81.52 | 83.64 |
| RGB blur S3 + Coordinate noise S3 | 74.22 | 41.41 | 74.61 | 73.24 | 74.00 | 77.30 |
| RGB blur S4 + Coordinate noise S1 | 56.27 | 58.49 | 68.89 | 73.45 | 72.56 | 74.82 |
| RGB blur S4 + Coordinate noise S4 | 56.27 | 19.78 | 54.78 | 52.80 | 54.09 | 55.61 |
| RGB occlusion S1 + Body-part dropout S4 | 83.26 | 6.36 | 82.70 | 75.53 | 78.27 | 83.01 |
| RGB occlusion S2 + Body-part dropout S2 | 79.86 | 60.44 | 82.10 | 80.23 | 80.82 | 82.43 |
| RGB occlusion S3 + Body-part dropout S3 | 75.74 | 65.40 | 82.82 | 81.17 | 80.79 | 83.02 |
| RGB occlusion S4 + Body-part dropout S1 | 71.62 | 80.76 | 84.77 | 82.95 | 81.32 | 84.15 |
| RGB occlusion S4 + Body-part dropout S4 | 71.62 | 6.36 | 70.30 | 61.69 | 64.31 | 71.23 |
| RGB temporal mask S1 + Joint dropout S4 | 45.76 | 20.50 | 47.77 | 48.76 | 48.55 | 54.10 |
| RGB temporal mask S2 + Joint dropout S2 | 53.27 | 55.15 | 66.69 | 66.19 | 64.76 | 71.51 |
| RGB temporal mask S3 + Joint dropout S3 | 35.07 | 34.71 | 47.00 | 49.71 | 47.22 | 52.97 |
| RGB temporal mask S4 + Joint dropout S1 | 24.57 | 65.29 | 64.23 | 62.04 | 56.32 | 64.23 |
| RGB temporal mask S4 + Joint dropout S4 | 24.57 | 20.50 | 31.64 | 34.74 | 33.90 | 35.18 |

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
| Concat fusion | 91.92 | 85.45 | 85.49 | 85.47 | 0.930 |
| Global-weight fusion | 91.44 | 82.72 | 86.30 | 84.51 | 0.924 |
| Attention fusion | 92.86 | 87.76 | 87.78 | 87.77 | 0.945 |

### Diagnostic: fusion vs the healthy modality (3.9)

Healthy = the uncorrupted modality alone. **Bold** = fusion below the healthy modality, i.e. Acc(fusion) < Acc(healthy). Conditions below healthy (of 30): Average fusion 14, Concat fusion 17, Global-weight fusion 19, Attention fusion 10.

| Condition | RGB only (R3D-18) | Skeleton only (CTR-GCN) | Average fusion | Concat fusion | Global-weight fusion | Attention fusion |
|---|---|---|---|---|---|---|
| RGB occlusion S1 | 84.78 | 86.23 | 91.17 | 91.13 | 90.37 | 92.23 |
| RGB occlusion S2 | 81.10 | 86.23 | 90.31 | 90.11 | 88.85 | 91.40 |
| RGB occlusion S3 | 76.30 | 86.23 | 89.48 | 88.99 | 87.32 | 90.72 |
| RGB occlusion S4 | 71.50 | 86.23 | 88.79 | 87.82 | **85.69** | 89.83 |
| RGB occlusion S5 | 65.92 | 86.23 | 87.96 | 86.59 | **83.95** | 89.07 |
| RGB blur S1 | 85.71 | 86.23 | 91.34 | 91.33 | 90.63 | 92.43 |
| RGB blur S2 | 82.60 | 86.23 | 90.53 | 90.25 | 89.17 | 91.75 |
| RGB blur S3 | 74.75 | 86.23 | 89.06 | 87.93 | **86.15** | 90.31 |
| RGB blur S4 | 54.25 | 86.23 | 86.96 | **83.30** | **80.09** | 87.84 |
| RGB blur S5 | 32.36 | 86.23 | **86.21** | **78.77** | **73.61** | **86.10** |
| RGB temporal mask S1 | 51.26 | 86.23 | 86.83 | **84.69** | **82.05** | 87.26 |
| RGB temporal mask S2 | 48.48 | 86.23 | **85.15** | **84.80** | **81.71** | 86.48 |
| RGB temporal mask S3 | 25.97 | 86.23 | **81.04** | **80.99** | **76.22** | **81.14** |
| RGB temporal mask S4 | 21.03 | 86.23 | **82.55** | **79.41** | **74.49** | **80.57** |
| RGB temporal mask S5 | 10.87 | 86.23 | **83.36** | **75.69** | **70.45** | **79.24** |
| Joint dropout S1 | 87.52 | 68.91 | 88.71 | 88.87 | 88.84 | 90.11 |
| Joint dropout S2 | 87.52 | 57.79 | 87.76 | **87.21** | 87.58 | 88.96 |
| Joint dropout S3 | 87.52 | 32.04 | **87.14** | **84.32** | **85.34** | 87.69 |
| Joint dropout S4 | 87.52 | 13.70 | **86.88** | **82.45** | **83.96** | **86.80** |
| Joint dropout S5 | 87.52 | 5.67 | **86.70** | **81.29** | **83.50** | **86.49** |
| Body-part dropout S1 | 87.52 | 83.79 | 91.43 | 91.55 | 91.06 | 92.57 |
| Body-part dropout S2 | 87.52 | 62.08 | 87.72 | 87.87 | 88.31 | 89.53 |
| Body-part dropout S3 | 87.52 | 65.66 | 89.54 | 90.02 | 89.73 | 91.20 |
| Body-part dropout S4 | 87.52 | 5.42 | **86.51** | **80.52** | **83.10** | **81.45** |
| Body-part dropout S5 | 87.52 | 2.77 | **87.02** | **81.64** | **83.26** | **83.02** |
| Coordinate noise S1 | 87.52 | 57.68 | 87.64 | 88.21 | 88.05 | 89.19 |
| Coordinate noise S2 | 87.52 | 49.66 | **87.33** | **87.12** | **87.18** | 88.75 |
| Coordinate noise S3 | 87.52 | 29.43 | **86.94** | **85.41** | **85.82** | 87.73 |
| Coordinate noise S4 | 87.52 | 10.24 | **86.36** | **83.34** | **84.54** | **86.86** |
| Coordinate noise S5 | 87.52 | 5.10 | **86.15** | **82.58** | **84.16** | **86.43** |

### Dual corruption (3.10)

Both modalities corrupted independently.

| Condition | RGB only (R3D-18) | Skeleton only (CTR-GCN) | Average fusion | Concat fusion | Global-weight fusion | Attention fusion |
|---|---|---|---|---|---|---|
| RGB blur S1 + Coordinate noise S4 | 85.71 | 10.24 | 84.53 | 81.86 | 82.97 | 85.60 |
| RGB blur S2 + Coordinate noise S2 | 82.60 | 49.66 | 83.44 | 84.64 | 84.14 | 86.42 |
| RGB blur S3 + Coordinate noise S3 | 74.75 | 29.43 | 74.90 | 75.24 | 75.17 | 79.08 |
| RGB blur S4 + Coordinate noise S1 | 54.25 | 57.68 | 69.43 | 74.49 | 71.51 | 77.07 |
| RGB blur S4 + Coordinate noise S4 | 54.25 | 10.24 | 47.90 | 50.13 | 51.50 | 54.69 |
| RGB occlusion S1 + Body-part dropout S4 | 84.78 | 5.42 | 83.58 | 77.00 | 79.71 | 77.99 |
| RGB occlusion S2 + Body-part dropout S2 | 81.10 | 62.08 | 83.60 | 83.81 | 83.79 | 85.83 |
| RGB occlusion S3 + Body-part dropout S3 | 76.30 | 65.66 | 84.23 | 84.98 | 83.72 | 87.26 |
| RGB occlusion S4 + Body-part dropout S1 | 71.50 | 83.79 | 87.67 | 86.96 | 84.87 | 88.88 |
| RGB occlusion S4 + Body-part dropout S4 | 71.50 | 5.42 | 69.21 | 59.96 | 63.69 | 61.25 |
| RGB temporal mask S1 + Joint dropout S4 | 51.26 | 13.70 | 51.17 | 51.53 | 52.02 | 56.56 |
| RGB temporal mask S2 + Joint dropout S2 | 48.48 | 57.79 | 64.74 | 69.77 | 67.01 | 71.52 |
| RGB temporal mask S3 + Joint dropout S3 | 25.97 | 32.04 | 36.60 | 46.99 | 43.06 | 45.84 |
| RGB temporal mask S4 + Joint dropout S1 | 21.03 | 68.91 | 64.53 | 63.90 | 59.11 | 65.37 |
| RGB temporal mask S4 + Joint dropout S4 | 21.03 | 13.70 | 24.21 | 29.39 | 27.47 | 30.67 |

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
