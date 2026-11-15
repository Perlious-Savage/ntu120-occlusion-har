# Ablation (4.15)

| Protocol | Setting | Variant | Method | Clean | mCA | Mixed | Mixed ECE | Mixed AURC |
|---|---|---|---|---|---|---|---|---|
| xset | A | A (no calibration, fixed) | average | 91.41 | 86.67 | 84.04 | 0.1267 | 0.0331 |
| xset | A | B (+ calibration) | calibrated_average | 92.55 | 84.96 | 83.54 | 0.0898 | 0.0337 |
| xset | A | C (+ dynamic entropy weights) | entropy | 92.47 | 86.65 | 84.36 | 0.0357 | 0.0289 |
| xset | A | D (+ learned reliability) | reliability_correct | 92.39 | 86.77 | 84.95 | 0.0654 | 0.0286 |
| xset | B | A (no calibration, fixed) | average | 91.41 | 86.67 | 84.04 | 0.1267 | 0.0331 |
| xset | B | B (+ calibration) | calibrated_average | 92.55 | 84.96 | 83.54 | 0.0898 | 0.0337 |
| xset | B | C (+ dynamic entropy weights) | entropy | 92.47 | 86.65 | 84.36 | 0.0357 | 0.0289 |
| xset | B | D (+ learned reliability) | reliability_correct | 92.47 | 88.25 | 85.83 | 0.0356 | 0.0250 |
| xsub | A | A (no calibration, fixed) | average | 89.49 | 85.54 | 83.44 | 0.1160 | 0.0358 |
| xsub | A | B (+ calibration) | calibrated_average | 90.70 | 86.15 | 84.52 | 0.0980 | 0.0325 |
| xsub | A | C (+ dynamic entropy weights) | entropy | 90.66 | 86.76 | 84.79 | 0.0370 | 0.0280 |
| xsub | A | D (+ learned reliability) | reliability_correct | 90.41 | 85.83 | 84.20 | 0.0644 | 0.0311 |
| xsub | B | A (no calibration, fixed) | average | 89.49 | 85.54 | 83.44 | 0.1160 | 0.0358 |
| xsub | B | B (+ calibration) | calibrated_average | 90.70 | 86.15 | 84.52 | 0.0980 | 0.0325 |
| xsub | B | C (+ dynamic entropy weights) | entropy | 90.66 | 86.76 | 84.79 | 0.0370 | 0.0280 |
| xsub | B | D (+ learned reliability) | reliability_correct | 90.55 | 86.90 | 84.88 | 0.0353 | 0.0280 |

Full (temporal reliability + consistency loss): pending (4.8-4.10).