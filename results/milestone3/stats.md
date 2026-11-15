# Fusion vs the healthy modality, with uncertainty (review item A4)

Fusion heads from `outputs/m3_fusion` (one training setting for every head). Gap = Acc(fusion) - Acc(healthy), points; 95% paired bootstrap CI; exact McNemar, Holm over the 120 cells of a protocol. "Healthy" is an oracle selector. Single seed (head seeds pending).

## XSUB

| Method | below (raw) | below, Holm p<0.05 | below by > 0.5 pt (CI), Holm p<0.05 | mean gap | worst gap (95% CI) |
|---|---|---|---|---|---|
| Average fusion | 15 | 13 | 6 | +0.89 | -1.80 (-2.00, -1.61) RGB temporal mask S4 |
| Concat fusion | 21 | 20 | 18 | -2.73 | -11.69 (-12.05, -11.32) RGB temporal mask S5 |
| Global-weight fusion | 20 | 19 | 16 | -2.56 | -15.96 (-16.37, -15.56) RGB temporal mask S5 |
| Attention fusion | 10 | 8 | 6 | +1.75 | -3.00 (-3.23, -2.78) Coordinate noise S5 |

Headroom (mean over the 30 conditions, %):

| Average fusion | Healthy (oracle selector) | Max-confidence selector | Either branch correct (oracle) |
|---|---|---|---|
| 85.54 | 84.65 | 85.17 | 88.94 |

- Average fusion: closes +0.0% of the average-to-either-correct gap on average.
- Concat fusion: closes -223.1% of the average-to-either-correct gap on average.
- Global-weight fusion: closes -164.7% of the average-to-either-correct gap on average.
- Attention fusion: closes +2.5% of the average-to-either-correct gap on average.

## XSET

| Method | below (raw) | below, Holm p<0.05 | below by > 0.5 pt (CI), Holm p<0.05 | mean gap | worst gap (95% CI) |
|---|---|---|---|---|---|
| Average fusion | 16 | 15 | 10 | +0.62 | -3.87 (-4.04, -3.70) Joint dropout S4 |
| Concat fusion | 19 | 17 | 17 | -1.78 | -10.08 (-10.39, -9.78) Joint dropout S5 |
| Global-weight fusion | 18 | 18 | 17 | -1.67 | -12.40 (-12.74, -12.05) RGB temporal mask S5 |
| Attention fusion | 14 | 13 | 12 | -0.48 | -14.85 (-15.16, -14.54) Joint dropout S4 |

Headroom (mean over the 30 conditions, %):

| Average fusion | Healthy (oracle selector) | Max-confidence selector | Either branch correct (oracle) |
|---|---|---|---|
| 86.67 | 86.05 | 86.33 | 90.20 |

- Average fusion: closes +0.0% of the average-to-either-correct gap on average.
- Concat fusion: closes -117.7% of the average-to-either-correct gap on average.
- Global-weight fusion: closes -96.1% of the average-to-either-correct gap on average.
- Attention fusion: closes -62.1% of the average-to-either-correct gap on average.
