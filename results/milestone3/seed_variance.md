# Fusion-head seed variance (review item A4)

Head seeds 42, 1, 2, 3, 4 on the same frozen branch features. mCA = mean top-1 (%) over the 30 single-modality conditions. "Below healthy" = conditions where fusion is more than 0.5 points below the uncorrupted modality alone (oracle reference).

| Protocol | Setting | Method | Seeds | Clean (mean ± SD) | mCA (mean ± SD) | Below healthy (mean, min-max) |
|---|---|---|---|---|---|---|
| xset | trainfeat | attention | 5 | 91.15 ± 0.21 | 82.39 ± 0.79 | 9.6 (9-12) |
| xset | trainfeat | concat | 5 | 92.00 ± 0.07 | 85.98 ± 0.15 | 13.6 (13-14) |
| xset | trainfeat | global | 5 | 91.85 ± 0.08 | 86.28 ± 0.16 | 13.6 (12-15) |
| xset | val_aux | attention | 5 | 92.19 ± 0.34 | 83.66 ± 1.97 | 15.0 (13-17) |
| xset | val_aux | concat | 5 | 91.25 ± 0.03 | 83.92 ± 0.21 | 17.4 (17-18) |
| xset | val_aux | global | 5 | 90.96 ± 0.20 | 84.22 ± 0.21 | 17.6 (17-18) |
| xset | val_noaux | attention | 1 | 92.57 ± nan | 85.57 ± nan | 13.0 (13-13) |
| xset | val_noaux | concat | 1 | 91.35 ± nan | 84.34 ± nan | 17.0 (17-17) |
| xset | val_noaux | global | 1 | 91.22 ± nan | 85.18 ± nan | 17.0 (17-17) |
| xsub | trainfeat | attention | 5 | 89.49 ± 0.21 | 82.34 ± 0.77 | 8.2 (7-9) |
| xsub | trainfeat | concat | 5 | 89.95 ± 0.04 | 84.30 ± 0.16 | 14.2 (13-15) |
| xsub | trainfeat | global | 5 | 89.84 ± 0.07 | 84.43 ± 0.20 | 12.6 (12-13) |
| xsub | val_aux | attention | 5 | 90.62 ± 0.06 | 85.86 ± 0.43 | 8.8 (8-10) |
| xsub | val_aux | concat | 5 | 88.43 ± 0.22 | 82.03 ± 0.25 | 19.4 (18-20) |
| xsub | val_aux | global | 5 | 88.34 ± 0.16 | 81.86 ± 0.18 | 19.2 (18-20) |
| xsub | val_noaux | attention | 1 | 90.68 ± nan | 86.40 ± nan | 8.0 (8-8) |
| xsub | val_noaux | concat | 1 | 88.70 ± nan | 81.96 ± nan | 19.0 (19-19) |
| xsub | val_noaux | global | 1 | 88.41 ± nan | 82.41 ± nan | 19.0 (19-19) |