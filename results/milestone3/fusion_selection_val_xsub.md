# Fusion setting chosen on corrupted validation data (review item A1)

Validation-trained settings scored out of fold. mCA = mean top-1 over the 30 single-modality corrupted conditions. **Bold** = chosen on validation; its test numbers are the ones to report.

| Protocol | Method | Setting | Val clean | Val mCA | Test clean | Test mCA |
|---|---|---|---|---|---|---|
| xsub | average | **val_aux** | 90.58 | 86.61 | 89.49 | 85.54 |
| xsub | concat | **val_aux** | 92.89 | 86.72 | 88.46 | 81.92 |
| xsub | global | val_aux | 92.96 | 86.65 | 88.44 | 82.09 |
| xsub | attention | **val_aux** | 91.81 | 86.70 | 90.68 | 86.40 |
| xsub | concat | val_noaux | 93.00 | 86.51 | 88.70 | 81.96 |
| xsub | global | **val_noaux** | 92.94 | 87.13 | 88.41 | 82.41 |
| xsub | attention | **val_noaux** | 91.81 | 86.70 | 90.68 | 86.40 |
| xsub | concat | trainfeat | 91.43 | 85.28 | 89.88 | 84.20 |
| xsub | global | trainfeat | 91.22 | 85.65 | 89.84 | 84.60 |
| xsub | attention | trainfeat | 91.11 | 84.28 | 89.87 | 83.67 |