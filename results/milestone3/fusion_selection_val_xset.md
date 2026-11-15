# Fusion setting chosen on corrupted validation data (review item A1)

Validation-trained settings scored out of fold. mCA = mean top-1 over the 30 single-modality corrupted conditions. **Bold** = chosen on validation; its test numbers are the ones to report.

| Protocol | Method | Setting | Val clean | Val mCA | Test clean | Test mCA |
|---|---|---|---|---|---|---|
| xset | average | **val_aux** | 91.68 | 87.73 | 91.41 | 86.67 |
| xset | concat | val_aux | 91.53 | 84.86 | 91.29 | 84.27 |
| xset | global | val_aux | 91.64 | 85.22 | 91.08 | 84.38 |
| xset | attention | **val_aux** | 92.61 | 85.53 | 92.57 | 85.57 |
| xset | concat | val_noaux | 91.64 | 85.34 | 91.35 | 84.34 |
| xset | global | val_noaux | 91.68 | 85.81 | 91.22 | 85.18 |
| xset | attention | **val_noaux** | 92.61 | 85.53 | 92.57 | 85.57 |
| xset | concat | **trainfeat** | 92.39 | 87.08 | 91.88 | 85.81 |
| xset | global | **trainfeat** | 92.43 | 87.43 | 91.75 | 86.20 |
| xset | attention | trainfeat | 91.44 | 82.91 | 90.96 | 81.69 |