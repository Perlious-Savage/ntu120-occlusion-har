# Natural occlusion (Kinect inferred joints), clean test split

Inferred share = fraction of the actors' joint observations that Kinect could not see. Top-1 accuracy (%).

## XSUB

| Split | Inferred share | Samples | RGB only (R3D-18) | Skeleton only (CTR-GCN) | Average fusion | Concat fusion | Global-weight fusion | Attention fusion |
|---|---|---|---|---|---|---|---|---|
| quintile 1 | 0.00-0.05 | 10189 | 82.0 | 84.95 | 87.6 | 87.09 | 86.66 | 89.15 |
| quintile 2 | 0.05-0.11 | 10198 | 86.78 | 86.94 | 90.9 | 89.79 | 89.86 | 92.22 |
| quintile 3 | 0.11-0.16 | 10164 | 87.6 | 85.95 | 91.17 | 90.05 | 89.93 | 92.13 |
| quintile 4 | 0.16-0.23 | 10184 | 86.68 | 82.86 | 89.59 | 88.96 | 88.72 | 90.43 |
| quintile 5 | 0.23-0.74 | 10184 | 85.71 | 76.98 | 88.18 | 86.38 | 86.9 | 89.48 |
| within-class low | 0.00-0.27 | 25503 | 87.72 | 88.37 | 91.77 | 90.41 | 90.08 | 92.8 |
| within-class high | 0.06-0.74 | 25416 | 85.97 | 81.13 | 89.2 | 87.97 | 88.13 | 90.25 |

## XSET

| Split | Inferred share | Samples | RGB only (R3D-18) | Skeleton only (CTR-GCN) | Average fusion | Concat fusion | Global-weight fusion | Attention fusion |
|---|---|---|---|---|---|---|---|---|
| quintile 1 | 0.00-0.06 | 11897 | 85.32 | 89.06 | 91.33 | 91.64 | 91.4 | 92.59 |
| quintile 2 | 0.06-0.11 | 11901 | 87.38 | 88.61 | 92.4 | 93.05 | 92.64 | 93.65 |
| quintile 3 | 0.11-0.17 | 11893 | 88.59 | 86.84 | 92.63 | 92.8 | 92.62 | 93.55 |
| quintile 4 | 0.17-0.23 | 11891 | 87.93 | 83.78 | 91.16 | 91.66 | 91.7 | 92.4 |
| quintile 5 | 0.23-0.74 | 11895 | 86.62 | 76.33 | 89.55 | 90.23 | 90.37 | 90.65 |
| within-class low | 0.00-0.25 | 29771 | 87.96 | 89.37 | 92.78 | 93.2 | 92.91 | 93.84 |
| within-class high | 0.06-0.74 | 29706 | 86.29 | 80.43 | 89.98 | 90.48 | 90.5 | 91.23 |
