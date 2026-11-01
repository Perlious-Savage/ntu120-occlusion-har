# Baseline results (clean NTU RGB+D 120, test split, best-val-accuracy checkpoint)

| Model | Protocol | Top-1 Acc. | Top-5 Acc. | Macro-F1 | Best Epoch |
|---|---|---|---|---|---|
| R3D-18 | X-Sub | 85.75% | 98.02% | 0.8669 | 26 |
| CTR-GCN | X-Sub | 83.54% | 96.79% | 0.8380 | 58 |
| R3D-18 | X-Set | 87.52% | 98.39% | 0.8735 | 26 |
| CTR-GCN | X-Set | 86.23% | 97.09% | 0.8620 | 64 |

Best Epoch counts from 0, as in training_log.csv (65 CTR-GCN epochs = 0-64, 30 R3D-18 epochs = 0-29).

### X-Sub: 20 weakest classes for RGB (R3D-18)

| Action | rgb_accuracy | skeleton_accuracy | rgb_minus_skeleton |
|---|---|---|---|
| A073 staple book | 38.9% | 28.0% | 10.9% |
| A072 make victory sign | 42.6% | 43.1% | -0.5% |
| A071 make ok sign | 47.1% | 43.5% | 3.7% |
| A076 cutting paper (using scissors) | 49.9% | 56.0% | -6.1% |
| A069 thumb up | 54.6% | 59.3% | -4.7% |
| A074 counting money | 56.1% | 51.8% | 4.4% |
| A077 snapping fingers | 59.4% | 68.5% | -9.1% |
| A082 fold paper | 60.0% | 65.6% | -5.6% |
| A084 play magic cube | 60.8% | 66.3% | -5.4% |
| A029 playing with phone/tablet | 62.2% | 62.9% | -0.7% |
| A012 writing | 65.4% | 53.3% | 12.1% |
| A078 open bottle | 65.4% | 73.1% | -7.7% |
| A083 ball up paper | 65.6% | 65.7% | -0.2% |
| A105 blow nose | 66.4% | 60.3% | 6.1% |
| A103 yawn | 67.0% | 63.5% | 3.5% |
| A041 sneeze/cough | 69.2% | 74.6% | -5.4% |
| A086 apply cream on hand back | 69.9% | 71.3% | -1.4% |
| A107 wield knife towards other person | 72.0% | 67.2% | 4.9% |
| A106 hit other person with something | 73.7% | 61.4% | 12.3% |
| A002 eat meal/snack | 74.2% | 71.3% | 2.9% |

### X-Sub: 20 weakest classes for skeleton (CTR-GCN)

| Action | rgb_accuracy | skeleton_accuracy | rgb_minus_skeleton |
|---|---|---|---|
| A073 staple book | 38.9% | 28.0% | 10.9% |
| A072 make victory sign | 42.6% | 43.1% | -0.5% |
| A071 make ok sign | 47.1% | 43.5% | 3.7% |
| A074 counting money | 56.1% | 51.8% | 4.4% |
| A012 writing | 65.4% | 53.3% | 12.1% |
| A076 cutting paper (using scissors) | 49.9% | 56.0% | -6.1% |
| A011 reading | 75.5% | 58.2% | 17.2% |
| A069 thumb up | 54.6% | 59.3% | -4.7% |
| A105 blow nose | 66.4% | 60.3% | 6.1% |
| A106 hit other person with something | 73.7% | 61.4% | 12.3% |
| A075 cutting nails | 82.6% | 62.6% | 20.0% |
| A029 playing with phone/tablet | 62.2% | 62.9% | -0.7% |
| A103 yawn | 67.0% | 63.5% | 3.5% |
| A030 typing on a keyboard | 91.6% | 64.7% | 26.9% |
| A082 fold paper | 60.0% | 65.6% | -5.6% |
| A083 ball up paper | 65.6% | 65.7% | -0.2% |
| A084 play magic cube | 60.8% | 66.3% | -5.4% |
| A107 wield knife towards other person | 72.0% | 67.2% | 4.9% |
| A077 snapping fingers | 59.4% | 68.5% | -9.1% |
| A017 take off a shoe | 83.2% | 69.3% | 13.9% |

### X-Sub: RGB clearly better (by >= 10%)

| Action | rgb_accuracy | skeleton_accuracy | rgb_minus_skeleton |
|---|---|---|---|
| A030 typing on a keyboard | 91.6% | 64.7% | 26.9% |
| A075 cutting nails | 82.6% | 62.6% | 20.0% |
| A011 reading | 75.5% | 58.2% | 17.2% |
| A094 throw up cap/hat | 97.0% | 82.5% | 14.5% |
| A017 take off a shoe | 83.2% | 69.3% | 13.9% |
| A106 hit other person with something | 73.7% | 61.4% | 12.3% |
| A085 apply cream on face | 95.1% | 82.9% | 12.2% |
| A012 writing | 65.4% | 53.3% | 12.1% |
| A005 drop | 98.2% | 86.5% | 11.6% |
| A073 staple book | 38.9% | 28.0% | 10.9% |

### X-Sub: skeleton clearly better (by >= 10%)

| Action | rgb_accuracy | skeleton_accuracy | rgb_minus_skeleton |
|---|---|---|---|
| A104 stretch oneself | 81.4% | 92.7% | -11.3% |

### X-Set: 20 weakest classes for RGB (R3D-18)

| Action | rgb_accuracy | skeleton_accuracy | rgb_minus_skeleton |
|---|---|---|---|
| A071 make ok sign | 44.9% | 46.9% | -2.0% |
| A072 make victory sign | 46.1% | 51.6% | -5.5% |
| A073 staple book | 46.7% | 48.6% | -1.8% |
| A076 cutting paper (using scissors) | 46.9% | 56.4% | -9.4% |
| A084 play magic cube | 59.3% | 64.4% | -5.1% |
| A074 counting money | 60.1% | 67.9% | -7.8% |
| A077 snapping fingers | 61.1% | 76.3% | -15.1% |
| A069 thumb up | 61.2% | 60.8% | 0.4% |
| A082 fold paper | 61.5% | 83.3% | -21.8% |
| A103 yawn | 62.0% | 66.7% | -4.7% |
| A012 writing | 65.5% | 52.8% | 12.7% |
| A083 ball up paper | 67.8% | 74.9% | -7.1% |
| A107 wield knife towards other person | 67.9% | 69.1% | -1.2% |
| A067 hush (quite) | 70.9% | 71.5% | -0.6% |
| A078 open bottle | 71.5% | 69.9% | 1.6% |
| A086 apply cream on hand back | 74.6% | 83.6% | -9.0% |
| A010 clapping | 75.4% | 87.8% | -12.4% |
| A029 playing with phone/tablet | 75.7% | 67.9% | 7.8% |
| A045 touch chest (stomachache/heart pain) | 77.4% | 83.9% | -6.5% |
| A075 cutting nails | 77.5% | 63.0% | 14.6% |

### X-Set: 20 weakest classes for skeleton (CTR-GCN)

| Action | rgb_accuracy | skeleton_accuracy | rgb_minus_skeleton |
|---|---|---|---|
| A071 make ok sign | 44.9% | 46.9% | -2.0% |
| A073 staple book | 46.7% | 48.6% | -1.8% |
| A072 make victory sign | 46.1% | 51.6% | -5.5% |
| A012 writing | 65.5% | 52.8% | 12.7% |
| A076 cutting paper (using scissors) | 46.9% | 56.4% | -9.4% |
| A069 thumb up | 61.2% | 60.8% | 0.4% |
| A011 reading | 81.0% | 61.2% | 19.8% |
| A075 cutting nails | 77.5% | 63.0% | 14.6% |
| A084 play magic cube | 59.3% | 64.4% | -5.1% |
| A103 yawn | 62.0% | 66.7% | -4.7% |
| A074 counting money | 60.1% | 67.9% | -7.8% |
| A029 playing with phone/tablet | 75.7% | 67.9% | 7.8% |
| A106 hit other person with something | 80.2% | 68.2% | 12.0% |
| A107 wield knife towards other person | 67.9% | 69.1% | -1.2% |
| A078 open bottle | 71.5% | 69.9% | 1.6% |
| A017 take off a shoe | 86.1% | 71.3% | 14.7% |
| A067 hush (quite) | 70.9% | 71.5% | -0.6% |
| A105 blow nose | 79.2% | 71.7% | 7.5% |
| A030 typing on a keyboard | 83.5% | 72.5% | 11.0% |
| A091 open a box | 79.2% | 74.5% | 4.7% |

### X-Set: RGB clearly better (by >= 10%)

| Action | rgb_accuracy | skeleton_accuracy | rgb_minus_skeleton |
|---|---|---|---|
| A011 reading | 81.0% | 61.2% | 19.8% |
| A094 throw up cap/hat | 96.1% | 80.3% | 15.8% |
| A017 take off a shoe | 86.1% | 71.3% | 14.7% |
| A075 cutting nails | 77.5% | 63.0% | 14.6% |
| A012 writing | 65.5% | 52.8% | 12.7% |
| A106 hit other person with something | 80.2% | 68.2% | 12.0% |
| A005 drop | 98.6% | 87.5% | 11.2% |
| A030 typing on a keyboard | 83.5% | 72.5% | 11.0% |

### X-Set: skeleton clearly better (by >= 10%)

| Action | rgb_accuracy | skeleton_accuracy | rgb_minus_skeleton |
|---|---|---|---|
| A082 fold paper | 61.5% | 83.3% | -21.8% |
| A077 snapping fingers | 61.1% | 76.3% | -15.1% |
| A010 clapping | 75.4% | 87.8% | -12.4% |
