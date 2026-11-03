# Corruption protocol (Milestone 3)

Six modality-specific corruptions, each with severities S1-S5. Code: `src/corruptions/`; levels:
`configs/corruptions.yaml`; tests: `tests/test_corruptions.py`; examples: `scripts/visualize_corruptions.py`.

## Common rules

- **Interface.** `get(name).apply(x, severity, seed, **context)` returns a corrupted copy; severity 0
  returns the input unchanged.
- **Where corruption is applied.** On the exact model input of the frozen Milestone 2 evaluation
  pipeline:
  - RGB: the uint8 clip of 16 uniformly sampled frames, resized to 128 x 171 and centre-cropped to
    112 x 112 (before normalisation);
  - skeleton: the 3 x 64 x 25 x 2 CTR-GCN input after the official eval crop (central 95%) and resize
    to 64 frames.
- **Determinism.** Every random choice uses `numpy.random.default_rng(seed)` with
  `seed = crc32("<sample_id>|<corruption>|<severity>|42")` (`src/corruptions/base.py:sample_seed`). The same
  sample therefore receives the same corruption on any machine, in any run and in any batch order.
- **Severities are independent draws.** S3 is not S1 with extra masking added; each severity has its own seed.
- **The other modality stays clean.** In single-modality experiments, only one modality is corrupted.
  In dual-corruption experiments, both are corrupted independently, each with its own seed.
- **Untracked joints.** In NTU skeletons, a joint that is all zeros means "not tracked". Dropout uses
  that same value. Noise is never added to untracked joints.
- **Persons.** For a single actor, CTR-GCN's convention copies the actor into slot 2. The skeleton
  corruptions corrupt that actor once and copy the result again, so the copy carries no clean
  information. Two real actors are corrupted independently.

## RGB

| Corruption | Definition | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|---|
| `rgb_occlusion` | One black rectangle per person, fixed in every frame, covering a fraction of that person's bounding box | 10% | 20% | 30% | 40% | 50% |
| `rgb_blur` | Gaussian blur of every frame; kernel size in px (sigma from OpenCV's rule) | 3 | 5 | 7 | 11 | 15 |
| `rgb_temporal_mask` | One contiguous block of black frames, as a fraction of the 16 frames (frames masked) | 10% (2) | 20% (3) | 30% (5) | 40% (6) | 50% (8) |

**Person boxes for occlusion** (`scripts/person_boxes.py`, stored in `data/metadata/person_boxes.csv`):

- **How a box is built.** One box per actor, from the skeleton's colour-camera joint projections: the
  2nd-98th percentile of that actor's tracked joints over all frames, in 1920 x 1080 pixels. The box is
  then mapped into the 112 x 112 crop.
- **What it covers.** A box spans where the person is during the whole clip. For a moving person, the
  rectangle therefore acts as a static obstacle in their path.
- **Rectangle placement.** The aspect ratio is drawn from [0.5, 2] and the position is uniform inside the box.
- **Two-person samples.** Each person gets their own rectangle (25,622 samples). One box around both
  people would mostly cover empty floor between them.
- **Missing box.** 1,381 samples have no tracked joints for a box. They use the central half of the frame.
- **How small the people are.** The median actor box is 173 x 446 px in 1080p, about 15 x 53 px in the
  model input. At S5, one person therefore loses about 400 px², roughly 3% of the frame.
- **What the results can show.** How much R3D-18 relies on the person rather than on the scene.

## Skeleton

| Corruption | Definition | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|---|
| `skeleton_joint_dropout` | Fraction of each person's tracked (frame, joint) observations set to zero, drawn independently | 5% | 10% | 20% | 30% | 40% |
| `skeleton_part_dropout` | Joint groups removed in all frames for all persons | one leg | one arm | lower body | upper body | upper body + one leg |
| `skeleton_noise` | i.i.d. Gaussian on tracked joints, sigma = alpha x body height (median per-frame y extent of the person) | 0.01 | 0.02 | 0.04 | 0.08 | 0.12 |

**Body-part groups** (0-based NTU joints):

| Group | Joints |
|---|---|
| Left arm | 4-7, 21, 22 |
| Right arm | 8-11, 23, 24 |
| Left leg | 12-15 |
| Right leg | 16-19 |
| Upper body | both arms + 1, 2, 3, 20 (spine middle, neck, head, spine shoulder), 16 joints |
| Lower body | both legs + 0 (spine base), 9 joints |

- **Joints lost per severity:** 4, 6, 9, 16 and 20 of 25.
- **Which side.** The leg or arm is chosen at random (seeded). The same group is removed for both persons.
- **Why S1 is a leg and S2 an arm.** Severity is ordered by the number of joints removed. Most NTU
  actions are arm-driven, so an uneven drop in accuracy between severities is itself a result about
  which body parts matter.

## Visual validation (step 3.5)

- **Samples.** `scripts/visualize_corruptions.py --n 10` renders clean, S1, S3 and S5 for 10 X-Sub test
  samples of 10 different actions. They span cameras C001-C003, three subjects, standing and seated
  actions, and one- and two-person actions.
- **Setup coverage.** Only setup S001 has RGB on the development PC, which is why all 10 come from it.

**What was checked:**

- S1 is mild in every case; S3 is clearly degraded; S5 is severe but still recognisable as a person or pose.
- Occlusion lies on the people, including both people in two-person actions.
- The temporal mask removes 2, 5 and 8 contiguous frames at S1, S3 and S5.
- Body-part dropout removes the stated joints.
- Noise is scaled to each person's height.

**Observations:**

- S5 noise (sigma about 0.12 x height, roughly 19 cm for a standing adult) and S5 part dropout
  (20 of 25 joints) are the harshest settings. They follow the guide's suggested schedule.
- S5 blur (15 px kernel at 112 px) still shows the body outline but no limb detail.

## Benchmark runs

- **Script.** `scripts/evaluate_corruptions.py` evaluates each frozen Milestone 2 checkpoint
  (`best_val_accuracy.pt`) on its test split.
- **Conditions.** Clean plus the 15 conditions of its own modality (3 corruptions x S1-S5). All 16 are
  built from one load of each sample.
- **What is saved per condition.** Probabilities, branch features, predictions, per-class accuracy and metrics.
- **Fusion.** The fusion baselines (`scripts/train_fusion.py`) reuse these features. A condition with
  RGB corrupted and skeleton clean combines the corrupted RGB features with the clean skeleton features.

**Dual-corruption cases** (`configs/fusion.yaml`):

- Corruption pairs: occlusion + body-part dropout, blur + coordinate noise, and temporal mask + joint dropout.
- Severity pairs (RGB, skeleton): (2, 2), (3, 3), (4, 4), (4, 1) and (1, 4).
