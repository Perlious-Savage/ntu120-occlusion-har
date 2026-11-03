# Occlusion-Robust RGB-Skeleton Action Recognition — NTU RGB+D 120

Milestones 1-3: multimodal data pipeline, clean baselines, corruption benchmark and fusion baselines.

## Layout

```
data/
  zips/          original NTU zips (never renamed)
  raw/rgb/       S###C###P###R###A###_rgb.avi   (extracted, original names)
  raw/skeleton/  S###C###P###R###A###.skeleton  (extracted, original names)
  metadata/      metadata.csv, integrity report, statistics, official NTU missing-skeleton list
  splits/        X-Sub / X-Set train/test + internal train/val CSVs
  processed/     rgb_256p/ (shrunk videos + frame counts), skeleton_xyz_f16.bin + skeleton_index.csv
src/datasets/    skeleton_loader.py, rgb_loader.py, temporal.py, ntu120_multimodal.py, baseline_datasets.py
src/corruptions/ six corruptions (common apply(x, severity, seed)), registry.py
src/models/      baselines.py (R3D-18, CTR-GCN), fusion.py (concat, global, attention heads)
docs/            corruption_protocol.md
src/utils/       ntu.py (sample-ID parser, protocols, skeleton bones)
scripts/         command-line pipeline steps (below)
configs/         data.yaml, baseline configs, corruptions.yaml, fusion.yaml (seed 42)
outputs/         visualizations and experiment results
results/         committed milestone results (tables, figures)
```

## Environment

```
conda create -n ntu120 -c conda-forge --override-channels python=3.12
conda activate ntu120
pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu130
pip install numpy pandas pyyaml opencv-python matplotlib scikit-learn tqdm tensorboard
```

## Data

Download from https://rose1.ntu.edu.sg/dataset/actionRecognition/download (login required):
`nturgbd_skeletons_s001_to_s017.zip`, `nturgbd_skeletons_s018_to_s032.zip`,
`nturgbd_rgb_s001.zip` … `nturgbd_rgb_s032.zip` (~269 GB). Put them in `data/zips/` and extract
the files (without the zip's inner folder) into `data/raw/skeleton/` and `data/raw/rgb/`.

`data/metadata/NTU_RGBD120_samples_with_missing_skeletons.txt` is NTU's official list of 535 samples
with missing/incomplete skeletons (from github.com/shahroudy/NTURGB-D).

## Rebuild everything (run from the project root)

```
python tests/test_core.py                 # parser, protocol and temporal-sampling checks
python scripts/shrink_rgb.py              # raw/rgb -> processed/rgb_256p/*_rgb.mp4 + frame_counts_all.csv
                                          #   (or, zip by zip with little disk: scripts/process_rgb_zips.py)
python scripts/build_metadata.py          # -> data/metadata/metadata.csv (parser round-trip on every file)
python scripts/check_skeletons.py         # -> skeleton_integrity_report.csv + packed skeleton cache
python scripts/make_splits.py             # -> data/splits/*.csv
python scripts/dataset_statistics.py      # -> dataset_statistics.txt / _per_class.csv
python scripts/visualize_sync.py --split data/splits/xsub_train.csv   # -> outputs/sync_check/*.png
```

All settings (data root, seed, frames, crop size, max bodies, validation fraction) live in
`configs/data.yaml`; every script accepts `--config`.

### On Google Colab (not enough disk for 269 GB of raw videos)

With the raw zips on Google Drive (`/content/drive/MyDrive/ntu120/raw`), each Colab session runs:

```
python scripts/process_rgb_zips.py --zips /content/drive/MyDrive/ntu120/raw --store /content/drive/MyDrive/ntu120_work
```

Every session takes zips from a shared queue (a `.claim` file on the store per zip), copies and
extracts 3 at a time, shrinks them, and saves one `rgb_256p_sXXX.tar` per setup to the store, so
any number of sessions split the work and a disconnected session simply resumes. Afterwards, on
one session: `--restore` (unpacks all tars), unzip the two skeleton zips into `data/raw/skeleton`
(`unzip -j`), then run the remaining steps above. The full run used 3 CPU sessions for about 1.5 h.

## Milestone 1 results (full dataset)

| | |
|---|---|
| Sample IDs | 114,480 (all with RGB and skeleton; 0 missing files) |
| Usable after integrity filtering | 113,945 (the 535 samples on NTU's missing-skeleton list removed) |
| Fatal skeleton problems found | 356 (no tracked body), all on NTU's official list |
| X-Sub | train 63,026 / test 50,919; internal train 56,594 / val 6,432 (7 subjects, 46-60 per class) |
| X-Set | train 54,468 / test 59,477; internal train 49,021 / val 5,447 |
| RGB vs skeleton frame count | identical for all 114,480 samples (min 15, max 300, mean 71.6) |
| Noise bodies (official denoising) | removed in 3,296 samples; a simple top-2-by-motion rule would have kept one as a "person" in 2,873 (2.5%) |
| Two-person actions with only one person ever tracked | 684 of 24,732 (Kinect limitation; the single actor is used) |

## Milestone 2 results (clean data, seed 42, test split, best-validation-accuracy checkpoint)

| Model | Protocol | Top-1 Acc. | Top-5 Acc. | Macro-F1 | Best Epoch (0-based) |
|---|---|---|---|---|---|
| R3D-18 | X-Sub | 85.75% | 98.02% | 0.8669 | 26 |
| R3D-18 | X-Set | 87.52% | 98.39% | 0.8735 | 26 |
| CTR-GCN (joint) | X-Sub | 83.54% | 96.79% | 0.8380 | 58 |
| CTR-GCN (joint) | X-Set | 86.23% | 97.09% | 0.8620 | 64 |

Measured per-class differences (`results/milestone2/baseline_results.md`, `class_comparison_*.csv`):
RGB is clearly better (>= 10 points) on object- and appearance-defined actions (typing on a keyboard,
cutting nails, reading, writing, take off a shoe, throw up cap/hat, drop); skeleton is clearly better
on motion-defined ones (stretch oneself; fold paper, snapping fingers, clapping on X-Set). Both are
weakest on fine hand gestures (make OK sign, victory sign, thumb up, staple book, counting money).

Per run, `results/milestone2/<run>/` keeps the config, training log, test metrics, per-class accuracy,
learning curves and confusion matrix. Checkpoints, `predictions.csv` and `probabilities.npy` are on
Google Drive (`ntu120_work/outputs/<run>/`). Runs were moved between Colab GPUs mid-training by
resuming from `last.pt`; the logs show one continuous run per experiment.

## Milestone 2 commands

```
python scripts/train.py --config configs/ctrgcn_xsub.yaml --overfit 64 --epochs 60   # tiny-subset check
python scripts/train.py --config configs/r3d18_xsub.yaml        # resumes from outputs/<run>/last.pt
python scripts/evaluate.py --run outputs/r3d18_xsub_seed42
python scripts/compare_baselines.py                              # results table + class comparison
```

Both baselines reach 100% training accuracy on the 64-sample overfit test. On a 6 GB GPU use
`--batch-size 8 --workers 4` for local checks.

## Milestone 3 commands (corruption benchmark and fusion baselines)

Corruption definitions and levels: `docs/corruption_protocol.md`, `configs/corruptions.yaml`.
The Milestone 2 checkpoints (`best_val_accuracy.pt`) stay frozen throughout.

```
python tests/test_corruptions.py                                  # determinism, S1-S5 schedules
python scripts/person_boxes.py                                    # per-actor boxes -> data/metadata/person_boxes.csv
python scripts/visualize_corruptions.py --n 10                    # clean / S1 / S3 / S5 examples

# single-modality robustness: clean + 15 conditions of the model's modality, plus branch features
python scripts/evaluate_corruptions.py --run outputs/r3d18_xsub_seed42 --split test
python scripts/evaluate_corruptions.py --run outputs/ctrgcn_xsub_seed42 --split test
python scripts/evaluate_corruptions.py --run outputs/r3d18_xsub_seed42 --split val --clean-only
python scripts/evaluate_corruptions.py --run outputs/ctrgcn_xsub_seed42 --split val --clean-only
python scripts/evaluate_corruptions.py --run outputs/r3d18_xsub_seed42 --split train --clean-only   # ablation only
python scripts/evaluate_corruptions.py --run outputs/ctrgcn_xsub_seed42 --split train --clean-only  # ablation only

# fusion baselines (average, concat, global, attention), tested on 46 conditions without retraining
python scripts/train_fusion.py --protocol xsub --method attention
python scripts/train_fusion.py --protocol xsub --method attention --train-split train --epochs 30   # ablation

python scripts/m3_report.py                                       # results/milestone3: CSVs, tables, figures
```

Repeat with `xset` for X-Set. Each `evaluate_corruptions.py` run writes resumable shards, so a
reclaimed Colab session continues where it stopped. On an A100, the RGB test benchmark takes about
20-25 minutes per protocol and the skeleton benchmark about 10 minutes.

**Fusion training data.** Both frozen backbones fit their own training data almost perfectly
(R3D-18 99.98%, CTR-GCN 99.6-99.9%). Heads trained on those features cannot learn when a branch
fails; attention fusion collapsed onto RGB, with a weight of 1.0 for every sample. The fusion heads
are therefore trained on the held-out validation split, where the backbones make realistic errors.
5-fold class-stratified cross-validation inside that split picks the stopping epoch. Subject-grouped
folds were tried for X-Sub and fail: NTU-120 ties actions to subject groups, so a held-out fold can
contain classes missing from its training part. The test set is never used. Training on the training
features is kept as an ablation in `outputs/m3_fusion_trainfeat/`.

## Decisions (documented per the milestone rules)

- **Labels**: `action_id` 1-120, `label = action_id - 1` (0-119).
- **X-Sub**: the 53 official training subjects in `src/utils/ntu.py`; all others are test.
  **X-Set**: even setup IDs train, odd setup IDs test.
- **Usable samples**: both modalities present, not on NTU's official missing-skeleton list, and no
  fatal integrity issue. The same set is used for every modality so results are comparable.
- **Internal validation** (10% of each training set): X-Sub holds out whole training subjects.
  Subjects differ a lot in size (subject 8 alone has 15% of X-Sub train), so subjects are added in
  a seeded random order until they hold 10% of the training samples; the first draw from seed 42
  that is within 10% +- 0.5% and gives every class >= 75% of its fair share is used (seed 51:
  subjects 25, 53, 54, 55, 97, 98, 100). X-Set uses a class-stratified 10% split (seed 42).
  The test sets are never used for model selection.
- **Skeleton tensor**: C x T x V x M = 3 x T x 25 x 2 (x, y, z in metres). Actors are chosen with
  the official NTU denoising used by CTR-GCN (Microsoft SGN `get_raw_denoised_data.py`): bodies
  tracked <= 11 frames, or wider than tall in >= 69.754% of their frames (Kinect false detections
  such as furniture), are removed; the rest are ranked by motion; bodies whose frames do not overlap
  an actor's span are merged into it (a person whose tracking ID changed). A single actor is copied
  into the second slot, as the official `align_frames` does. Unlike the official code, true frame
  numbers are kept so RGB and skeleton frames stay one-to-one. Untracked frames are zeros.
- **Skeleton cache**: float16 (~2 mm precision) packed into one file for fast loading.
- **RGB storage**: raw 1080p MPEG-4 takes ~2 s per video to decode, so videos are re-encoded once
  to 256 x 456 H.264 (`shrink_rgb.py`, same sample ID, every frame kept, frame count verified;
  ~33 KB per video). Mean absolute pixel difference of the final clip vs raw: 0.004. Metadata,
  training and the sync visualization all use these files (config `rgb_dir`); the exact raw
  frame counts recorded during shrinking are used for the statistics.
- **RGB**: every frame decoded (exact count), resized to 128 x 171, center-cropped to 112 x 112,
  values in [0, 1] (normalization belongs to the model).
- **Synchronization**: `uniform_temporal_indices(n, 16)` maps the same 16 normalised positions
  (0 to 1, both ends included) into each modality's own frame count.
- **Integrity**: fatal = parse error, empty sequence, no bodies, NaN, Inf, all-zero.
  Recorded but non-fatal = more than 2 bodies, frames without any body.
