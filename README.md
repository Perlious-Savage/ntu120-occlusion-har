# Occlusion-Robust RGB-Skeleton Action Recognition — NTU RGB+D 120

Milestone 1: multimodal data pipeline (metadata, official splits, synchronized RGB + skeleton loading).

## Layout

```
data/
  zips/          original NTU zips (never renamed)
  raw/rgb/       S###C###P###R###A###_rgb.avi   (extracted, original names)
  raw/skeleton/  S###C###P###R###A###.skeleton  (extracted, original names)
  metadata/      metadata.csv, integrity report, statistics, official NTU missing-skeleton list
  splits/        X-Sub / X-Set train/test + internal train/val CSVs
  processed/     rgb_256p/ (shrunk videos + frame counts), skeleton_xyz_f16.bin + skeleton_index.csv
src/datasets/    skeleton_loader.py, rgb_loader.py, temporal.py, ntu120_multimodal.py
src/utils/       ntu.py (sample-ID parser, protocols, skeleton bones)
scripts/         command-line pipeline steps (below)
configs/         data.yaml (all data settings, seed 42)
outputs/         visualizations and experiment results
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
| X-Sub | train 63,026 / test 50,919; internal train 51,557 / val 11,469 (subjects 8, 47, 70, 82, 103) |
| X-Set | train 54,468 / test 59,477; internal train 49,021 / val 5,447 |
| RGB vs skeleton frame count | identical for all 114,480 samples (min 15, max 300, mean 71.6) |

## Milestone 2 (code ready; full runs after Milestone 1 review)

```
python scripts/train.py --config configs/ctrgcn_xsub.yaml --overfit 64 --epochs 60   # tiny-subset check
python scripts/train.py --config configs/r3d18_xsub.yaml        # resumes from outputs/<run>/last.pt
python scripts/evaluate.py --run outputs/r3d18_xsub_seed42
python scripts/compare_baselines.py                              # results table + class comparison
```

Both baselines reach 100% training accuracy on the 64-sample overfit test. On a 6 GB GPU use
`--batch-size 8 --workers 4` for local checks.

## Decisions (documented per the milestone rules)

- **Labels**: `action_id` 1-120, `label = action_id - 1` (0-119).
- **X-Sub**: the 53 official training subjects in `src/utils/ntu.py`; all others are test.
  **X-Set**: even setup IDs train, odd setup IDs test.
- **Usable samples**: both modalities present, not on NTU's official missing-skeleton list, and no
  fatal integrity issue. The same set is used for every modality so results are comparable.
- **Internal validation** (seed 42, 10%): X-Sub holds out whole training subjects (re-drawn with
  incremented seeds until every class is covered); X-Set uses a class-stratified 10% split.
  The test sets are never used for model selection.
- **Skeleton tensor**: C x T x V x M = 3 x T x 25 x 2 (x, y, z in metres). When more than 2 people
  are tracked, the 2 with the most motion are kept. Untracked frames are zeros.
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
