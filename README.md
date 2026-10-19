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
  processed/     skeleton_xyz_f16.bin + skeleton_index.csv (packed skeleton cache)
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
python scripts/build_metadata.py          # -> data/metadata/metadata.csv (parser round-trip on every file)
python scripts/check_skeletons.py         # -> skeleton_integrity_report.csv + packed skeleton cache
python scripts/make_splits.py             # -> data/splits/*.csv
python scripts/rgb_frame_counts.py        # -> data/metadata/rgb_frame_counts.csv (exact decoded counts)
python scripts/shrink_rgb.py              # -> data/processed/rgb_256p/*_rgb.mp4 (every frame kept, counts verified)
python scripts/dataset_statistics.py      # -> dataset_statistics.txt / _per_class.csv
python scripts/visualize_sync.py --split data/splits/xsub_train.csv   # -> outputs/sync_check/*.png
```

All settings (data root, seed, frames, crop size, max bodies, validation fraction) live in
`configs/data.yaml`; every script accepts `--config`.

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
  ~33 KB per video). Mean absolute pixel difference of the final clip vs raw: 0.004. Set
  `rgb_dir: null` in the config to read raw videos instead.
- **RGB**: every frame decoded (exact count), resized to 128 x 171, center-cropped to 112 x 112,
  values in [0, 1] (normalization belongs to the model).
- **Synchronization**: `uniform_temporal_indices(n, 16)` maps the same 16 normalised positions
  (0 to 1, both ends included) into each modality's own frame count.
- **Integrity**: fatal = parse error, empty sequence, no bodies, NaN, Inf, all-zero.
  Recorded but non-fatal = more than 2 bodies, frames without any body.
