# Training and Evaluation

This guide explains how to train and evaluate the AGHRI configurations provided in **AGHRI-BEVFusion**.

The main user-facing entrypoint is [`tools/run_aghri.py`](../tools/run_aghri.py). It selects and validates an AGHRI experiment, then calls the existing BEVFusion training or evaluation code. It does not implement a separate optimizer, model, or evaluator.

## Environment

Use a BEVFusion-compatible environment with the upstream CUDA/PyTorch/MMCV/MMDetection stack, together with TorchPack and MPI.

After preparing the environment, install this repository from its root:

```bash
python setup.py develop
```

The upstream [BEVFusion repository](https://github.com/mit-han-lab/bevfusion) provides the base installation guidance.

## Data Paths

Download the original AGHRI dataset from the [University of Lincoln record](https://doi.org/10.24385/lincoln.32982638), then run discovery, conversion, and PKL generation with [AGHRI-nuScenes-tools](https://github.com/Prabuddhi-05/AGHRI-nuScenes-tools). The tools README documents the source extraction layout and `paths.env` settings. The launcher reads the generated dataset and PKL roots.

Use the tools repository's `AGHRI_DATASET_ROOT` and `AGHRI_PKL_OUTPUT_ROOT` values:

```text
--dataset-root /path/to/aghri_nuscenes_run/AGHRI_nuScenes_camera_lidar_full
--pkl-root /path/to/aghri_pkls
```

Camera and fusion experiments use:

```text
pkls/fisheye_lidar/pkls/aghri_fisheye_lidar_infos_{train,val,test}.pkl
```

LiDAR-only experiments use:

```text
pkls/zed_lidar/pkls/aghri_zed_lidar_infos_{train,val,test}.pkl
```

The launcher checks the required data and checkpoint paths in normal execution mode.

Use:

```text
--dry-run
```

to inspect the resolved experiment, paths, initialization, and underlying command without starting a job.

## Selecting an Experiment

`--experiment` accepts the experiment IDs documented in [EXPERIMENTS.md](EXPERIMENTS.md).

For example:

```text
C1-Pinhole
C1-DA
C2-Pinhole
C2-DA
L1
L2
F1
F2
D-F1
D-F2
D-F3
D-F4
```

Training can also be selected through the launcher's modality, geometry, and initialization options. Unsupported combinations are rejected rather than silently changed.

`--gpus N` controls the number of TorchPack processes through `-np N`.

## Training Schedules

The current configs use:

- **20 epochs** for camera-only and LiDAR-only configurations;
- **6 epochs** for the fusion configurations in this release.

The trainer validates after each epoch and selects the best checkpoint using AGHRI validation mAP.

## Pretrained Weights

Upstream initialization weights can be stored locally at:

```text
pretrained/
├── swint-nuimages-pretrained.pth
└── lidar-only-det.pth
```

or supplied explicitly using:

```text
--camera-pretrained-checkpoint
--lidar-pretrained-checkpoint
```

The compatible AGHRI checkpoint loader records what was loaded, skipped, or incompatible in:

```text
checkpoint_load_report.json
```

inside the selected run directory.

## Training Examples

### Distortion-Aware Fusion Baseline

The following example selects `F2`:

```bash
python tools/run_aghri.py train --experiment F2   --dataset-root /path/to/aghri_nuscenes_run/AGHRI_nuScenes_camera_lidar_full   --pkl-root /path/to/aghri_pkls   --run-dir runs/F2   --gpus 4   --dry-run
```

Remove `--dry-run` when you are ready to start training.

### D-F4

D-F experiments use AGHRI-trained camera and LiDAR branches.

For example:

```bash
python tools/run_aghri.py train --experiment D-F4   --camera-checkpoint weights/C2-DA_best_epoch_20.pth   --lidar-checkpoint weights/L2_best_epoch_1.pth   --dataset-root /path/to/aghri_nuscenes_run/AGHRI_nuScenes_camera_lidar_full   --pkl-root /path/to/aghri_pkls   --run-dir runs/D-F4   --gpus 4   --dry-run
```

For D-F, the launcher first calls [`build_aghri_df_branch_init.py`](../tools/build_aghri_df_branch_init.py) to prepare the combined initialization checkpoint, then starts [`train_aghri.py`](../tools/train_aghri.py).

The generated run files include:

```text
branch_encoder_initialization.pth
branch_encoder_initialization.json
checkpoint_load_report.json
```

These are run outputs rather than source files.

## Underlying TorchPack Training Command

For users who want to see the underlying command, the launcher resolves to the general pattern:

```bash
torchpack dist-run -np N python tools/train_aghri.py CONFIG   --run-dir RUN_DIR   --dataset_root DATASET   --train_ann_file TRAIN_PKL   --val_ann_file VAL_PKL   --test_ann_file TEST_PKL
```

Depending on the selected experiment, it also supplies the required camera initialization or AGHRI-compatible pretrained checkpoint.

## Evaluation

Use the validation split while selecting a model, and the held-out test split for final evaluation.

Example validation command:

```bash
python tools/run_aghri.py eval --experiment F2   --checkpoint /path/to/F2-best.pth   --split val   --dataset-root /path/to/aghri_nuscenes_run/AGHRI_nuScenes_camera_lidar_full   --pkl-root /path/to/aghri_pkls   --gpus 1   --dry-run
```

Change:

```text
--split val
```

to:

```text
--split test
```

for test evaluation.

The launcher calls the existing BEVFusion test entrypoint using the selected AGHRI config and PKL.

The underlying pattern is:

```bash
torchpack dist-run -np N python tools/test.py CONFIG CHECKPOINT   --eval bbox   --cfg-options   data.test.dataset_root=DATASET   data.test.ann_file=SPLIT_PKL
```

## AGHRI Evaluation Metrics

The AGHRI evaluator reports human-detection performance using centre-distance matching.

The main reported metrics are:

- **AGHRI mAP** — mean AP across centre-distance thresholds of 0.5 m, 1 m, 2 m, and 4 m;
- **ATE** — translation error;
- **ASE** — scale error.

The evaluator also provides **velocity error (AVE)** as a secondary diagnostic at the 2 m matching threshold.

Attribute error and nuScenes Detection Score are not used for this AGHRI evaluation.

For the experiment definitions and initialization relationships, see [EXPERIMENTS.md](EXPERIMENTS.md).

## Visualization

Use the corrected fisheye test PKL with the complete D-F4 checkpoint to generate raw-camera overlays and LiDAR BEV visualizations:

```bash
torchpack dist-run -np 1 python tools/visualize_aghri.py \
  configs/aghri/fusion/fusion_distortion_aware_df.yaml \
  --mode pred \
  --checkpoint weights/D-F4_best_epoch_1.pth \
  --split test \
  --dataset-root /path/to/aghri_nuscenes_run/AGHRI_nuScenes_camera_lidar_full \
  --ann-file /path/to/aghri_pkls/fisheye_lidar/pkls/aghri_fisheye_lidar_infos_test.pkl \
  --out-dir output/aghri_D-F4_test \
  --max-samples 10
```
