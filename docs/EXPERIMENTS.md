# AGHRI Experiments

This page summarizes the experiment configurations used in **AGHRI-BEVFusion** and maps each experiment label to its modality, geometry, initialization, config, best epoch, and training schedule.

All experiment configs are stored under:

```text
configs/aghri/
```

## Experiment Summary

| ID | Modality / geometry | Camera initialization | LiDAR initialization | Config under `configs/aghri/` | Best epoch | Current config epochs |
|---|---|---|---|---|---:|---:|
| `C1-Pinhole` | Camera / pinhole | Random | — | `camera/camera_only_scratch_pinhole.yaml` | 11 | 20 |
| `C1-DA` | Camera / distortion-aware | Random | — | `camera/camera_only_scratch_distortion_aware.yaml` | 3 | 20 |
| `C2-Pinhole` | Camera / pinhole | nuImages Swin | — | `camera/camera_only_nuimages_pinhole.yaml` | 9 | 20 |
| `C2-DA` | Camera / distortion-aware | nuImages Swin | — | `camera/camera_only_nuimages_distortion_aware.yaml` | 20 | 20 |
| `L1` | LiDAR-only | — | Random | `lidar/lidar_only_scratch.yaml` | 4 | 20 |
| `L2` | LiDAR-only | — | nuScenes | `lidar/lidar_only_nuscenes_pretrained.yaml` | 1 | 20 |
| `F1` | Fusion / pinhole | nuImages Swin | nuScenes | `fusion/fusion_pinhole.yaml` | 1 | 6 |
| `F2` | Fusion / distortion-aware | nuImages Swin | nuScenes | `fusion/fusion_distortion_aware.yaml` | 1 | 6 |
| `D-F1` | Fusion / distortion-aware | C1-DA | L1 | `fusion/fusion_distortion_aware_df.yaml` | 2 | 6 |
| `D-F2` | Fusion / distortion-aware | C1-DA | L2 | same D-F config | 1 | 6 |
| `D-F3` | Fusion / distortion-aware | C2-DA | L1 | same D-F config | 4 | 6 |
| `D-F4` | Fusion / distortion-aware | C2-DA | L2 | same D-F config | 1 | 6 |

`DA` means **distortion-aware LSS**.

## Initialization Groups

### Camera-only

- **C1-Pinhole** — camera-only model trained from random initialization on AGHRI using standard pinhole LSS.
- **C1-DA** — camera-only model trained from random initialization on AGHRI using distortion-aware LSS.
- **C2-Pinhole** — camera-only model initialized from the nuImages-pretrained Swin backbone and fine-tuned on AGHRI using pinhole LSS.
- **C2-DA** — camera-only model initialized from the nuImages-pretrained Swin backbone and fine-tuned on AGHRI using distortion-aware LSS.

### LiDAR-only

- **L1** — LiDAR-only model trained from random initialization on AGHRI.
- **L2** — LiDAR-only model initialized from the nuScenes-pretrained LiDAR detector and fine-tuned on AGHRI.

### Fusion Baselines

- **F1** — camera–LiDAR fusion using the upstream pretrained camera and LiDAR branches with standard pinhole LSS.
- **F2** — the same pretrained fusion setup with distortion-aware LSS.

### D-F Fusion

The **D-F1–D-F4** experiments use AGHRI-trained unimodal branches to initialize distortion-aware camera–LiDAR fusion:

| ID | Camera branch | LiDAR branch |
|---|---|---|
| `D-F1` | C1-DA | L1 |
| `D-F2` | C1-DA | L2 |
| `D-F3` | C2-DA | L1 |
| `D-F4` | C2-DA | L2 |

The D-F initialization helper transfers the camera backbone together with the LiDAR encoder, decoder, and head. The camera neck/LSS and fusion module are initialized for the new fusion run.

## Required Upstream Weights

The following experiments require upstream initialization files:

- **C2-Pinhole, C2-DA, F1, and F2** use `swint-nuimages-pretrained.pth`.
- **L2, F1, and F2** use `lidar-only-det.pth`.
- **C1-Pinhole, C1-DA, and L1** do not require upstream pretrained weights.

D-F experiments use **AGHRI-trained unimodal checkpoints** supplied through:

```text
--camera-checkpoint
--lidar-checkpoint
```

These are different from the original upstream initialization binaries.

## Running an Experiment

The same launcher is used for every experiment:

```bash
python tools/run_aghri.py train --experiment ID   --dataset-root /path/to/aghri_nuscenes_run/AGHRI_nuScenes_camera_lidar_full   --pkl-root /path/to/aghri_pkls   --run-dir runs/ID   --gpus 4   --dry-run
```

Use `--dry-run` to inspect the selected config, PKLs, initialization, and underlying command.

Remove `--dry-run` when the required dataset and checkpoints are available and you are ready to start the run.

For D-F experiments, also provide the corresponding AGHRI camera and LiDAR branch checkpoints.

## Checkpoints

Selected AGHRI-trained unimodal and fusion checkpoints are listed under `weights/` and configured for Git LFS. See [`weights/README.md`](../weights/README.md) for the C2-DA and L2 branch checkpoints used by D-F4, plus the F1, F2, and D-F4 complete fusion checkpoints.

For detailed launcher, training, and evaluation instructions, see [TRAINING_AND_EVALUATION.md](TRAINING_AND_EVALUATION.md).
