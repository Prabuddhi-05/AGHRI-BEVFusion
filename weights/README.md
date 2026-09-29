# AGHRI Checkpoints

This folder holds selected **AGHRI-BEVFusion unimodal and fusion checkpoints** for use with the corresponding AGHRI configs. The `.pth` files are configured for Git LFS and can be fetched when this repository is published on GitHub.

## Available Checkpoints

| ID | Local file | Config | Model / geometry |
|---|---|---|---|
| `C2-DA` | [C2-DA_best_epoch_20.pth](C2-DA_best_epoch_20.pth) | `configs/aghri/camera/camera_only_nuimages_distortion_aware.yaml` | Camera-only, distortion-aware fisheye LSS |
| `L2` | [L2_best_epoch_1.pth](L2_best_epoch_1.pth) | `configs/aghri/lidar/lidar_only_nuscenes_pretrained.yaml` | LiDAR-only |
| `F1` | [F1_best_epoch_1.pth](F1_best_epoch_1.pth) | `configs/aghri/fusion/fusion_pinhole.yaml` | Fusion, standard pinhole LSS |
| `F2` | [F2_best_epoch_1.pth](F2_best_epoch_1.pth) | `configs/aghri/fusion/fusion_distortion_aware.yaml` | Fusion, distortion-aware fisheye LSS |
| `D-F4` | [D-F4_best_epoch_1.pth](D-F4_best_epoch_1.pth) | `configs/aghri/fusion/fusion_distortion_aware_df.yaml` | Fusion, distortion-aware fisheye LSS |

C2-DA is the AGHRI camera-only checkpoint fine-tuned from nuImages initialization; L2 is the AGHRI LiDAR-only checkpoint fine-tuned from nuScenes initialization. D-F4 was initialized from these two AGHRI checkpoints. F1, F2, and D-F4 are complete fusion-model checkpoints.

## Evaluation Data

C2-DA and the three fusion checkpoints use the corrected fisheye PKL family:

```text
pkls/fisheye_lidar/pkls/aghri_fisheye_lidar_infos_val.pkl
pkls/fisheye_lidar/pkls/aghri_fisheye_lidar_infos_test.pkl
```

L2 uses the `zed_lidar` PKL family as required by the existing LiDAR-only batch interface. When constructing C2-DA, F1, or F2 for evaluation, the launcher also requires the upstream nuImages Swin initialization file.

The C2-DA and L2 files here are the AGHRI-trained unimodal checkpoints used to initialize D-F4 training; they are not the original upstream pretrained binaries.

## File Sizes

| File | Bytes |
|---|---:|
| `C2-DA_best_epoch_20.pth` | 418817104 |
| `L2_best_epoch_1.pth` | 99783205 |
| `F1_best_epoch_1.pth` | 462981640 |
| `F2_best_epoch_1.pth` | 462981640 |
| `D-F4_best_epoch_1.pth` | 462981704 |

For evaluation and visualization commands, see [`docs/TRAINING_AND_EVALUATION.md`](../docs/TRAINING_AND_EVALUATION.md).
