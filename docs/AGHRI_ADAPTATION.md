# AGHRI Adaptation

This guide explains the main changes that adapt [MIT-HAN-Lab BEVFusion](https://github.com/mit-han-lab/bevfusion) for 3D human detection on the **AGHRI** dataset.

AGHRI is used through its **AGHRI-to-nuScenes-style representation**, while the original BEVFusion architecture remains the foundation of the model. The adaptation focuses on AGHRI data loading, one-class human detection, camera geometry, modality-specific configurations, and evaluation.

## Data and Model

The AGHRI dataset interface is implemented in [`aghri_dataset.py`](../mmdet3d/datasets/aghri_dataset.py).

`AGHRIDataset`:

- reads the AGHRI information PKLs;
- resolves image and LiDAR paths relative to `dataset_root`;
- uses the single detection class `human`;
- enforces the expected camera order; and
- uses LiDAR keyframes without additional LiDAR sweeps.

The AGHRI evaluator computes mean average precision for human-centre distance thresholds of **0.5 m, 1 m, 2 m, and 4 m**. At the 2 m matching threshold, it also reports translation error (ATE), scale error (ASE), and velocity error (AVE) as a secondary diagnostic.

These metrics are evaluated for the AGHRI detection setting rather than as the complete official nuScenes evaluation protocol.

### Sensor Inputs

For fisheye experiments, the camera order is:

```text
cam_fish_front
cam_fish_left
cam_fish_right
```

The AGHRI image loader is implemented in [`loading_aghri.py`](../mmdet3d/datasets/pipelines/loading_aghri.py). It reads the raw camera images and converts RGBA images to RGB **in memory** where required. The stored images themselves are not rectified or undistorted.

For LiDAR-only experiments, the camera encoder and fusion module are disabled, so camera features are not computed even though the current batch structure still carries image and calibration fields.

## AGHRI Spatial Configuration

The shared AGHRI settings are defined in [`configs/aghri/default.yaml`](../configs/aghri/default.yaml).

The main spatial settings are:

| Setting | AGHRI value |
|---|---|
| Point-cloud range | `[-24, -24, -5, 24, 24, 3]` m |
| Voxel size | `[0.075, 0.075, 0.2]` m |
| Network image size | `256 × 704` |
| Camera BEV x/y bounds | `[-24, 24]` m |
| Camera BEV step | `0.3` m |
| LSS depth range | `[0.5, 24.5]` m |
| LSS depth step | `0.5` m |

The supported model modes use these settings differently:

- **Camera-only** uses the camera BEV representation and a camera-only decoder path.
- **LiDAR-only** uses voxelized LiDAR features with the camera encoder disabled.
- **Fusion** combines camera and LiDAR BEV features through `ConvFuser` before the shared decoder and TransFusion head.

## Pinhole and Distortion-Aware LSS

The AGHRI camera path supports both **standard pinhole LSS** and **distortion-aware LSS**.

[`AGHRILSSTransform`](../mmdet3d/models/vtransforms/aghri_lss.py) uses the inherited pinhole LSS geometry when:

```text
distortion_aware: false
```

In the pinhole path, an image pixel is converted to a camera ray using the intrinsic matrix and then transformed into the LiDAR/BEV coordinate system.

When:

```text
distortion_aware: true
```

the transform first reverses the image augmentation and then uses the camera model and distortion coefficients stored in the AGHRI PKLs to recover the camera ray before applying the LSS depth bins.

The camera-model calculations are implemented in [`camera_geometry.py`](../mmdet3d/models/vtransforms/camera_geometry.py).

### ZED RGB

The ZED camera uses the source `rational_polynomial` distortion description. The distortion-aware path iteratively inverts the eight-parameter Brown–Conrady rational model before forming the camera ray.

### Fisheye Cameras

The front, left, and right fisheye cameras use the `equidistant` model. The distortion-aware path iteratively inverts:

```text
θ_d = θ(1 + k₁θ² + k₂θ⁴ + k₃θ⁶ + k₄θ⁸)
```

and then uses:

```text
r = tan(θ)
```

to construct the camera ray.

The resulting ray follows the LSS camera-forward depth convention.

## How Camera Metadata Reaches LSS

The camera entries in the AGHRI PKLs provide:

- camera intrinsics;
- distortion model;
- distortion coefficients;
- camera model information; and
- camera-to-LiDAR transforms.

The dataset adapter reads this information, the data pipeline carries it in fixed-shape tensors, and `BEVFusion.forward` passes it to the AGHRI LSS transform.

## Visualization

[`visualize_aghri.py`](../tools/visualize_aghri.py) provides a common visualization path for AGHRI ground truth and model predictions.

For fisheye views, it uses the camera model and calibration information from the supplied PKL to project 3D information onto the raw camera images, together with a LiDAR BEV view.

See the main [README](../README.md) for an example command.
