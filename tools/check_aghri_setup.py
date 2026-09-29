"""One-sample AGHRI configuration, pipeline, and forward-pass preflight."""

import argparse

import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="validate config and one dataset sample without constructing the GPU model",
    )
    args = parser.parse_args()

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    dataset = build_dataset(cfg.data.train)
    if not dataset:
        raise RuntimeError("AGHRI training dataset is empty")
    sample = dataset[0]
    required = {
        "img",
        "points",
        "gt_bboxes_3d",
        "gt_labels_3d",
        "camera_intrinsics",
        "camera2lidar",
        "lidar_aug_matrix",
        "metas",
    }
    missing = required.difference(sample)
    if missing:
        raise KeyError(f"AGHRI sample is missing keys: {sorted(missing)}")

    expected_views = len(cfg.expected_camera_order)
    distortion_aware = bool(
        cfg.model.encoders.camera.vtransform.get("distortion_aware", False)
    )
    if distortion_aware:
        distortion_keys = {
            "camera_distortion_models",
            "camera_distortion_coefficients",
        }
        missing_distortion = distortion_keys.difference(sample)
        if missing_distortion:
            raise KeyError(
                "Distortion-aware AGHRI sample is missing keys: "
                f"{sorted(missing_distortion)}"
            )

        model_codes = sample["camera_distortion_models"].data
        coefficients = sample["camera_distortion_coefficients"].data
        if model_codes.shape != (expected_views,):
            raise ValueError(
                "Expected one distortion-model code per camera, got "
                f"{tuple(model_codes.shape)}"
            )
        if coefficients.shape != (expected_views, 8):
            raise ValueError(
                "Expected eight padded distortion coefficients per camera, got "
                f"{tuple(coefficients.shape)}"
            )
        if not torch.isfinite(coefficients).all():
            raise FloatingPointError("Camera distortion coefficients are non-finite")

    image_tensor = sample["img"].data
    if image_tensor.ndim != 4 or image_tensor.shape[1] != 3:
        raise ValueError(
            f"Expected camera tensor [N, 3, H, W], got {tuple(image_tensor.shape)}"
        )
    if image_tensor.shape[0] != expected_views:
        raise ValueError(
            f"Expected {expected_views} cameras, got {image_tensor.shape[0]}"
        )

    points = sample["points"].data
    if points.ndim != 2 or points.shape[1] != 5:
        raise ValueError(f"Expected five-value LiDAR records, got {tuple(points.shape)}")

    if args.skip_forward:
        print("[PREFLIGHT] AGHRI dataset records:", len(dataset))
        print("[PREFLIGHT] Camera tensor:", tuple(image_tensor.shape))
        print("[PREFLIGHT] LiDAR tensor:", tuple(points.shape))
        print("[PREFLIGHT] Distortion-aware geometry:", distortion_aware)
        if distortion_aware:
            print("[PREFLIGHT] Distortion model codes:", model_codes.tolist())
        print("[PREFLIGHT] DATA PIPELINE PASS (GPU forward skipped)")
        return

    model = build_model(cfg.model)
    model.init_weights()
    model.cuda().train()
    batch = scatter(collate([sample], samples_per_gpu=1), [torch.cuda.current_device()])[0]
    # Do not use torch.no_grad() here. This BEVFusion fork classifies returned
    # values as ``loss/...`` only when the tensors require gradients; under
    # no_grad(), genuine head losses are exposed as ``stats/...`` instead.
    outputs = model(**batch)

    losses = {
        key: float(value.detach().cpu())
        for key, value in outputs.items()
        if key.startswith("loss/") and torch.is_tensor(value)
    }
    if not losses:
        raise RuntimeError("AGHRI forward pass returned no training losses")
    if not all(torch.isfinite(torch.tensor(value)) for value in losses.values()):
        raise FloatingPointError(f"Non-finite AGHRI preflight loss: {losses}")

    print("[PREFLIGHT] AGHRI dataset records:", len(dataset))
    print("[PREFLIGHT] Camera tensor:", tuple(image_tensor.shape))
    print("[PREFLIGHT] LiDAR tensor:", tuple(points.shape))
    print("[PREFLIGHT] Distortion-aware geometry:", distortion_aware)
    if distortion_aware:
        print("[PREFLIGHT] Distortion model codes:", model_codes.tolist())
    print("[PREFLIGHT] Finite forward losses:", losses)
    print("[PREFLIGHT] PASS")


if __name__ == "__main__":
    main()
