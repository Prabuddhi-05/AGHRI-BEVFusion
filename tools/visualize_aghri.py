"""Visualize AGHRI fisheye GT and predictions with true fisheye projection.

The upstream visualizer uses the pinhole ``lidar2image`` matrix for
camera overlays. This script keeps the same dataset/model workflow, but draws
3D boxes on raw fisheye images using the calibration stored in the AGHRI PKL:
``cam_intrinsic``, ``distortion_model``, ``distortion_coefficients`` and the
camera-to-LiDAR extrinsics.
"""

import argparse
import copy
import json
import os
import pickle
from pathlib import Path

import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs
from tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_lidar
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


HUMAN_COLOR_BGR = (230, 0, 0)
FISHEYE_CHANNELS = ("cam_fish_front", "cam_fish_left", "cam_fish_right")
MODEL_CODES = {"pinhole": 0, "rational_polynomial": 1, "equidistant": 2}
BOX_EDGES = (
    (0, 1),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 5),
    (3, 2),
    (3, 7),
    (4, 5),
    (4, 7),
    (2, 6),
    (5, 6),
    (6, 7),
)
BOX_EDGE_SAMPLES = 33


def recursive_eval(obj, globals_=None):
    if globals_ is None:
        globals_ = copy.deepcopy(obj)
    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals_)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            obj[index] = recursive_eval(value, globals_)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals_)
        obj = recursive_eval(obj, globals_)
    return obj


def as_lidar_boxes(boxes, keep=None):
    """Return display-corrected AGHRI boxes in LiDAR coordinates.

    This mirrors ``tools/visualize_aghri.py``: AGHRI PKLs store geometric
    centre-z values, while ``LiDARInstance3DBoxes`` treats z as bottom-centred.
    The subtraction is visualization-only.
    """
    if isinstance(boxes, LiDARInstance3DBoxes):
        tensor = boxes.tensor.detach().cpu()
    elif hasattr(boxes, "tensor"):
        tensor = boxes.tensor.detach().cpu()
    else:
        tensor = torch.as_tensor(boxes, dtype=torch.float32)
    if keep is not None:
        tensor = tensor[torch.as_tensor(keep, dtype=torch.bool)]
    tensor = tensor.clone()
    if tensor.ndim == 2 and tensor.shape[0] > 0:
        tensor[:, 2] -= tensor[:, 5] / 2
    box_dim = tensor.shape[-1] if tensor.ndim == 2 else 9
    return LiDARInstance3DBoxes(tensor, box_dim=box_dim)


def load_infos(path):
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    infos = payload["infos"] if isinstance(payload, dict) else payload
    if not infos:
        raise ValueError(f"No infos were found in {path}")
    return infos


def resolve_path(dataset_root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(dataset_root) / path


def lidar_to_camera(points_lidar, camera):
    rotation = np.asarray(camera["sensor2lidar_rotation"], dtype=np.float64)
    translation = np.asarray(camera["sensor2lidar_translation"], dtype=np.float64)
    return (points_lidar - translation) @ rotation


def project_pinhole(points_camera, intrinsic):
    normalized = points_camera[:, :2] / points_camera[:, 2:3]
    homogeneous = np.column_stack((normalized, np.ones(len(normalized))))
    pixels = homogeneous @ intrinsic.T
    return pixels[:, :2] / pixels[:, 2:3]


def project_distorted(points_camera, intrinsic, distortion, model):
    object_points = points_camera.reshape(-1, 1, 3).astype(np.float64)
    if model == "equidistant":
        pixels, _ = cv2.fisheye.projectPoints(
            object_points,
            np.zeros(3),
            np.zeros(3),
            intrinsic,
            distortion[:4],
        )
    elif model == "rational_polynomial":
        pixels, _ = cv2.projectPoints(
            object_points,
            np.zeros(3),
            np.zeros(3),
            intrinsic,
            distortion[:8],
        )
    elif model == "pinhole":
        pixels = project_pinhole(points_camera, intrinsic).reshape(-1, 1, 2)
    else:
        raise ValueError(f"Unsupported distortion model: {model}")
    return pixels.reshape(-1, 2)


def safe_int_point(point, width, height):
    if not np.isfinite(point).all():
        return None
    limit_x = max(width * 10, 1)
    limit_y = max(height * 10, 1)
    x = int(np.rint(np.clip(point[0], -limit_x, limit_x)))
    y = int(np.rint(np.clip(point[1], -limit_y, limit_y)))
    return x, y


def project_box_edge(start, end, intrinsic, distortion, model):
    """Project a densely sampled 3D edge.

    A fisheye camera does not generally map a straight 3D line to a straight
    image-space line.  Projecting only the two endpoints therefore draws an
    incorrect chord near the image edge.  Sampling the edge before applying
    the camera model preserves the curved fisheye projection.
    """
    amount = np.linspace(0.0, 1.0, BOX_EDGE_SAMPLES, dtype=np.float64)[:, None]
    points = start[None] * (1.0 - amount) + end[None] * amount
    return project_distorted(points, intrinsic, distortion, model)


def draw_projected_edge(canvas, pixels, color, thickness):
    height, width = canvas.shape[:2]
    finite = np.isfinite(pixels).all(axis=1)
    pixels = pixels[finite]
    if len(pixels) < 2:
        return
    limit_x = max(width * 10, 1)
    limit_y = max(height * 10, 1)
    pixels = pixels.copy()
    pixels[:, 0] = np.clip(pixels[:, 0], -limit_x, limit_x)
    pixels[:, 1] = np.clip(pixels[:, 1], -limit_y, limit_y)
    polyline = np.rint(pixels).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [polyline], False, color, thickness, cv2.LINE_AA)


def draw_fisheye_boxes(
    output_path,
    image_path,
    camera,
    bboxes,
    labels,
    classes,
    score_values=None,
    thickness=3,
):
    image = mmcv.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = image.shape[:2]
    canvas = image.copy()

    if bboxes is not None and len(bboxes) > 0:
        corners = bboxes.corners.detach().cpu().numpy()
        labels = np.asarray(labels)
        order = np.argsort(-corners[:, :, 0].mean(axis=1))

        intrinsic = np.asarray(camera["cam_intrinsic"], dtype=np.float64)
        model = camera.get("distortion_model", "pinhole")
        distortion = np.asarray(
            camera.get("distortion_coefficients", []), dtype=np.float64
        ).reshape(-1)

        for box_index in order:
            corners_camera = lidar_to_camera(corners[box_index], camera)
            if not np.all(corners_camera[:, 2] > 1.0e-4):
                continue

            class_name = classes[int(labels[box_index])]
            color = HUMAN_COLOR_BGR
            for start, end in BOX_EDGES:
                edge_pixels = project_box_edge(
                    corners_camera[start],
                    corners_camera[end],
                    intrinsic,
                    distortion,
                    model,
                )
                draw_projected_edge(canvas, edge_pixels, color, thickness)

            if score_values is not None and len(score_values) > box_index:
                pixels = project_distorted(
                    corners_camera, intrinsic, distortion, model
                )
                anchor = safe_int_point(pixels[0], width, height)
                if anchor is not None:
                    cv2.putText(
                        canvas,
                        f"{class_name} {score_values[box_index]:.2f}",
                        anchor,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

    mmcv.mkdir_or_exist(os.path.dirname(output_path))
    mmcv.imwrite(canvas, output_path)


def draw_calibration_diagnostic(
    output_path,
    box_overlay_path,
    camera,
    lidar_points,
    lidar_timestamp,
    channel,
):
    """Overlay projected LiDAR points on the distortion-aware box render."""
    canvas = mmcv.imread(str(box_overlay_path))
    if canvas is None:
        raise FileNotFoundError(box_overlay_path)
    height, width = canvas.shape[:2]

    points_camera = lidar_to_camera(lidar_points[::5, :3], camera)
    positive = points_camera[:, 2] > 0.05
    points_camera = points_camera[positive]
    intrinsic = np.asarray(camera["cam_intrinsic"], dtype=np.float64)
    distortion = np.asarray(
        camera.get("distortion_coefficients", []), dtype=np.float64
    ).reshape(-1)
    model = camera.get("distortion_model", "pinhole")
    pixels = project_distorted(points_camera, intrinsic, distortion, model)
    inside = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    visible_pixels = pixels[inside]
    visible_depths = points_camera[inside, 2]
    order = np.argsort(visible_depths)[::-1]
    for index in order:
        depth = float(visible_depths[index])
        fraction = float(np.clip((depth - 0.5) / 14.5, 0.0, 1.0))
        color = (int(round(255 * (1.0 - fraction))), 80, int(round(255 * fraction)))
        u, v = np.rint(visible_pixels[index]).astype(int)
        cv2.circle(canvas, (int(u), int(v)), 1, color, -1, cv2.LINE_AA)

    camera_timestamp = int(camera.get("timestamp", lidar_timestamp))
    delta_ms = (camera_timestamp - int(lidar_timestamp)) / 1000.0
    label = (
        f"{channel}: LiDAR projection + 3D boxes | "
        f"camera-LiDAR dt={delta_ms:+.1f} ms"
    )
    cv2.rectangle(canvas, (0, 0), (width - 1, 25), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        label,
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    mmcv.mkdir_or_exist(os.path.dirname(output_path))
    mmcv.imwrite(canvas, output_path)
    return {
        "channel": channel,
        "camera_timestamp": camera_timestamp,
        "lidar_timestamp": int(lidar_timestamp),
        "camera_minus_lidar_ms": delta_ms,
        "input_lidar_points": int(len(lidar_points)),
        "positive_depth_points_after_stride": int(len(points_camera)),
        "in_image_points_after_stride": int(np.count_nonzero(inside)),
        "distortion_model": model,
        "diagnostic_image": str(output_path),
    }


def load_lidar_points(dataset_root, lidar_path):
    path = resolve_path(dataset_root, lidar_path)
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 5:
        raise ValueError(f"LiDAR payload is not Nx5 float32: {path}")
    return values.reshape(-1, 5)


def verify_fisheye_projection(infos, dataset_root, output_dir, sample_count):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = np.unique(np.linspace(0, len(infos) - 1, sample_count, dtype=int))
    records = []
    failures = []

    for info_index in selected:
        info = infos[int(info_index)]
        lidar = load_lidar_points(dataset_root, info["lidar_path"])[::10, :3].astype(
            np.float64
        )
        for channel in FISHEYE_CHANNELS:
            if channel not in info["cams"]:
                failures.append(f"{info['token']} missing {channel}")
                continue
            camera = info["cams"][channel]
            model = camera.get("distortion_model")
            if model != "equidistant":
                failures.append(
                    f"{info['token']} {channel} has distortion_model={model!r}"
                )
                continue
            coeffs = np.asarray(camera.get("distortion_coefficients", []))
            if coeffs.shape[0] != 4:
                failures.append(
                    f"{info['token']} {channel} has {coeffs.shape[0]} coefficients"
                )
                continue

            image = mmcv.imread(str(resolve_path(dataset_root, camera["data_path"])))
            if image is None:
                failures.append(f"{info['token']} {channel} image missing")
                continue
            height, width = image.shape[:2]
            camera_points = lidar_to_camera(lidar, camera)
            camera_points = camera_points[camera_points[:, 2] > 1.0e-4]
            if len(camera_points) == 0:
                failures.append(f"{info['token']} {channel} has no front points")
                continue

            intrinsic = np.asarray(camera["cam_intrinsic"], dtype=np.float64)
            distortion = np.asarray(camera["distortion_coefficients"], dtype=np.float64)
            pinhole = project_pinhole(camera_points, intrinsic)
            distorted = project_distorted(camera_points, intrinsic, distortion, model)
            inside = (
                np.isfinite(distorted).all(axis=1)
                & (distorted[:, 0] >= 0)
                & (distorted[:, 0] < width)
                & (distorted[:, 1] >= 0)
                & (distorted[:, 1] < height)
            )
            if not np.any(inside):
                failures.append(f"{info['token']} {channel} projects no points")
                continue

            shift = np.linalg.norm(distorted[inside] - pinhole[inside], axis=1)
            record = {
                "info_index": int(info_index),
                "token": info["token"],
                "channel": channel,
                "distortion_model": model,
                "distortion_coefficients": distortion.tolist(),
                "points_checked": int(inside.sum()),
                "median_pinhole_vs_fisheye_shift_px": float(np.median(shift)),
                "p95_pinhole_vs_fisheye_shift_px": float(np.percentile(shift, 95)),
            }
            records.append(record)

    report = {
        "verification": "fisheye raw-image projection",
        "dataset_root": str(dataset_root),
        "selected_info_indices": selected.tolist(),
        "passed": not failures and bool(records),
        "failures": failures,
        "records": records,
    }
    path = output_dir / "projection_verification.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError(f"Fisheye projection verification failed: {path}")


def main():
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", choices=["gt", "pred"], required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--ann-file", required=True, help="corrected release PKL for the selected split")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--out-dir", default="output/aghri_visualizations")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--verify-first", action="store_true")
    parser.add_argument("--verification-samples", type=int, default=3)
    parser.add_argument(
        "--calibration-diagnostic-samples",
        type=int,
        default=20,
        help="Number of rendered samples that also receive LiDAR projection overlays",
    )
    args, opts = parser.parse_known_args()

    if args.mode == "pred" and not args.checkpoint:
        parser.error("--checkpoint is required for prediction visualization")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.verification_samples <= 0:
        parser.error("--verification-samples must be positive")
    if args.calibration_diagnostic_samples < 0:
        parser.error("--calibration-diagnostic-samples cannot be negative")

    configs.load(args.config, recursive=True)
    configs.update(opts)
    cfg = Config(recursive_eval(configs), filename=args.config)
    cfg.data[args.split].ann_file = args.ann_file
    cfg.data[args.split].dataset_root = args.dataset_root

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    infos = load_infos(cfg.data[args.split].ann_file)
    infos_by_token = {info["token"]: info for info in infos}
    if args.verify_first:
        verify_fisheye_projection(
            infos,
            args.dataset_root,
            os.path.join(args.out_dir, "_projection_verification"),
            args.verification_samples,
        )

    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    model = None
    if args.mode == "pred":
        # Full checkpoint supplies trained camera weights; avoid needing the original
        # Swin binary solely to construct a visualization model.
        camera_cfg = cfg.model.encoders.get("camera")
        if camera_cfg is not None:
            camera_cfg.backbone.init_cfg = None
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        model.eval()

    rendered = 0
    rendered_tokens = []
    calibration_records = []

    for sample_index, data in enumerate(tqdm(dataflow)):
        if args.max_samples is not None and rendered >= args.max_samples:
            break

        metas = data["metas"].data[0][0]
        token = metas["token"]
        if token not in infos_by_token:
            raise KeyError(f"Sample token {token} not found in {cfg.data[args.split].ann_file}")
        info = infos_by_token[token]
        name = f"{sample_index:06d}_{metas['timestamp']}_{token}"

        if args.mode == "pred":
            with torch.inference_mode():
                outputs = model(**data)
            result = outputs[0]
            box_tensor = result["boxes_3d"].tensor.detach().cpu()
            scores = result["scores_3d"].detach().cpu().numpy()
            labels = result["labels_3d"].detach().cpu().numpy()
            keep = np.ones(len(labels), dtype=bool)
            if args.bbox_classes is not None:
                keep &= np.isin(labels, args.bbox_classes)
            if args.bbox_score is not None:
                keep &= scores >= args.bbox_score
            labels = labels[keep]
            kept_scores = scores[keep]
            bboxes = as_lidar_boxes(box_tensor, keep)
        else:
            ann = dataset.get_ann_info(sample_index)
            labels = np.asarray(ann["gt_labels_3d"])
            keep = np.ones(len(labels), dtype=bool)
            if args.bbox_classes is not None:
                keep &= np.isin(labels, args.bbox_classes)
            labels = labels[keep]
            kept_scores = None
            bboxes = as_lidar_boxes(ann["gt_bboxes_3d"], keep)

        lidar = None
        if "points" in data:
            lidar = data["points"].data[0][0].numpy()
        elif "lidar_path" in info:
            try:
                lidar = load_lidar_points(args.dataset_root, info["lidar_path"])
            except FileNotFoundError:
                lidar = None

        for channel in FISHEYE_CHANNELS:
            camera = info["cams"].get(channel)
            if camera is None:
                continue
            box_output = os.path.join(args.out_dir, channel, f"{name}.png")
            draw_fisheye_boxes(
                box_output,
                resolve_path(args.dataset_root, camera["data_path"]),
                camera,
                bboxes,
                labels,
                cfg.object_classes,
                score_values=kept_scores,
            )
            if lidar is not None and rendered < args.calibration_diagnostic_samples:
                record = draw_calibration_diagnostic(
                    os.path.join(
                        args.out_dir,
                        "_calibration_diagnostic",
                        channel,
                        f"{name}.png",
                    ),
                    box_output,
                    camera,
                    lidar,
                    info["timestamp"],
                    channel,
                )
                record.update(
                    {
                        "sample_index": int(sample_index),
                        "sample_token": token,
                        "mode": args.mode,
                    }
                )
                calibration_records.append(record)

        visualize_lidar(
            os.path.join(args.out_dir, "lidar", f"{name}.png"),
            lidar,
            bboxes=bboxes,
            labels=labels,
            xlim=[cfg.point_cloud_range[index] for index in (0, 3)],
            ylim=[cfg.point_cloud_range[index] for index in (1, 4)],
            classes=cfg.object_classes,
            color=(0, 0, 230),
        )

        rendered += 1
        rendered_tokens.append(token)

    manifest = {
        "mode": args.mode,
        "split": args.split,
        "ann_file": cfg.data[args.split].ann_file,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "bbox_score": args.bbox_score,
        "rendered": rendered,
        "tokens": rendered_tokens,
        "camera_projection": {
            "raw_images": True,
            "projection": "OpenCV fisheye/equidistant from AGHRI PKL calibration",
            "fisheye_channels": list(FISHEYE_CHANNELS),
            "uses_pinhole_lidar2image_for_camera_overlay": False,
        },
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_dir, "visualization_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    calibration_report = {
        "purpose": "visual camera-LiDAR calibration diagnostic",
        "applies_calibration_correction": False,
        "note": (
            "These images overlay distortion-aware projected LiDAR points and "
            "projected 3D boxes. They expose alignment residuals but do not apply "
            "an unmeasured pixel offset or alter dataset calibration."
        ),
        "requested_samples": args.calibration_diagnostic_samples,
        "records": calibration_records,
    }
    calibration_dir = Path(args.out_dir, "_calibration_diagnostic")
    calibration_dir.mkdir(parents=True, exist_ok=True)
    calibration_dir.joinpath("calibration_diagnostic.json").write_text(
        json.dumps(calibration_report, indent=2), encoding="utf-8"
    )
    print(f"[AGHRI-FISHEYE-VIZ] Rendered {rendered} {args.split} samples")
    print("[AGHRI-FISHEYE-VIZ] Camera overlays use sampled curved fisheye edges.")
    print("[AGHRI-FISHEYE-VIZ] Raw images are not undistorted or rectified.")
    print(
        "[AGHRI-FISHEYE-VIZ] Calibration diagnostics: "
        f"{calibration_dir / 'calibration_diagnostic.json'}"
    )


if __name__ == "__main__":
    main()
