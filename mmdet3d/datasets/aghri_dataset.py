"""AGHRI dataset adapter and one-class detection evaluator.

The input PKLs follow the legacy MMDetection3D nuScenes information schema,
but AGHRI is not evaluated through the official nuScenes database or splits.
"""

import json
import math
import os
from collections import defaultdict

import numpy as np
from mmdet.datasets import DATASETS

from .custom_3d import Custom3DDataset
from .nuscenes_dataset import NuScenesDataset


@DATASETS.register_module()
class AGHRIDataset(NuScenesDataset):
    """Read the converted AGHRI PKLs without modifying their contents."""

    CLASSES = ("human",)

    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        map_classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        use_valid_flag=False,
        expected_camera_order=None,
        center_distance_thresholds=(0.5, 1.0, 2.0, 4.0),
        tp_match_threshold=2.0,
    ):
        object_classes = object_classes or self.CLASSES
        if tuple(object_classes) != self.CLASSES:
            raise ValueError(
                f"AGHRI detection classes must be {self.CLASSES}, "
                f"got {tuple(object_classes)}"
            )
        if dataset_root is None:
            raise ValueError("dataset_root is required for relative AGHRI payload paths")

        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        self.map_classes = map_classes
        self.with_velocity = with_velocity
        self.expected_camera_order = tuple(expected_camera_order or ())
        self.center_distance_thresholds = tuple(
            float(value) for value in center_distance_thresholds
        )
        self.tp_match_threshold = float(tp_match_threshold)

        # Deliberately bypass NuScenesDataset.__init__: it creates an official
        # nuScenes evaluation config whose classes/splits do not contain AGHRI.
        Custom3DDataset.__init__(
            self,
            dataset_root=dataset_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=object_classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )

        if self.version != "v1.0-aghri":
            raise ValueError(
                f"Expected PKL metadata version 'v1.0-aghri', got {self.version!r}"
            )
        if self.modality is None:
            self.modality = dict(
                use_camera=True,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

    def _resolve_payload_path(self, path):
        path = os.fspath(path)
        if os.path.isabs(path):
            return os.path.normpath(path)
        return os.path.normpath(os.path.join(self.dataset_root, path))

    def _validate_record(self, info, index):
        camera_order = tuple(info.get("cams", {}).keys())
        if self.expected_camera_order and camera_order != self.expected_camera_order:
            raise ValueError(
                f"AGHRI camera order mismatch at index {index}: expected "
                f"{self.expected_camera_order}, got {camera_order}"
            )
        if info.get("sweeps"):
            raise ValueError(
                f"AGHRI index {index} unexpectedly contains LiDAR sweeps; "
                "the baseline is keyframe-only"
            )
        names = set(np.asarray(info.get("gt_names", []), dtype=object).tolist())
        unknown = names.difference(self.CLASSES)
        if unknown:
            raise ValueError(
                f"AGHRI index {index} contains unexpected detection classes: "
                f"{sorted(unknown)}"
            )

    def get_data_info(self, index):
        info = self.data_infos[index]
        self._validate_record(info, index)
        data = super().get_data_info(index)
        data["lidar_path"] = self._resolve_payload_path(data["lidar_path"])
        if "image_paths" in data:
            data["image_paths"] = [
                self._resolve_payload_path(path) for path in data["image_paths"]
            ]
        return data

    @staticmethod
    def _to_numpy(value):
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    def _ground_truth_for_index(self, index):
        info = self.data_infos[index]
        boxes = np.asarray(info["gt_boxes"], dtype=np.float64)
        names = np.asarray(info["gt_names"], dtype=object)
        if self.use_valid_flag and "valid_flag" in info:
            keep = np.asarray(info["valid_flag"], dtype=bool)
        else:
            keep = np.asarray(info["num_lidar_pts"]) > 0
        keep &= names == "human"
        boxes = boxes[keep]

        velocity = np.asarray(
            info.get("gt_velocity", np.full((len(names), 2), np.nan)),
            dtype=np.float64,
        )
        velocity = velocity.reshape(-1, 2)[keep]
        return boxes, velocity

    def _predictions_for_index(self, result, index):
        if "pts_bbox" in result:
            result = result["pts_bbox"]
        required = {"boxes_3d", "scores_3d", "labels_3d"}
        missing = required.difference(result)
        if missing:
            raise KeyError(
                f"AGHRI result {index} is missing keys {sorted(missing)}"
            )

        boxes = self._to_numpy(result["boxes_3d"].tensor).astype(np.float64)
        scores = self._to_numpy(result["scores_3d"]).astype(np.float64)
        labels = self._to_numpy(result["labels_3d"]).astype(np.int64)
        if len(boxes) != len(scores) or len(scores) != len(labels):
            raise ValueError(f"Inconsistent prediction lengths at index {index}")
        if np.any(labels != 0):
            raise ValueError(
                f"One-class AGHRI model emitted nonzero labels at index {index}"
            )
        return boxes, scores

    @staticmethod
    def _scale_error(pred_box, gt_box):
        pred_size = np.maximum(pred_box[3:6], 0.0)
        gt_size = np.maximum(gt_box[3:6], 0.0)
        intersection = float(np.prod(np.minimum(pred_size, gt_size)))
        union = float(np.prod(pred_size) + np.prod(gt_size) - intersection)
        return 1.0 if union <= 0.0 else 1.0 - intersection / union

    @staticmethod
    def _yaw_error(pred_box, gt_box):
        delta = float(pred_box[6] - gt_box[6])
        return abs((delta + math.pi) % (2.0 * math.pi) - math.pi)

    @staticmethod
    def _interpolated_ap(recall, precision):
        if len(recall) == 0:
            return 0.0
        recall_grid = np.linspace(0.0, 1.0, 101)
        sampled = np.asarray(
            [
                np.max(precision[recall >= level])
                if np.any(recall >= level)
                else 0.0
                for level in recall_grid
            ]
        )
        # Match the nuScenes minimum-recall/minimum-precision convention.
        sampled = sampled[recall_grid >= 0.1]
        return float(np.mean(np.clip((sampled - 0.1) / 0.9, 0.0, 1.0)))

    def _match(self, predictions, ground_truth, threshold):
        matched = defaultdict(set)
        rows = []
        num_gt = sum(len(item[0]) for item in ground_truth)

        ordered = []
        for sample_index, (boxes, scores) in enumerate(predictions):
            for pred_index, score in enumerate(scores):
                ordered.append((float(score), sample_index, pred_index))
        ordered.sort(key=lambda item: item[0], reverse=True)

        tp = []
        fp = []
        for score, sample_index, pred_index in ordered:
            pred_box = predictions[sample_index][0][pred_index]
            gt_boxes, gt_velocity = ground_truth[sample_index]
            available = [
                gt_index
                for gt_index in range(len(gt_boxes))
                if gt_index not in matched[sample_index]
            ]
            best_index = None
            best_distance = float("inf")
            for gt_index in available:
                distance = float(
                    np.linalg.norm(pred_box[:2] - gt_boxes[gt_index, :2])
                )
                if distance < best_distance:
                    best_distance = distance
                    best_index = gt_index

            is_match = best_index is not None and best_distance <= threshold
            tp.append(1.0 if is_match else 0.0)
            fp.append(0.0 if is_match else 1.0)
            if is_match:
                matched[sample_index].add(best_index)
                rows.append(
                    (
                        score,
                        pred_box,
                        gt_boxes[best_index],
                        gt_velocity[best_index],
                        best_distance,
                    )
                )

        if not ordered or num_gt == 0:
            return 0.0, rows, np.asarray([]), np.asarray([])
        tp = np.cumsum(np.asarray(tp))
        fp = np.cumsum(np.asarray(fp))
        recall = tp / float(num_gt)
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        return self._interpolated_ap(recall, precision), rows, recall, precision

    def evaluate(self, results, metric="bbox", jsonfile_prefix=None, **kwargs):
        if len(results) != len(self):
            raise ValueError(
                f"Expected {len(self)} AGHRI results, received {len(results)}"
            )

        predictions = [
            self._predictions_for_index(result, index)
            for index, result in enumerate(results)
        ]
        ground_truth = [
            self._ground_truth_for_index(index) for index in range(len(self))
        ]

        metrics = {}
        aps = []
        for threshold in self.center_distance_thresholds:
            ap, _, _, _ = self._match(predictions, ground_truth, threshold)
            metrics[f"human_AP_dist_{threshold:g}"] = ap
            aps.append(ap)
        metrics["aghri_mAP"] = float(np.mean(aps)) if aps else 0.0

        _, matches, recall, precision = self._match(
            predictions, ground_truth, self.tp_match_threshold
        )
        if matches:
            metrics["human_ATE"] = float(np.mean([row[4] for row in matches]))
            metrics["human_ASE"] = float(
                np.mean([self._scale_error(row[1], row[2]) for row in matches])
            )
            metrics["human_AOE_secondary"] = float(
                np.mean([self._yaw_error(row[1], row[2]) for row in matches])
            )

            velocity_errors = []
            for _, pred_box, _, gt_velocity, _ in matches:
                if len(pred_box) >= 9 and np.isfinite(gt_velocity).all():
                    velocity_errors.append(
                        float(np.linalg.norm(pred_box[7:9] - gt_velocity))
                    )
            metrics["human_AVE_secondary"] = (
                float(np.mean(velocity_errors)) if velocity_errors else float("nan")
            )
            metrics["human_velocity_match_coverage"] = (
                float(len(velocity_errors)) / float(len(matches))
            )
        else:
            metrics.update(
                human_ATE=float("nan"),
                human_ASE=float("nan"),
                human_AOE_secondary=float("nan"),
                human_AVE_secondary=float("nan"),
                human_velocity_match_coverage=0.0,
            )

        if len(recall):
            f1 = 2.0 * precision * recall / np.maximum(
                precision + recall, np.finfo(np.float64).eps
            )
            best = int(np.argmax(f1))
            metrics["human_best_F1_diagnostic"] = float(f1[best])
            metrics["human_precision_at_best_F1"] = float(precision[best])
            metrics["human_recall_at_best_F1"] = float(recall[best])
        else:
            metrics["human_best_F1_diagnostic"] = 0.0
            metrics["human_precision_at_best_F1"] = 0.0
            metrics["human_recall_at_best_F1"] = 0.0

        metrics["human_num_gt"] = float(
            sum(len(item[0]) for item in ground_truth)
        )
        metrics["human_num_predictions"] = float(
            sum(len(item[0]) for item in predictions)
        )

        print("\nAGHRI one-class detection evaluation")
        print(f"mAP:  {metrics['aghri_mAP']:.4f}")
        print(f"mATE: {metrics['human_ATE']:.4f}")
        print(f"mASE: {metrics['human_ASE']:.4f}")
        print(f"mAOE (secondary): {metrics['human_AOE_secondary']:.4f}")
        print(f"mAVE (secondary): {metrics['human_AVE_secondary']:.4f}")
        print("mAAE: N/A (AGHRI has no detection attributes)")
        print("NDS:  N/A (not an official nuScenes evaluation)")
        print("\nPer-class results:")
        print("Object Class\tAP\tATE\tASE\tAOE\tAVE")
        print(
            "human\t"
            f"{metrics['aghri_mAP']:.3f}\t"
            f"{metrics['human_ATE']:.3f}\t"
            f"{metrics['human_ASE']:.3f}\t"
            f"{metrics['human_AOE_secondary']:.3f}\t"
            f"{metrics['human_AVE_secondary']:.3f}"
        )

        if jsonfile_prefix:
            output_dir = os.path.dirname(jsonfile_prefix) or "."
            os.makedirs(output_dir, exist_ok=True)
            with open(
                os.path.join(output_dir, "aghri_metrics_summary.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(metrics, handle, indent=2, allow_nan=True)
        return metrics
