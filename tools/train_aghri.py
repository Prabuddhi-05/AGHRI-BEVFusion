import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
from mmcv import Config
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval


def is_class_dependent_key(key):
    """Identify classification tensors that must be reinitialized for one class."""
    return (
        "heads.object.heatmap_head.1" in key
        or "heads.object.class_encoding" in key
        or (
            "heads.object.prediction_heads" in key
            and ".heatmap." in key
        )
    )


def load_compatible_pretrained_weights(model, checkpoint_path, report_path, logger):
    """Load only shape-compatible, non-class-specific tensors.

    ``strict=False`` does not protect against equal-name tensors with different
    shapes. AGHRI has one class while the source nuScenes detector has ten, so
    filtering is performed before ``load_state_dict``.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint.get("state_dict", checkpoint)
    target_state = model.state_dict()

    compatible = {}
    excluded_class_dependent = []
    shape_mismatches = []
    unexpected = []

    for original_key, value in source_state.items():
        key = original_key[7:] if original_key.startswith("module.") else original_key
        if is_class_dependent_key(key):
            excluded_class_dependent.append(key)
            continue
        if key not in target_state:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "source_shape": list(value.shape),
                    "target_shape": list(target_state[key].shape),
                }
            )
            continue
        compatible[key] = value

    load_result = model.load_state_dict(compatible, strict=False)
    report = {
        "source_checkpoint": os.path.abspath(checkpoint_path),
        "loaded_tensor_count": len(compatible),
        "loaded_keys": sorted(compatible),
        "excluded_class_dependent_keys": sorted(excluded_class_dependent),
        "shape_mismatches": shape_mismatches,
        "unexpected_source_keys": sorted(unexpected),
        "missing_target_keys": sorted(load_result.missing_keys),
        "unexpected_after_load": sorted(load_result.unexpected_keys),
    }
    if dist.rank() == 0:
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    logger.info(
        "AGHRI compatible checkpoint load: %d tensors loaded, %d class-dependent "
        "excluded, %d shape mismatches, %d unexpected source keys. Report: %s",
        len(compatible),
        len(excluded_class_dependent),
        len(shape_mismatches),
        len(unexpected),
        report_path,
    )


def main():
    dist.init()
    rank = dist.rank()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    parser.add_argument(
        "--aghri-pretrained-checkpoint",
        metavar="FILE",
        help="load only compatible non-class-specific tensors from this checkpoint",
    )
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

    # The cluster baseline does not require TensorBoard, and the old image may
    # otherwise import an incompatible tensorboard/distutils combination.
    if cfg.get("log_config"):
        cfg.log_config["hooks"] = [
            hook
            for hook in cfg.log_config.get("hooks", [])
            if hook.get("type") != "TensorboardLoggerHook"
        ]

    # dump config
    if rank == 0:
        cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))

    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}_rank{rank}.log")
    logger = get_root_logger(log_file=log_file)

    # log some basic info
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    if cfg.seed is not None:
        logger.info(
            f"Set random seed to {cfg.seed}, "
            f"deterministic mode: {cfg.deterministic}"
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    datasets = [build_dataset(cfg.data.train)]

    model = build_model(cfg.model)
    model.init_weights()
    if args.aghri_pretrained_checkpoint:
        report_path = os.path.join(cfg.run_dir, "checkpoint_load_report.json")
        load_compatible_pretrained_weights(
            model,
            args.aghri_pretrained_checkpoint,
            report_path,
            logger,
        )
        # The compatible tensors are already loaded. Prevent the runner from
        # trying to load the unfiltered ten-class checkpoint a second time.
        cfg.load_from = None
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    logger.info(f"Model:\n{model}")
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    #torch.cuda.empty_cache()
    print(">>> About to train model")
    train_model(
        model,
        datasets,
        cfg,
        distributed=True,
        #distributed=distributed,  # instead of hardcoded True
        validate=True,
        timestamp=timestamp,
    )
    print(">>> Training started")


if __name__ == "__main__":
    main()
