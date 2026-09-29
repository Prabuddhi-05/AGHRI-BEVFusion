"""Portable AGHRI experiment selector for the existing TorchPack entrypoints.

This launcher selects immutable public configs and supplies local data paths.
It does not implement training, evaluation, checkpoint transfer, or geometry.
"""

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple


REPO = Path(__file__).resolve().parents[1]
PUBLIC_CONFIGS = Path("configs/aghri")
CAMERA_PRETRAINED = REPO / "pretrained/swint-nuimages-pretrained.pth"
LIDAR_PRETRAINED = REPO / "pretrained/lidar-only-det.pth"


@dataclass(frozen=True)
class Experiment:
    name: str
    modality: str
    geometry: Optional[str]
    camera_init: Optional[str]
    lidar_init: Optional[str]
    config: Path
    pkl_variant: str


def experiment(name, modality, geometry, camera_init, lidar_init, config, variant):
    return Experiment(
        name, modality, geometry, camera_init, lidar_init, PUBLIC_CONFIGS / config,
        variant,
    )


EXPERIMENTS: Dict[str, Experiment] = {
    spec.name: spec
    for spec in (
        experiment("C1-Pinhole", "camera", "pinhole", "scratch", None,
                   "camera/camera_only_scratch_pinhole.yaml", "fisheye_lidar"),
        experiment("C1-DA", "camera", "distortion-aware", "scratch", None,
                   "camera/camera_only_scratch_distortion_aware.yaml", "fisheye_lidar"),
        experiment("C2-Pinhole", "camera", "pinhole", "nuimages", None,
                   "camera/camera_only_nuimages_pinhole.yaml", "fisheye_lidar"),
        experiment("C2-DA", "camera", "distortion-aware", "nuimages", None,
                   "camera/camera_only_nuimages_distortion_aware.yaml", "fisheye_lidar"),
        experiment("L1", "lidar", None, None, "scratch",
                   "lidar/lidar_only_scratch.yaml", "zed_lidar"),
        experiment("L2", "lidar", None, None, "nuscenes",
                   "lidar/lidar_only_nuscenes_pretrained.yaml", "zed_lidar"),
        experiment("F1", "fusion", "pinhole", "nuimages", "nuscenes",
                   "fusion/fusion_pinhole.yaml", "fisheye_lidar"),
        experiment("F2", "fusion", "distortion-aware", "nuimages", "nuscenes",
                   "fusion/fusion_distortion_aware.yaml", "fisheye_lidar"),
        experiment("D-F1", "fusion", "distortion-aware", "aghri-scratch",
                   "aghri-scratch", "fusion/fusion_distortion_aware_df.yaml",
                   "fisheye_lidar"),
        experiment("D-F2", "fusion", "distortion-aware", "aghri-scratch",
                   "aghri-nuscenes", "fusion/fusion_distortion_aware_df.yaml",
                   "fisheye_lidar"),
        experiment("D-F3", "fusion", "distortion-aware", "aghri-nuimages",
                   "aghri-scratch", "fusion/fusion_distortion_aware_df.yaml",
                   "fisheye_lidar"),
        experiment("D-F4", "fusion", "distortion-aware", "aghri-nuimages",
                   "aghri-nuscenes", "fusion/fusion_distortion_aware_df.yaml",
                   "fisheye_lidar"),
    )
}
BY_SELECTION: Dict[Tuple[Optional[str], ...], Experiment] = {
    (item.modality, item.geometry, item.camera_init, item.lidar_init): item
    for item in EXPERIMENTS.values()
}


def absolute(value: str) -> Path:
    return Path(value).expanduser().resolve()


def pkl_paths(pkl_root: Path, variant: str) -> Dict[str, Path]:
    prefix = "aghri_fisheye_lidar" if variant == "fisheye_lidar" else "aghri_zed_lidar"
    directory = pkl_root / variant / "pkls"
    return {
        split: directory / f"{prefix}_infos_{split}.pkl"
        for split in ("train", "val", "test")
    }


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise ValueError(f"{description} not found: {path}")


def require_dataset(root: Path) -> None:
    for child in ("samples", "v1.0-aghri"):
        if not (root / child).is_dir():
            raise ValueError(f"dataset root lacks {child}/: {root}")


def resolve_train_experiment(args: argparse.Namespace) -> Experiment:
    selected = {
        "modality": args.modality,
        "geometry": args.geometry,
        "camera_init": args.camera_init,
        "lidar_init": args.lidar_init,
    }
    if args.experiment:
        spec = EXPERIMENTS[args.experiment]
        for field, value in selected.items():
            if value is not None and value != getattr(spec, field):
                raise ValueError(
                    f"{args.experiment} requires {field}={getattr(spec, field)!r}, "
                    f"not {value!r}"
                )
        return spec
    key = (args.modality, args.geometry, args.camera_init, args.lidar_init)
    if key not in BY_SELECTION:
        raise ValueError(
            "Unsupported modality/geometry/initialization combination: "
            f"{key}. LiDAR-only has no geometry or camera initialization. "
            "Use --experiment to see an exact supported mapping."
        )
    return BY_SELECTION[key]


def init_paths(
    args: argparse.Namespace, spec: Experiment
) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    """Return external camera/LiDAR and AGHRI camera/LiDAR paths, respectively."""
    camera_external = None
    lidar_external = None
    camera_branch = None
    lidar_branch = None

    if spec.camera_init == "nuimages":
        camera_external = absolute(
            args.camera_pretrained_checkpoint or str(CAMERA_PRETRAINED)
        )
    elif args.camera_pretrained_checkpoint:
        raise ValueError("External camera weights are not used by this experiment")

    if spec.lidar_init == "nuscenes":
        lidar_external = absolute(
            args.lidar_pretrained_checkpoint or str(LIDAR_PRETRAINED)
        )
    elif args.lidar_pretrained_checkpoint:
        raise ValueError("External LiDAR weights are not used by this experiment")

    if spec.camera_init in ("aghri-scratch", "aghri-nuimages"):
        if not args.camera_checkpoint:
            raise ValueError(f"{spec.name} requires --camera-checkpoint")
        camera_branch = absolute(args.camera_checkpoint)
    elif args.camera_checkpoint:
        raise ValueError("--camera-checkpoint is only for D-F fusion")

    if spec.lidar_init in ("aghri-scratch", "aghri-nuscenes"):
        if not args.lidar_checkpoint:
            raise ValueError(f"{spec.name} requires --lidar-checkpoint")
        lidar_branch = absolute(args.lidar_checkpoint)
    elif args.lidar_checkpoint:
        raise ValueError("--lidar-checkpoint is only for D-F fusion")

    return camera_external, lidar_external, camera_branch, lidar_branch


def show_command(command: Sequence[str]) -> None:
    print("  " + shlex.join(command))


def run_train(args: argparse.Namespace) -> None:
    spec = resolve_train_experiment(args)
    config = REPO / spec.config
    require_file(config, "public config")
    dataset_root = absolute(args.dataset_root)
    pkl_root = absolute(args.pkl_root)
    run_dir = absolute(args.run_dir)
    pkls = pkl_paths(pkl_root, spec.pkl_variant)
    camera_external, lidar_external, camera_branch, lidar_branch = init_paths(
        args, spec
    )

    branch_init = run_dir / "branch_encoder_initialization.pth"
    commands = []
    if camera_branch is not None:
        commands.append([
            sys.executable, "tools/build_aghri_df_branch_init.py",
            "--camera", str(camera_branch),
            "--lidar", str(lidar_branch),
            "--output", str(branch_init),
        ])

    train_command = [
        "torchpack", "dist-run", "-np", str(args.gpus), sys.executable,
        "tools/train_aghri.py", str(spec.config),
        "--run-dir", str(run_dir),
        # These top-level TorchPack overrides feed the inherited ${...}
        # references in configs/aghri/default.yaml.
        "--dataset_root", str(dataset_root),
        "--train_ann_file", str(pkls["train"]),
        "--val_ann_file", str(pkls["val"]),
        "--test_ann_file", str(pkls["test"]),
    ]
    if camera_external is not None:
        train_command.extend([
            "--model.encoders.camera.backbone.init_cfg.checkpoint",
            str(camera_external),
        ])
    if lidar_external is not None:
        train_command.extend([
            "--aghri-pretrained-checkpoint", str(lidar_external)
        ])
    if camera_branch is not None:
        train_command.extend([
            "--aghri-pretrained-checkpoint", str(branch_init)
        ])
    commands.append(train_command)

    print(f"Experiment: {spec.name}")
    print(f"Action: train | modality: {spec.modality} | geometry: {spec.geometry or 'N/A'}")
    print(f"Config: {spec.config}")
    print(f"Dataset root: {dataset_root}")
    print(f"PKL root: {pkl_root}")
    for split, path in pkls.items():
        print(f"{split.capitalize()} PKL: {path}")
    print(f"Camera initialization: {spec.camera_init or 'disabled'}")
    print(f"LiDAR initialization: {spec.lidar_init or 'disabled'}")
    print(f"External camera checkpoint: {camera_external or 'none'}")
    print(f"External LiDAR checkpoint: {lidar_external or 'none'}")
    print(f"AGHRI camera checkpoint: {camera_branch or 'none'}")
    print(f"AGHRI LiDAR checkpoint: {lidar_branch or 'none'}")
    print(f"Run directory: {run_dir}")
    if camera_branch is not None:
        print(f"Generated branch initialization: {branch_init}")
    print("Commands:")
    for command in commands:
        show_command(command)
    if args.dry_run:
        print("Dry run only: input-file existence not checked; no command executed.")
        return

    require_dataset(dataset_root)
    for split, path in pkls.items():
        require_file(path, f"{split} PKL")
    for description, path in (
        ("external camera checkpoint", camera_external),
        ("external LiDAR checkpoint", lidar_external),
        ("AGHRI camera checkpoint", camera_branch),
        ("AGHRI LiDAR checkpoint", lidar_branch),
    ):
        if path is not None:
            require_file(path, description)
    if camera_branch is not None and branch_init.exists():
        raise ValueError(f"Refusing to overwrite branch initialization: {branch_init}")

    run_dir.mkdir(parents=True, exist_ok=True)
    for command in commands:
        subprocess.run(command, cwd=REPO, check=True)


def run_eval(args: argparse.Namespace) -> None:
    spec = EXPERIMENTS[args.experiment]
    config = REPO / spec.config
    require_file(config, "public config")
    dataset_root = absolute(args.dataset_root)
    pkl_root = absolute(args.pkl_root)
    checkpoint = absolute(args.checkpoint)
    ann_file = pkl_paths(pkl_root, spec.pkl_variant)[args.split]

    # Swin init_weights runs while the model is built, before the full
    # evaluation checkpoint is loaded. Supply its external checkpoint for
    # C2/F1/F2 rather than silently assuming it is already on this machine.
    camera_external = None
    if spec.camera_init == "nuimages":
        camera_external = absolute(
            args.camera_pretrained_checkpoint or str(CAMERA_PRETRAINED)
        )
    elif args.camera_pretrained_checkpoint:
        raise ValueError("This evaluation config does not use external Swin init")

    command = [
        "torchpack", "dist-run", "-np", str(args.gpus), sys.executable,
        "tools/test.py", str(spec.config), str(checkpoint), "--eval", "bbox",
        "--cfg-options",
        f"data.test.dataset_root={dataset_root}",
        f"data.test.ann_file={ann_file}",
    ]
    if camera_external is not None:
        command.append(
            "model.encoders.camera.backbone.init_cfg.checkpoint="
            + str(camera_external)
        )

    print(f"Experiment: {spec.name}")
    print(f"Action: eval | split: {args.split} | modality: {spec.modality}")
    print(f"Geometry: {spec.geometry or 'N/A'} | config: {spec.config}")
    print(f"Dataset root: {dataset_root}")
    print(f"PKL root: {pkl_root}")
    print(f"Evaluation PKL: {ann_file}")
    print(f"Camera initialization: {spec.camera_init or 'disabled'}")
    print(f"LiDAR initialization: {spec.lidar_init or 'disabled'}")
    print(f"External camera checkpoint used during model build: {camera_external or 'none'}")
    print(f"Model checkpoint: {checkpoint}")
    print("Run directory: N/A (evaluation)")
    print("Command:")
    show_command(command)
    if args.dry_run:
        print("Dry run only: input-file existence not checked; no command executed.")
        return

    require_dataset(dataset_root)
    require_file(ann_file, "evaluation PKL")
    require_file(checkpoint, "model checkpoint")
    if camera_external is not None:
        require_file(camera_external, "external camera checkpoint")
    subprocess.run(command, cwd=REPO, check=True)


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("GPU count must be positive")
    return result


def parser_for_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)

    train = actions.add_parser("train", help="launch one supported AGHRI experiment")
    train.add_argument("--experiment", choices=tuple(EXPERIMENTS))
    train.add_argument("--modality", choices=("camera", "lidar", "fusion"))
    train.add_argument("--geometry", choices=("pinhole", "distortion-aware"))
    train.add_argument("--camera-init", choices=(
        "scratch", "nuimages", "aghri-scratch", "aghri-nuimages"
    ))
    train.add_argument("--lidar-init", choices=(
        "scratch", "nuscenes", "aghri-scratch", "aghri-nuscenes"
    ))
    train.add_argument("--camera-checkpoint")
    train.add_argument("--lidar-checkpoint")
    train.add_argument("--camera-pretrained-checkpoint")
    train.add_argument("--lidar-pretrained-checkpoint")
    train.add_argument("--dataset-root", required=True)
    train.add_argument("--pkl-root", required=True)
    train.add_argument("--gpus", type=positive_int, default=1)
    train.add_argument("--run-dir", required=True)
    train.add_argument("--dry-run", action="store_true")
    train.set_defaults(handler=run_train)

    evaluate = actions.add_parser("eval", help="evaluate a full AGHRI checkpoint")
    evaluate.add_argument("--experiment", choices=tuple(EXPERIMENTS), required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=("val", "test"), required=True)
    evaluate.add_argument("--dataset-root", required=True)
    evaluate.add_argument("--pkl-root", required=True)
    evaluate.add_argument("--camera-pretrained-checkpoint")
    evaluate.add_argument("--gpus", type=positive_int, default=1)
    evaluate.add_argument("--dry-run", action="store_true")
    evaluate.set_defaults(handler=run_eval)
    return parser


def main() -> None:
    parser = parser_for_cli()
    args = parser.parse_args()
    try:
        args.handler(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
