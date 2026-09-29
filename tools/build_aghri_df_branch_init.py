"""Build a checked encoder-initialization checkpoint for D-F1 through D-F4.

This reproduces the established AGHRI transfer rule: camera backbone from the
camera-only model; LiDAR encoder, decoder, and detector head from LiDAR-only.
The fusion model's camera neck/LSS and fuser remain newly initialized.
"""

import argparse
import json
from pathlib import Path

import torch


def state_dict(path):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a state dictionary: " + str(path))
    return {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", required=True)
    parser.add_argument("--lidar", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    camera = state_dict(args.camera)
    lidar = state_dict(args.lidar)
    selected = {
        key: value for key, value in camera.items()
        if key.startswith("encoders.camera.backbone.")
    }
    for key, value in lidar.items():
        if key.startswith(("encoders.lidar.", "decoder.", "heads.")):
            if key in selected:
                raise RuntimeError("Duplicate initialization key: " + key)
            selected[key] = value

    counts = {
        "camera_backbone": sum(key.startswith("encoders.camera.backbone.") for key in selected),
        "lidar_encoder": sum(key.startswith("encoders.lidar.") for key in selected),
        "decoder": sum(key.startswith("decoder.") for key in selected),
        "head": sum(key.startswith("heads.") for key in selected),
    }
    expected = {"camera_backbone": 187, "lidar_encoder": 126, "decoder": 84, "head": 96}
    if counts != expected or len(selected) != 493:
        raise RuntimeError("Unexpected checkpoint coverage: " + repr(counts))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": selected, "meta": {
        "camera_source": args.camera,
        "lidar_source": args.lidar,
        "selection": "camera backbone; LiDAR encoder, decoder and head",
    }}, output)
    manifest = {
        "camera_source": args.camera,
        "lidar_source": args.lidar,
        "selected_tensor_counts": counts,
        "selected_tensor_total": len(selected),
        "training_loader_note": "tools/train_aghri.py filters class-dependent or shape-incompatible tensors",
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("[BRANCH INIT]", json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
