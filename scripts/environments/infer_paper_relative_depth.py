#!/usr/bin/env python3
"""Infer a reproducible relative-depth illustration from a paper RGB image.

This script uses MiDaS DPT-Hybrid by default to estimate relative inverse depth
from one RGB image. Its output is suitable as a qualitative visualization only: it is
not metric depth and must not be described as an Isaac Sim or physical sensor
measurement.

Example:

    python scripts/environments/infer_paper_relative_depth.py \
        source/isaaclab_tasks/isaaclab_tasks/direct/robot_inspection/paper/UR10_RGB.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source RGB image.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: a relative_depth folder beside the input).",
    )
    parser.add_argument(
        "--model",
        choices=("DPT_Hybrid", "MiDaS_small"),
        default="DPT_Hybrid",
        help="MiDaS model (DPT_Hybrid is slower but preserves more detail).",
    )
    parser.add_argument(
        "--model-repo",
        type=Path,
        default=Path(torch.hub.get_dir()) / "intel-isl_MiDaS_master",
        help="Local MiDaS torch-hub repository.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device.",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def robust_proximity(inverse_depth: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Normalize inverse depth robustly so near pixels are bright/warm."""
    low, high = np.percentile(inverse_depth, (2.0, 98.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise RuntimeError(f"Invalid predicted depth range: [{low}, {high}]")
    proximity = np.clip((inverse_depth - low) / (high - low), 0.0, 1.0)
    return proximity, float(low), float(high)


def ensure_midas_small_backbone() -> None:
    """Cache MiDaS Small's declared EfficientNet dependency non-interactively."""
    dependency_dir = Path(torch.hub.get_dir()) / "rwightman_gen-efficientnet-pytorch_master"
    if dependency_dir.is_dir():
        return
    print("[INFO] Caching MiDaS EfficientNet dependency...")
    # MiDaS v2.1 Small loads this exact public torch-hub repository internally.
    # Loading it once with trust_repo=True avoids an interactive trust prompt in
    # unattended runs. No pretrained classification weights are requested here.
    torch.hub.load(
        "rwightman/gen-efficientnet-pytorch",
        "tf_efficientnet_lite3",
        pretrained=False,
        trust_repo=True,
    )


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not (args.model_repo / "hubconf.py").is_file():
        raise FileNotFoundError(
            f"MiDaS repository not found at {args.model_repo}. "
            "Install it with torch.hub before running this script."
        )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_path.parent / "relative_depth"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(Image.open(input_path).convert("RGB"))
    height, width = rgb.shape[:2]
    device = select_device(args.device)

    if args.model == "MiDaS_small":
        ensure_midas_small_backbone()
    print(f"[INFO] Loading MiDaS {args.model} on {device}...")
    model = torch.hub.load(
        str(args.model_repo),
        args.model,
        source="local",
        pretrained=True,
    ).to(device)
    transforms = torch.hub.load(
        str(args.model_repo),
        "transforms",
        source="local",
    )
    model.eval()

    transform = (
        transforms.dpt_transform
        if args.model == "DPT_Hybrid"
        else transforms.small_transform
    )
    batch = transform(rgb).to(device)
    print(f"[INFO] Inferring relative depth for {width}x{height} image...")
    with torch.inference_mode():
        prediction = model(batch)
        prediction = functional.interpolate(
            prediction.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    inverse_depth = prediction.detach().float().cpu().numpy().astype(np.float32)
    proximity, percentile_2, percentile_98 = robust_proximity(inverse_depth)

    stem = input_path.stem
    raw_path = output_dir / f"{stem}_relative_inverse_depth.npy"
    gray_path = output_dir / f"{stem}_relative_depth_gray.png"
    turbo_path = output_dir / f"{stem}_relative_depth_turbo.png"
    pair_path = output_dir / f"{stem}_relative_depth_pair.png"
    triptych_path = output_dir / f"{stem}_rgb_depth_triptych.png"

    gray = np.uint8(proximity * 255.0 + 0.5)
    turbo_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    turbo = cv2.cvtColor(turbo_bgr, cv2.COLOR_BGR2RGB)
    gutter = np.full((height, max(8, width // 256), 3), 255, dtype=np.uint8)

    np.save(raw_path, inverse_depth)
    Image.fromarray(gray, mode="L").save(gray_path)
    Image.fromarray(turbo, mode="RGB").save(turbo_path)
    gray_rgb = np.repeat(gray[..., None], 3, axis=-1)
    Image.fromarray(
        np.concatenate((gray_rgb, gutter, turbo), axis=1), mode="RGB"
    ).save(pair_path)
    Image.fromarray(
        np.concatenate((rgb, gutter, gray_rgb, gutter, turbo), axis=1), mode="RGB"
    ).save(triptych_path)

    print(f"[INFO] Saved outputs to: {output_dir}")
    print(f"[INFO] Raw prediction: {raw_path.name}")
    print(
        "[INFO] Visualization normalization: "
        f"2nd={percentile_2:.6f}, 98th={percentile_98:.6f}; near=bright/warm"
    )


if __name__ == "__main__":
    main()
