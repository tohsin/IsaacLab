#!/usr/bin/env python3
"""Download native-OpenUSD inspection scenes used by inspection_null.py."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSET_ROOT = REPO_ROOT / "assets/environments"

PACKS = {
    "usd_explorer_factory": {
        "url": "https://d4i3qtqj3r0z5.cloudfront.net/USD_Explorer_Sample_NVD%4010011.zip",
        "size": "approximately 508 MB",
        "directory": "nvidia_usd_explorer",
        "entry": "Usd_Explorer/Samples/Examples/2023_2/Factory/Factory.usd",
    },
    "defect_workshop": {
        "url": "https://d4i3qtqj3r0z5.cloudfront.net/DefectDet_DemoPack_NVDA%401.0.1.zip",
        "size": "approximately 325 MB",
        "directory": "nvidia_defect_detection",
        "entry": "shop.usdc",
    },
}


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "IsaacLab-inspection-scene-installer"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r[INFO] Downloaded {downloaded / total:6.1%}", end="", flush=True)
    print()


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if destination_resolved != member_path and destination_resolved not in member_path.parents:
                raise RuntimeError(f"Unsafe ZIP member path: {member.filename}")
        archive.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", choices=PACKS)
    parser.add_argument(
        "--asset_root",
        type=Path,
        default=DEFAULT_ASSET_ROOT,
        help="Directory under which the selected pack is extracted.",
    )
    args = parser.parse_args()

    pack = PACKS[args.pack]
    asset_root = args.asset_root.expanduser().resolve()
    destination = asset_root / pack["directory"]
    expected_scene = destination / pack["entry"]
    if expected_scene.is_file():
        print(f"[INFO] Scene is already installed: {expected_scene}")
        return
    if destination.exists():
        raise RuntimeError(
            f"Incomplete destination already exists: {destination}. "
            "Move it aside or remove it manually before retrying."
        )

    asset_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading {args.pack} ({pack['size']}) from NVIDIA...")
    with tempfile.TemporaryDirectory(prefix="inspection_scene_") as temporary_directory:
        archive_path = Path(temporary_directory) / "asset_pack.zip"
        extract_path = Path(temporary_directory) / "extracted"
        extract_path.mkdir()
        _download(pack["url"], archive_path)
        print("[INFO] Extracting asset pack...")
        _safe_extract(archive_path, extract_path)
        if not (extract_path / pack["entry"]).is_file():
            raise RuntimeError(f"Downloaded pack does not contain expected scene: {pack['entry']}")
        shutil.move(str(extract_path), str(destination))

    print(f"[INFO] Installed scene: {expected_scene}")
    print(
        "[INFO] Launch with: python scripts/environments/inspection_null.py "
        f"--environment {args.pack}"
    )


if __name__ == "__main__":
    main()
