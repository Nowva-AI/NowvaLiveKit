#!/usr/bin/env python3
"""
Download RTMPose ONNX models for the biomechanics pipeline.

Usage:
    python scripts/download_models.py                # download default (rtmpose-m)
    python scripts/download_models.py --model rtmpose-l   # larger variant
    python scripts/download_models.py --list          # show available models
    python scripts/download_models.py --force         # re-download even if exists
"""

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

# Model registry — OpenMMLab ONNX SDK releases
MODELS = {
    "rtmpose-m": {
        "url": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
        ),
        "filename": "rtmpose-m-256x192.onnx",
        "description": "RTMPose-M (medium) 256x192 — good speed/accuracy balance",
    },
    "rtmpose-l": {
        "url": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "onnx_sdk/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.zip"
        ),
        "filename": "rtmpose-l-256x192.onnx",
        "description": "RTMPose-L (large) 256x192 — higher accuracy, slower",
    },
    "rtmpose-m-halpe26": {
        "url": (
            "https://huggingface.co/pocketpose/onnx-models/resolve/main/"
            "pose_estimation-halpe26/halpe26_td-cc_rtmpose-m_body7_256x192.onnx"
        ),
        "filename": "rtmpose-m-halpe26-256x192.onnx",
        "description": "RTMPose-M (medium) halpe26 256x192 — 26 keypoints with foot landmarks",
    },
}

DEFAULT_MODEL = "rtmpose-m"

# Destination directory
MODEL_DIR = Path(__file__).resolve().parents[2] / "src" / "biomechanics" / "pose" / "models"


def download_model(model_key: str, force: bool = False) -> Path:
    """Download and extract the specified RTMPose ONNX model."""
    info = MODELS[model_key]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    dest = MODEL_DIR / info["filename"]

    if dest.exists() and not force:
        print(f"Model already exists: {dest}")
        print(f"Use --force to re-download.")
        return dest

    url = info["url"]
    print(f"Downloading {info['description']}...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest}")

    tmp_path = dest.with_suffix(".download")

    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            percent = min(100, count * block_size * 100 // total_size)
            mb_done = count * block_size / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {percent}% ({mb_done:.1f}/{mb_total:.1f} MB)")
        else:
            mb_done = count * block_size / (1024 * 1024)
            sys.stdout.write(f"\r  Downloaded: {mb_done:.1f} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=progress_hook)
        print()  # newline after progress
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"\nDownload failed: {e}")
        sys.exit(1)

    # Extract .onnx from zip if needed
    if url.endswith(".zip"):
        print("  Extracting ONNX model from zip...")
        try:
            with zipfile.ZipFile(tmp_path, "r") as zf:
                onnx_files = [f for f in zf.namelist() if f.endswith(".onnx")]
                if not onnx_files:
                    print(f"  No .onnx file found in archive. Contents: {zf.namelist()}")
                    tmp_path.unlink()
                    sys.exit(1)

                # Extract the first .onnx file
                onnx_name = onnx_files[0]
                print(f"  Extracting: {onnx_name}")
                with zf.open(onnx_name) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
        except zipfile.BadZipFile:
            # Not a zip — might be a direct .onnx download
            print("  File is not a zip archive, treating as direct ONNX model.")
            tmp_path.rename(dest)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    else:
        tmp_path.rename(dest)

    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  Model saved: {dest} ({size_mb:.1f} MB)")
    return dest


def main():
    parser = argparse.ArgumentParser(description="Download RTMPose ONNX models")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default=DEFAULT_MODEL,
        help=f"Model variant to download (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if model already exists",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("Available RTMPose models:")
        for key, info in MODELS.items():
            marker = " (default)" if key == DEFAULT_MODEL else ""
            print(f"  {key}{marker}: {info['description']}")
            print(f"    filename: {info['filename']}")
        return

    print("=" * 60)
    print("  RTMPose Model Downloader")
    print("=" * 60)
    print()

    download_model(args.model, force=args.force)

    print()
    print("To use RTMPose, set in config/biomechanics.yaml:")
    print("  pose:")
    print("    backend: rtmpose")
    print(f"    model_path: src/biomechanics/pose/models/{MODELS[args.model]['filename']}")
    print()


if __name__ == "__main__":
    main()
