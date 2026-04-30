#!/usr/bin/env python3
"""Estimate Depth Anything V3 camera parameters for videos in test_data."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_TEST_DATA = PROJECT_ROOT / "test_data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "da3_camera_outputs"


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("This script needs ffmpeg and ffprobe on PATH.")


def probe_video(video_path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_count": int(stream["nb_frames"]) if str(stream.get("nb_frames", "")).isdigit() else None,
        "fps": stream.get("r_frame_rate", "30/1"),
    }


def ffmpeg_reader(video_path: Path):
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE)


def extract_sampled_frames(
    video_path: Path,
    frames_dir: Path,
    frame_stride: int,
    max_frames: int | None,
) -> tuple[list[int], list[Path], dict[str, Any]]:
    video_info = probe_video(video_path)
    frames_dir.mkdir(parents=True, exist_ok=True)
    width = video_info["width"]
    height = video_info["height"]
    frame_size = width * height * 3
    reader = ffmpeg_reader(video_path)
    assert reader.stdout is not None

    frame_indices = []
    frame_paths = []
    frame_idx = 0
    try:
        while True:
            raw = reader.stdout.read(frame_size)
            if not raw:
                break
            if len(raw) != frame_size:
                raise RuntimeError(f"Partial raw frame while reading {video_path}")
            should_keep = frame_idx % frame_stride == 0
            if max_frames is not None and len(frame_indices) >= max_frames:
                should_keep = False
            if should_keep:
                image = Image.frombytes("RGB", (width, height), raw)
                frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
                image.save(frame_path, quality=95)
                frame_indices.append(frame_idx)
                frame_paths.append(frame_path)
            frame_idx += 1
    finally:
        reader.stdout.close()
        reader.wait()

    if reader.returncode != 0:
        raise RuntimeError(f"ffmpeg failed while reading {video_path}")
    if not frame_paths:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return frame_indices, frame_paths, video_info


def load_da3_model(model_id: str, device: str):
    try:
        import torch
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:
        raise RuntimeError(
            "Depth Anything V3 is not installed. Install it following the official "
            "Depth-Anything-3 README, then rerun this script."
        ) from exc

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = DepthAnything3.from_pretrained(model_id)
    model = model.to(device=device)
    return model, device


def as_array(prediction: dict[str, Any], key: str) -> np.ndarray | None:
    value = prediction.get(key)
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def save_prediction(
    episode_id: str,
    output_dir: Path,
    frame_indices: list[int],
    frame_paths: list[Path],
    video_info: dict[str, Any],
    prediction: dict[str, Any],
    model_id: str,
    device: str,
    save_depth: bool,
) -> None:
    episode_dir = output_dir / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)

    intrinsics = as_array(prediction, "intrinsics")
    extrinsics = as_array(prediction, "extrinsics")
    if intrinsics is None or extrinsics is None:
        raise KeyError("DA3 prediction did not include both 'intrinsics' and 'extrinsics'")

    arrays = {
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
    }
    if save_depth:
        for key in ("depth", "conf"):
            value = as_array(prediction, key)
            if value is not None:
                arrays[key] = value

    npz_path = episode_dir / "da3_camera_parameters.npz"
    np.savez_compressed(npz_path, **arrays)

    processed_images = as_array(prediction, "processed_images")
    processed_hw = None
    if processed_images is not None and processed_images.ndim >= 3:
        processed_hw = [int(processed_images.shape[-3]), int(processed_images.shape[-2])]

    metadata = {
        "episode_id": episode_id,
        "model_id": model_id,
        "device": device,
        "video_info": video_info,
        "frame_indices": frame_indices,
        "frames": [str(path) for path in frame_paths],
        "npz_path": str(npz_path),
        "intrinsics_shape": list(intrinsics.shape),
        "extrinsics_shape": list(extrinsics.shape),
        "processed_hw": processed_hw,
        "saved_depth": save_depth,
        "extrinsics_convention": "world_to_camera, shape N x 3 x 4",
    }
    (episode_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Depth Anything V3 on videos and save per-frame camera parameters."
    )
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern", default="episode_*.mp4")
    parser.add_argument("--model-id", default="depth-anything/Depth-Anything-3-Large")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--process-res-method", default="upper_bound")
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument("--overwrite-frames", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")

    require_ffmpeg()
    videos = sorted(args.test_data.glob(args.pattern))
    if not videos:
        raise FileNotFoundError(f"No videos matched {args.test_data / args.pattern}")

    model, device = load_da3_model(args.model_id, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for video_path in videos:
        episode_id = video_path.stem
        episode_dir = args.output_dir / episode_id
        frames_dir = episode_dir / "frames"
        if frames_dir.exists() and args.overwrite_frames:
            shutil.rmtree(frames_dir)

        print(f"[{episode_id}] extracting frames from {video_path}")
        frame_indices, frame_paths, video_info = extract_sampled_frames(
            video_path=video_path,
            frames_dir=frames_dir,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
        )
        print(f"[{episode_id}] running DA3 on {len(frame_paths)} frames")
        prediction = model.inference(
            [str(path) for path in frame_paths],
            process_res=args.process_res,
            process_res_method=args.process_res_method,
        )
        save_prediction(
            episode_id=episode_id,
            output_dir=args.output_dir,
            frame_indices=frame_indices,
            frame_paths=frame_paths,
            video_info=video_info,
            prediction=prediction,
            model_id=args.model_id,
            device=device,
            save_depth=args.save_depth,
        )
        print(f"[{episode_id}] wrote {episode_dir / 'da3_camera_parameters.npz'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
