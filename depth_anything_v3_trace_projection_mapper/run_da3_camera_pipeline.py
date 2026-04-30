#!/usr/bin/env python3
"""Estimate Depth Anything V3 camera parameters for videos in test_data."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_TEST_DATA = PROJECT_ROOT / "test_data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "da3_camera_outputs"


def open_video_reader(video_path: Path):
    try:
        return imageio.get_reader(str(video_path))
    except ValueError as exc:
        raise RuntimeError(
            "Could not open video with imageio. Install the ffmpeg backend with: "
            "python -m pip install 'imageio[ffmpeg]'"
        ) from exc


def get_video_info(reader) -> dict[str, Any]:
    metadata = reader.get_meta_data()
    size = metadata.get("size")
    if size:
        width, height = size
    else:
        first_frame = reader.get_data(0)
        height, width = first_frame.shape[:2]

    frame_count = metadata.get("nframes")
    if frame_count is None or frame_count == float("inf"):
        frame_count = reader.count_frames()

    fps = metadata.get("fps")
    return {
        "width": int(width),
        "height": int(height),
        "frame_count": int(frame_count),
        "fps": float(fps) if fps is not None else None,
    }


def choose_frame_indices(total_frames: int, sampling_mode: str, num_uniform_frames: int) -> list[int]:
    if sampling_mode == "first_frame":
        return [0]
    if sampling_mode == "uniform":
        if num_uniform_frames <= 0:
            raise ValueError("--num-uniform-frames must be positive")
        count = min(num_uniform_frames, total_frames)
        return sorted(set(np.linspace(0, total_frames - 1, count, dtype=int).tolist()))
    raise ValueError(f"Unsupported sampling mode: {sampling_mode}")


def extract_sampled_frames(
    video_path: Path,
    frames_dir: Path,
    sampling_mode: str,
    num_uniform_frames: int,
) -> tuple[list[int], list[Path], dict[str, Any]]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("frame_*.jpg"):
        old_frame.unlink()

    reader = open_video_reader(video_path)
    try:
        video_info = get_video_info(reader)
        frame_indices = choose_frame_indices(
            video_info["frame_count"],
            sampling_mode=sampling_mode,
            num_uniform_frames=num_uniform_frames,
        )
        frame_paths = []
        for frame_idx in frame_indices:
            frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
            Image.fromarray(reader.get_data(frame_idx)).convert("RGB").save(frame_path, quality=95)
            frame_paths.append(frame_path)
    finally:
        reader.close()

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
    sampling_mode: str,
    num_uniform_frames: int,
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
        "sampling_mode": sampling_mode,
        "num_uniform_frames": num_uniform_frames if sampling_mode == "uniform" else None,
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
    parser.add_argument("--model-id", default="depth-anything/da3-large")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument(
        "--sampling-mode",
        choices=("first_frame", "uniform"),
        default="first_frame",
        help="DA3 input frame selection strategy. Default: uniform.",
    )
    parser.add_argument(
        "--num-uniform-frames",
        type=int,
        default=32,
        help="Number of frames for --sampling-mode uniform. Ignored for first_frame.",
    )
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--process-res-method", default="upper_bound")
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument("--overwrite-frames", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_uniform_frames <= 0:
        raise ValueError("--num-uniform-frames must be positive")

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
            sampling_mode=args.sampling_mode,
            num_uniform_frames=args.num_uniform_frames,
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
            sampling_mode=args.sampling_mode,
            num_uniform_frames=args.num_uniform_frames,
            save_depth=args.save_depth,
        )
        print(f"[{episode_id}] wrote {episode_dir / 'da3_camera_parameters.npz'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
