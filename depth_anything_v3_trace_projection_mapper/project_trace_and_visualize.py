#!/usr/bin/env python3
"""Project 3D gripper traces to 2D and visualize predicted/GT traces on video."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_TEST_DATA = PROJECT_ROOT / "test_data"
DEFAULT_CAMERA_DIR = SCRIPT_DIR / "da3_camera_outputs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "trace_projection_outputs"
DEFAULT_POSE_COLUMN = "other_information.observation_gripper_pose6d"
DEFAULT_FRAME_COLUMN = "frame_index"
DEFAULT_GT_TRACE_COLUMN = "annotation.trace"


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("This script needs ffmpeg and ffprobe on PATH.")


def load_rows(parquet_path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq

        return pq.read_table(parquet_path).to_pylist()
    except ImportError:
        pass

    try:
        import pandas as pd

        return pd.read_parquet(parquet_path).to_dict(orient="records")
    except ImportError:
        pass

    try:
        import polars as pl

        return pl.read_parquet(parquet_path).to_dicts()
    except ImportError:
        pass

    raise RuntimeError("No parquet reader found. Install pyarrow, pandas, or polars.")


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


def parse_trace(value: Any) -> list[list[float]] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = ast.literal_eval(text)
        except Exception:
            return None
    if not isinstance(value, (list, tuple)) or not value:
        return None
    points = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        points.append([float(point[0]), float(point[1])])
    return points


def extract_frame_xyz_and_gt(
    rows: list[dict[str, Any]],
    frame_column: str,
    pose_column: str,
    gt_trace_column: str,
) -> tuple[list[int], np.ndarray, list[list[list[float]] | None]]:
    items = []
    for row in rows:
        frame_idx = int(row[frame_column])
        pose = np.asarray(row[pose_column], dtype=float).reshape(-1)
        if pose.size < 3:
            raise ValueError(f"{pose_column} must contain at least xyz")
        items.append((frame_idx, pose[:3], parse_trace(row.get(gt_trace_column))))
    items.sort(key=lambda item: item[0])
    return (
        [item[0] for item in items],
        np.stack([item[1] for item in items], axis=0),
        [item[2] for item in items],
    )


def build_future_traces(xyz: np.ndarray, horizon: int, include_current: bool) -> list[np.ndarray]:
    start_offset = 0 if include_current else 1
    end_offset = horizon if include_current else horizon + 1
    return [xyz[i + start_offset : min(len(xyz), i + end_offset)] for i in range(len(xyz))]


def load_transform(path: Path | None) -> np.ndarray:
    if path is None:
        return np.eye(4, dtype=float)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "T_robot_to_da3" in data:
            data = data["T_robot_to_da3"]
        elif "T_robot_to_world" in data:
            data = data["T_robot_to_world"]
        elif "R" in data and "t" in data:
            T = np.eye(4, dtype=float)
            T[:3, :3] = np.asarray(data["R"], dtype=float).reshape(3, 3)
            T[:3, 3] = np.asarray(data["t"], dtype=float).reshape(3)
            return T
    T = np.asarray(data, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"Transform must be 4x4, got {T.shape}")
    return T


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points.reshape(0, 3)
    homog = np.concatenate([points, np.ones((len(points), 1), dtype=float)], axis=1)
    return (T @ homog.T).T[:, :3]


def load_camera_bundle(camera_npz_path: Path) -> dict[str, np.ndarray]:
    if not camera_npz_path.exists():
        raise FileNotFoundError(f"Camera file not found: {camera_npz_path}")
    data = np.load(camera_npz_path)
    return {
        "frame_indices": np.asarray(data["frame_indices"], dtype=int),
        "intrinsics": np.asarray(data["intrinsics"], dtype=float).reshape(-1, 3, 3),
        "extrinsics": np.asarray(data["extrinsics"], dtype=float).reshape(-1, 3, 4),
    }


def nearest_camera_index(camera_frame_indices: np.ndarray, frame_idx: int) -> int:
    return int(np.argmin(np.abs(camera_frame_indices - frame_idx)))


def project_da3_trace(
    trace_xyz_robot: np.ndarray,
    frame_idx: int,
    camera_bundle: dict[str, np.ndarray],
    T_robot_to_da3: np.ndarray,
) -> list[list[float] | None]:
    if len(trace_xyz_robot) == 0:
        return []
    cam_idx = nearest_camera_index(camera_bundle["frame_indices"], frame_idx)
    K = camera_bundle["intrinsics"][cam_idx]
    E = camera_bundle["extrinsics"][cam_idx]
    xyz_world = transform_points(trace_xyz_robot, T_robot_to_da3)
    homog = np.concatenate([xyz_world, np.ones((len(xyz_world), 1), dtype=float)], axis=1)
    xyz_cam = (E @ homog.T).T
    trace_2d = []
    for x, y, z in xyz_cam:
        if z <= 1e-6:
            trace_2d.append(None)
            continue
        pixel = K @ np.array([x / z, y / z, 1.0], dtype=float)
        trace_2d.append([float(pixel[0]), float(pixel[1])])
    return trace_2d


def fit_affine_from_gt(
    traces_3d: list[np.ndarray],
    gt_traces: list[list[list[float]] | None],
) -> np.ndarray | None:
    source = []
    target = []
    for trace_3d, gt_trace in zip(traces_3d, gt_traces):
        if not gt_trace:
            continue
        count = min(len(trace_3d), len(gt_trace))
        if count <= 0:
            continue
        for idx in range(count):
            source.append([trace_3d[idx, 0], trace_3d[idx, 1], trace_3d[idx, 2], 1.0])
            target.append(gt_trace[idx])
    if len(source) < 4:
        return None
    X = np.asarray(source, dtype=float)
    Y = np.asarray(target, dtype=float)
    affine, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return affine


def project_affine_trace(trace_xyz: np.ndarray, affine: np.ndarray | None) -> list[list[float] | None]:
    if affine is None:
        return [None for _ in trace_xyz]
    X = np.concatenate([trace_xyz, np.ones((len(trace_xyz), 1), dtype=float)], axis=1)
    return (X @ affine).tolist()


def visible_ratio(traces_2d: list[list[list[float] | None]], width: int, height: int) -> float:
    total = 0
    visible = 0
    for trace in traces_2d:
        for point in trace:
            if point is None:
                continue
            total += 1
            x, y = point
            if 0 <= x < width and 0 <= y < height:
                visible += 1
    return visible / total if total else 0.0


def trace_error(
    pred_trace: list[list[float] | None],
    gt_trace: list[list[float]] | None,
) -> dict[str, float | int | None]:
    if not gt_trace:
        return {"point_count": 0, "mean_l2_px": None, "endpoint_l2_px": None}
    dists = []
    count = min(len(pred_trace), len(gt_trace))
    for idx in range(count):
        pred = pred_trace[idx]
        if pred is None:
            continue
        dists.append(float(np.linalg.norm(np.asarray(pred[:2]) - np.asarray(gt_trace[idx][:2]))))
    endpoint = None
    if count > 0 and pred_trace[count - 1] is not None:
        endpoint = float(np.linalg.norm(np.asarray(pred_trace[count - 1][:2]) - np.asarray(gt_trace[count - 1][:2])))
    return {
        "point_count": len(dists),
        "mean_l2_px": float(np.mean(dists)) if dists else None,
        "endpoint_l2_px": endpoint,
    }


def draw_trace(
    draw: ImageDraw.ImageDraw,
    trace: list[list[float] | None],
    color: tuple[int, int, int, int],
    width: int,
    point_radius: int,
) -> None:
    points = []
    for point in trace:
        if point is None:
            continue
        x, y = point
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        points.append((int(round(x)), int(round(y))))
    if len(points) >= 2:
        draw.line(points, fill=color, width=width)
    for point in points:
        draw.ellipse(
            [point[0] - point_radius, point[1] - point_radius, point[0] + point_radius, point[1] + point_radius],
            fill=color,
            outline=(0, 0, 0, 180),
        )


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


def ffmpeg_writer(output_path: Path, width: int, height: int, fps: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        fps,
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def render_overlay_video(
    video_path: Path,
    output_path: Path,
    video_info: dict[str, Any],
    frame_indices: list[int],
    pred_traces: list[list[list[float] | None]],
    gt_traces: list[list[list[float]] | None],
    projection_label: str,
) -> None:
    reader = ffmpeg_reader(video_path)
    writer = ffmpeg_writer(output_path, video_info["width"], video_info["height"], video_info["fps"])
    assert reader.stdout is not None
    assert writer.stdin is not None

    frame_size = video_info["width"] * video_info["height"] * 3
    lookup = {frame_idx: idx for idx, frame_idx in enumerate(frame_indices)}
    frame_idx = 0
    try:
        while True:
            raw = reader.stdout.read(frame_size)
            if not raw:
                break
            if len(raw) != frame_size:
                raise RuntimeError("Partial raw frame from ffmpeg")
            image = Image.frombytes("RGB", (video_info["width"], video_info["height"]), raw)
            draw = ImageDraw.Draw(image, "RGBA")
            idx = lookup.get(frame_idx)
            if idx is not None:
                draw_trace(draw, gt_traces[idx] or [], (255, 80, 220, 230), width=3, point_radius=4)
                draw_trace(draw, pred_traces[idx], (30, 220, 255, 230), width=3, point_radius=3)
                draw.rectangle([6, 6, 238, 47], fill=(0, 0, 0, 150))
                draw.text((12, 12), f"frame {frame_idx}", fill=(255, 255, 255, 255))
                draw.text((12, 29), f"pred: cyan | gt: magenta | {projection_label}", fill=(235, 235, 235, 255))
            writer.stdin.write(image.tobytes())
            frame_idx += 1
    finally:
        reader.stdout.close()
        writer.stdin.close()
        reader.wait()
        writer.wait()

    if reader.returncode != 0:
        raise RuntimeError(f"ffmpeg reader failed with code {reader.returncode}")
    if writer.returncode != 0:
        raise RuntimeError(f"ffmpeg writer failed with code {writer.returncode}")


def evaluate_episode(
    episode_id: str,
    parquet_path: Path,
    video_path: Path,
    camera_npz_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows = load_rows(parquet_path)
    frame_indices, xyz, gt_traces = extract_frame_xyz_and_gt(
        rows,
        frame_column=args.frame_column,
        pose_column=args.pose_column,
        gt_trace_column=args.gt_trace_column,
    )
    traces_3d = build_future_traces(xyz, args.horizon, args.include_current)
    video_info = probe_video(video_path)
    camera_bundle = load_camera_bundle(camera_npz_path)
    T_robot_to_da3 = load_transform(args.robot_to_da3_json)

    da3_traces = [
        project_da3_trace(trace, frame_idx, camera_bundle, T_robot_to_da3)
        for frame_idx, trace in zip(frame_indices, traces_3d)
    ]
    projection_label = "da3"
    pred_traces = da3_traces
    da3_visible = visible_ratio(da3_traces, video_info["width"], video_info["height"])
    affine = None
    if args.projection_mode in ("fit", "auto"):
        affine = fit_affine_from_gt(traces_3d, gt_traces)
    if args.projection_mode == "fit" or (
        args.projection_mode == "auto" and da3_visible < args.da3_visible_threshold and affine is not None
    ):
        pred_traces = [project_affine_trace(trace, affine) for trace in traces_3d]
        projection_label = f"fit_from_gt(da3_visible={da3_visible:.2f})"

    records = []
    mean_errors = []
    endpoint_errors = []
    frames_with_gt = 0
    for frame_idx, trace_3d, pred_trace, gt_trace in zip(frame_indices, traces_3d, pred_traces, gt_traces):
        err = trace_error(pred_trace, gt_trace)
        if gt_trace:
            frames_with_gt += 1
        if err["mean_l2_px"] is not None:
            mean_errors.append(float(err["mean_l2_px"]))
        if err["endpoint_l2_px"] is not None:
            endpoint_errors.append(float(err["endpoint_l2_px"]))
        records.append(
            {
                "frame_index": frame_idx,
                "future_trace_3d": trace_3d.tolist(),
                "pred_trace_2d": pred_trace,
                "gt_trace_2d": gt_trace,
                "trace_error": err,
            }
        )

    episode_dir = output_dir / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)
    json_path = episode_dir / "trace_projection.json"
    video_out = episode_dir / "trace_projection_overlay.mp4"
    payload = {
        "episode_id": episode_id,
        "video": str(video_path),
        "parquet": str(parquet_path),
        "camera_npz": str(camera_npz_path),
        "pose_column": args.pose_column,
        "gt_trace_column": args.gt_trace_column,
        "horizon": args.horizon,
        "include_current": args.include_current,
        "projection_label": projection_label,
        "da3_visible_ratio": da3_visible,
        "mean_trace_l2_px": float(np.mean(mean_errors)) if mean_errors else None,
        "mean_endpoint_l2_px": float(np.mean(endpoint_errors)) if endpoint_errors else None,
        "frames_with_gt_trace": frames_with_gt,
        "records": records,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not args.no_video:
        render_overlay_video(
            video_path=video_path,
            output_path=video_out,
            video_info=video_info,
            frame_indices=frame_indices,
            pred_traces=pred_traces,
            gt_traces=gt_traces,
            projection_label=projection_label,
        )

    return {
        "episode_id": episode_id,
        "projection_label": projection_label,
        "da3_visible_ratio": da3_visible,
        "frames_with_gt_trace": frames_with_gt,
        "mean_trace_l2_px": payload["mean_trace_l2_px"],
        "mean_endpoint_l2_px": payload["mean_endpoint_l2_px"],
        "json_path": str(json_path),
        "video_path": str(video_out) if not args.no_video else None,
    }


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "projection_eval_summary.json"
    json_path.write_text(json.dumps({"episodes": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fields = [
        "episode_id",
        "projection_label",
        "da3_visible_ratio",
        "frames_with_gt_trace",
        "mean_trace_l2_px",
        "mean_endpoint_l2_px",
        "json_path",
        "video_path",
    ]
    csv_path = output_dir / "projection_eval_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
        writer.writerow({})
        writer.writerow(
            {
                "episode_id": "MEAN",
                "da3_visible_ratio": np.mean([r["da3_visible_ratio"] for r in rows]) if rows else None,
                "frames_with_gt_trace": np.mean([r["frames_with_gt_trace"] for r in rows]) if rows else None,
                "mean_trace_l2_px": np.mean([r["mean_trace_l2_px"] for r in rows if r["mean_trace_l2_px"] is not None])
                if any(r["mean_trace_l2_px"] is not None for r in rows)
                else None,
                "mean_endpoint_l2_px": np.mean(
                    [r["mean_endpoint_l2_px"] for r in rows if r["mean_endpoint_l2_px"] is not None]
                )
                if any(r["mean_endpoint_l2_px"] is not None for r in rows)
                else None,
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project 3D future traces to 2D and overlay predicted/GT traces on videos."
    )
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--camera-dir", type=Path, default=DEFAULT_CAMERA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern", default="episode_*.parquet")
    parser.add_argument("--frame-column", default=DEFAULT_FRAME_COLUMN)
    parser.add_argument("--pose-column", default=DEFAULT_POSE_COLUMN)
    parser.add_argument("--gt-trace-column", default=DEFAULT_GT_TRACE_COLUMN)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--include-current", action="store_true", default=True)
    parser.add_argument("--exclude-current", action="store_false", dest="include_current")
    parser.add_argument("--robot-to-da3-json", type=Path, default=None)
    parser.add_argument("--projection-mode", choices=("da3", "fit", "auto"), default="auto")
    parser.add_argument("--da3-visible-threshold", type=float, default=0.2)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    require_ffmpeg()
    parquets = sorted(args.test_data.glob(args.pattern))
    if not parquets:
        raise FileNotFoundError(f"No parquets matched {args.test_data / args.pattern}")

    summaries = []
    for parquet_path in parquets:
        episode_id = parquet_path.stem
        video_path = args.test_data / f"{episode_id}.mp4"
        camera_npz_path = args.camera_dir / episode_id / "da3_camera_parameters.npz"
        print(f"[{episode_id}] projecting traces")
        summary = evaluate_episode(
            episode_id=episode_id,
            parquet_path=parquet_path,
            video_path=video_path,
            camera_npz_path=camera_npz_path,
            output_dir=args.output_dir,
            args=args,
        )
        summaries.append(summary)
        print(
            f"[{episode_id}] mean_trace_l2_px={summary['mean_trace_l2_px']} "
            f"projection={summary['projection_label']}"
        )

    write_summary(args.output_dir, summaries)
    print(f"Wrote projection summary to {args.output_dir}")


if __name__ == "__main__":
    main()
