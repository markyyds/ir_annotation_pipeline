#!/usr/bin/env python3
"""Evaluate video_phase_segmentation_mapper against annotated test parquets."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from generate_gripper_phase_annotations import (
    DEFAULT_ANNOTATION_CONTACT_COLUMN,
    DEFAULT_FRAME_COLUMN,
    DEFAULT_GRIPPER_COLUMN,
    DEFAULT_POSE_COLUMN,
    DEFAULT_TCP_POSE_COLUMN,
    build_phase_by_step,
    determine_grasp_phases,
    extract_series,
    first_frame_for_phase,
    get_pose,
    get_row_by_frame,
    json_ready,
    load_rows,
    normalize_gripper,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_TEST_DATA = PROJECT_ROOT / "test_data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "eval_outputs"
DEFAULT_STATE_AFFORDANCE_COLUMN = "annotation.state_affordance"


def parse_int_like(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = int(float(text))
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def get_ground_truth_contact_frame(rows: list[dict[str, Any]], column: str) -> int | None:
    for row in rows:
        parsed = parse_int_like(row.get(column))
        if parsed is not None:
            return parsed
    return None


def parse_vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        values = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        text = str(value).strip()
        if not text or text == "[]":
            return None
        try:
            values = ast.literal_eval(text)
        except Exception:
            return None
    if not isinstance(values, (list, tuple)) or not values:
        return None
    return [float(item) for item in values]


def get_ground_truth_state_affordance(rows: list[dict[str, Any]], column: str) -> list[float] | None:
    for row in rows:
        parsed = parse_vector(row.get(column))
        if parsed is not None:
            return parsed
    return None


def l2(values_a: list[float] | None, values_b: list[float] | None, dims: slice) -> float | None:
    if values_a is None or values_b is None:
        return None
    a = np.asarray(values_a, dtype=float).reshape(-1)[dims]
    b = np.asarray(values_b, dtype=float).reshape(-1)[dims]
    if len(a) != len(b):
        return None
    return float(np.linalg.norm(a - b))


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def rmse(values: list[float]) -> float | None:
    return float(math.sqrt(np.mean(np.square(values)))) if values else None


def accuracy_at(abs_errors: list[float], tolerance: int) -> float | None:
    if not abs_errors:
        return None
    return float(np.mean([err <= tolerance for err in abs_errors]))


def phase_ranges_to_json(phases: dict[str, Any]) -> dict[str, list[list[int]]]:
    return {
        phase: [[int(start), int(end)] for start, end in ranges]
        for phase, ranges in phases["phase_ranges"].items()
    }


def evaluate_episode(
    parquet_path: Path,
    grasp_threshold: float,
    contact_threshold: float,
    frame_column: str,
    gripper_column: str,
    pose_column: str,
    tcp_pose_column: str,
    gt_contact_column: str,
    gt_state_affordance_column: str,
) -> dict[str, Any]:
    rows = load_rows(parquet_path)
    if not rows:
        raise ValueError(f"{parquet_path} is empty")

    frame_indices, gripper_positions = extract_series(rows, frame_column, gripper_column)
    normalized_gripper = normalize_gripper(gripper_positions)
    phases = determine_grasp_phases(
        normalized_gripper_actions=normalized_gripper,
        grasp_threshold=grasp_threshold,
        contact_threshold=contact_threshold,
    )
    phase_by_step = build_phase_by_step(phases, len(frame_indices))
    row_by_frame = get_row_by_frame(rows, frame_column)

    pred_contact_frame = first_frame_for_phase(frame_indices, phase_by_step, "contact")
    gt_contact_frame = get_ground_truth_contact_frame(rows, gt_contact_column)
    gt_state_affordance = get_ground_truth_state_affordance(rows, gt_state_affordance_column)

    pred_row = row_by_frame.get(pred_contact_frame) if pred_contact_frame is not None else None
    gt_row = row_by_frame.get(gt_contact_frame) if gt_contact_frame is not None else None

    pred_gripper_pose = get_pose(pred_row, pose_column) if pred_row is not None else None
    pred_tcp_pose = get_pose(pred_row, tcp_pose_column) if pred_row is not None else None
    gt_tcp_pose = get_pose(gt_row, tcp_pose_column) if gt_row is not None else None

    frame_error = (
        int(pred_contact_frame - gt_contact_frame)
        if pred_contact_frame is not None and gt_contact_frame is not None
        else None
    )
    abs_frame_error = abs(frame_error) if frame_error is not None else None
    episode_length = len(frame_indices)

    return {
        "episode_id": parquet_path.stem,
        "parquet": str(parquet_path),
        "num_frames": episode_length,
        "gt_contact_frame": gt_contact_frame,
        "pred_contact_frame": pred_contact_frame,
        "frame_error": frame_error,
        "abs_frame_error": abs_frame_error,
        "normalized_abs_frame_error": (
            abs_frame_error / episode_length if abs_frame_error is not None and episode_length else None
        ),
        "gt_state_affordance": gt_state_affordance,
        "gt_tcp_pose_at_gt_contact": gt_tcp_pose,
        "pred_tcp_pose_at_pred_contact": pred_tcp_pose,
        "pred_gripper_pose_at_pred_contact": pred_gripper_pose,
        "tcp_position_l2_error": l2(pred_tcp_pose, gt_state_affordance, slice(0, 3)),
        "tcp_rotation_l2_error": l2(pred_tcp_pose, gt_state_affordance, slice(3, 6)),
        "tcp_pose6d_l2_error": l2(pred_tcp_pose, gt_state_affordance, slice(0, 6)),
        "oracle_gt_tcp_position_l2_error": l2(gt_tcp_pose, gt_state_affordance, slice(0, 3)),
        "phase_ranges": phase_ranges_to_json(phases),
        "gripper_raw_min": min(gripper_positions),
        "gripper_raw_max": max(gripper_positions),
    }


def summarize(results: list[dict[str, Any]], tolerances: list[int]) -> dict[str, Any]:
    valid_frame = [row for row in results if row["abs_frame_error"] is not None]
    abs_errors = [float(row["abs_frame_error"]) for row in valid_frame]
    signed_errors = [float(row["frame_error"]) for row in valid_frame]
    normalized_errors = [
        float(row["normalized_abs_frame_error"])
        for row in valid_frame
        if row["normalized_abs_frame_error"] is not None
    ]
    position_errors = [
        float(row["tcp_position_l2_error"])
        for row in results
        if row["tcp_position_l2_error"] is not None
    ]
    rotation_errors = [
        float(row["tcp_rotation_l2_error"])
        for row in results
        if row["tcp_rotation_l2_error"] is not None
    ]
    pose6d_errors = [
        float(row["tcp_pose6d_l2_error"])
        for row in results
        if row["tcp_pose6d_l2_error"] is not None
    ]

    return {
        "num_episodes": len(results),
        "num_with_gt_and_prediction": len(valid_frame),
        "contact_frame": {
            "mae_frames": mean(abs_errors),
            "median_abs_error_frames": median(abs_errors),
            "rmse_frames": rmse(signed_errors),
            "mean_signed_error_frames": mean(signed_errors),
            "max_abs_error_frames": max(abs_errors) if abs_errors else None,
            "mean_normalized_abs_error": mean(normalized_errors),
            "accuracy_at_tolerance": {
                f"within_{tol}_frames": accuracy_at(abs_errors, tol) for tol in tolerances
            },
        },
        "contact_pose_vs_state_affordance": {
            "mean_tcp_position_l2_m": mean(position_errors),
            "median_tcp_position_l2_m": median(position_errors),
            "mean_tcp_rotation_l2_rad": mean(rotation_errors),
            "mean_tcp_pose6d_l2": mean(pose6d_errors),
        },
    }


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "episode_id",
        "num_frames",
        "gt_contact_frame",
        "pred_contact_frame",
        "frame_error",
        "abs_frame_error",
        "normalized_abs_frame_error",
        "tcp_position_l2_error",
        "tcp_rotation_l2_error",
        "tcp_pose6d_l2_error",
        "gripper_raw_min",
        "gripper_raw_max",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field) for field in fields})


def write_csv_with_summary(path: Path, results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    fields = [
        "episode_id",
        "num_frames",
        "gt_contact_frame",
        "pred_contact_frame",
        "frame_error",
        "abs_frame_error",
        "normalized_abs_frame_error",
        "tcp_position_l2_error",
        "tcp_rotation_l2_error",
        "tcp_pose6d_l2_error",
        "gripper_raw_min",
        "gripper_raw_max",
        "metric",
        "value",
    ]
    numeric_fields = [
        "num_frames",
        "gt_contact_frame",
        "pred_contact_frame",
        "frame_error",
        "abs_frame_error",
        "normalized_abs_frame_error",
        "tcp_position_l2_error",
        "tcp_rotation_l2_error",
        "tcp_pose6d_l2_error",
        "gripper_raw_min",
        "gripper_raw_max",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field) for field in fields})

        writer.writerow({})

        mean_row = {"episode_id": "MEAN"}
        for field in numeric_fields:
            values = [float(row[field]) for row in results if row.get(field) is not None]
            mean_row[field] = mean(values)
        writer.writerow(mean_row)

        writer.writerow({})
        frame_metrics = summary["contact_frame"]
        pose_metrics = summary["contact_pose_vs_state_affordance"]
        metric_rows = [
            ("evaluated_episodes", summary["num_with_gt_and_prediction"]),
            ("total_episodes", summary["num_episodes"]),
            ("contact_frame_mae_frames", frame_metrics["mae_frames"]),
            ("contact_frame_rmse_frames", frame_metrics["rmse_frames"]),
            ("contact_frame_median_abs_error_frames", frame_metrics["median_abs_error_frames"]),
            ("contact_frame_mean_signed_error_frames", frame_metrics["mean_signed_error_frames"]),
            ("contact_frame_max_abs_error_frames", frame_metrics["max_abs_error_frames"]),
            ("contact_frame_mean_normalized_abs_error", frame_metrics["mean_normalized_abs_error"]),
        ]
        metric_rows.extend(
            (name, value) for name, value in frame_metrics["accuracy_at_tolerance"].items()
        )
        metric_rows.extend(
            [
                ("mean_tcp_position_l2_m", pose_metrics["mean_tcp_position_l2_m"]),
                ("median_tcp_position_l2_m", pose_metrics["median_tcp_position_l2_m"]),
                ("mean_tcp_rotation_l2_rad", pose_metrics["mean_tcp_rotation_l2_rad"]),
                ("mean_tcp_pose6d_l2", pose_metrics["mean_tcp_pose6d_l2"]),
            ]
        )
        for metric, value in metric_rows:
            writer.writerow({"episode_id": "SUMMARY", "metric": metric, "value": value})


def parse_tolerances(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def threshold_tag(grasp_threshold: float, contact_threshold: float) -> str:
    def fmt(value: float) -> str:
        text = f"{value:.6g}".replace("-", "m").replace(".", ".")
        return text

    return f"grasp_{fmt(grasp_threshold)}__contact_{fmt(contact_threshold)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate predicted first-contact phase against annotation.contact_frame "
            "and predicted TCP pose against annotation.state_affordance."
        )
    )
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern", default="episode_*.parquet")
    parser.add_argument("--frame-column", default=DEFAULT_FRAME_COLUMN)
    parser.add_argument("--gripper-column", default=DEFAULT_GRIPPER_COLUMN)
    parser.add_argument("--pose-column", default=DEFAULT_POSE_COLUMN)
    parser.add_argument("--tcp-pose-column", default=DEFAULT_TCP_POSE_COLUMN)
    parser.add_argument("--gt-contact-column", default=DEFAULT_ANNOTATION_CONTACT_COLUMN)
    parser.add_argument("--gt-state-affordance-column", default=DEFAULT_STATE_AFFORDANCE_COLUMN)
    parser.add_argument("--grasp-threshold", type=float, default=0.4)
    parser.add_argument("--contact-threshold", type=float, default=0.9)
    parser.add_argument("--tolerances", default="0,1,3,5,10")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquets = sorted(args.test_data.glob(args.pattern))
    if not parquets:
        raise FileNotFoundError(f"No parquet files matched {args.test_data / args.pattern}")

    results = [
        evaluate_episode(
            parquet_path=parquet_path,
            grasp_threshold=args.grasp_threshold,
            contact_threshold=args.contact_threshold,
            frame_column=args.frame_column,
            gripper_column=args.gripper_column,
            pose_column=args.pose_column,
            tcp_pose_column=args.tcp_pose_column,
            gt_contact_column=args.gt_contact_column,
            gt_state_affordance_column=args.gt_state_affordance_column,
        )
        for parquet_path in parquets
    ]

    tolerances = parse_tolerances(args.tolerances)
    summary = summarize(results, tolerances)
    report = {
        "test_data": str(args.test_data),
        "pattern": args.pattern,
        "grasp_threshold": args.grasp_threshold,
        "contact_threshold": args.contact_threshold,
        "metrics": {
            "primary": "first contact frame absolute error and tolerance accuracy",
            "secondary": "TCP pose at predicted contact compared with annotation.state_affordance",
            "notes": (
                "annotation.state_affordance matches observation_tcp_pose6d at the "
                "annotated contact frame in the provided test data."
            ),
        },
        "summary": summary,
        "episodes": results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = threshold_tag(args.grasp_threshold, args.contact_threshold)
    report_path = args.output_dir / f"phase_segmentation_eval_report__{tag}.json"
    csv_path = args.output_dir / f"phase_segmentation_eval_per_episode__{tag}.csv"
    report_path.write_text(json.dumps(json_ready(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv_with_summary(csv_path, results, summary)

    frame_metrics = summary["contact_frame"]
    pose_metrics = summary["contact_pose_vs_state_affordance"]
    print(f"Evaluated {summary['num_with_gt_and_prediction']}/{summary['num_episodes']} episodes")
    print(f"Contact frame MAE: {frame_metrics['mae_frames']:.3f} frames")
    print(f"Contact frame RMSE: {frame_metrics['rmse_frames']:.3f} frames")
    for name, value in frame_metrics["accuracy_at_tolerance"].items():
        print(f"{name}: {value:.3f}")
    print(f"Mean TCP position L2: {pose_metrics['mean_tcp_position_l2_m']:.6f} m")
    print(f"Wrote report: {report_path}")
    print(f"Wrote per-episode CSV: {csv_path}")


if __name__ == "__main__":
    main()
