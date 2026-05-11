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
DEFAULT_PARQUET_PATTERN = "episode_*.parquet"
DEFAULT_STATE_AFFORDANCE_COLUMN = "annotation.state_affordance"
DEFAULT_OBJECT_BOX_KEY = "annotation.object_box"
DEFAULT_GRIPPER_BOX_KEY = "annotation.gripper_box"
VISUAL_SCORE_WEIGHTS = {
    "proximity": 0.0,
    "overlap": 0.571,
    "target_motion": 0.286,
    "base_prior": 0.143,
}


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


def parse_box(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text in {"[]", "-1", "None", "nan"}:
            return None
        try:
            value = ast.literal_eval(text)
        except Exception:
            try:
                value = json.loads(text)
            except Exception:
                return None

    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(point, (list, tuple)) and len(point) >= 2 for point in value)
    ):
        x1, y1 = float(value[0][0]), float(value[0][1])
        x2, y2 = float(value[1][0]), float(value[1][1])
    elif isinstance(value, (list, tuple)) and len(value) >= 4:
        x1, y1, x2, y2 = [float(item) for item in value[:4]]
    else:
        return None

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def box_diag(box: list[float]) -> float:
    return max(1.0, math.hypot(box[2] - box[0], box[3] - box[1]))


def box_edge_distance(box_a: list[float], box_b: list[float]) -> float:
    dx = max(box_b[0] - box_a[2], box_a[0] - box_b[2], 0.0)
    dy = max(box_b[1] - box_a[3], box_a[1] - box_b[3], 0.0)
    return float(math.hypot(dx, dy))


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    inter_w = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    inter_h = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    inter_area = inter_w * inter_h
    union_area = box_area(box_a) + box_area(box_b) - inter_area
    return float(inter_area / union_area) if union_area > 0 else 0.0


def target_motion_score(
    frame_idx: int,
    object_boxes: dict[int, list[float]],
    future_window: int,
) -> float:
    current_box = object_boxes.get(frame_idx)
    if current_box is None:
        return 0.0

    current_center = box_center(current_box)
    current_area = max(1.0, box_area(current_box))
    current_diag = box_diag(current_box)
    motions = []
    for offset in range(1, future_window + 1):
        future_box = object_boxes.get(frame_idx + offset)
        if future_box is None:
            continue
        future_center = box_center(future_box)
        center_delta = math.hypot(
            future_center[0] - current_center[0],
            future_center[1] - current_center[1],
        ) / current_diag
        area_delta = abs(box_area(future_box) - current_area) / current_area
        motions.append(center_delta + 0.35 * area_delta)

    if not motions:
        return 0.0
    return min(1.0, max(motions) / 0.08)


def visual_contact_candidate(
    frame_idx: int,
    base_frame: int,
    radius: int,
    object_boxes: dict[int, list[float]],
    gripper_boxes: dict[int, list[float]],
    future_window: int,
) -> dict[str, Any]:
    object_box = object_boxes.get(frame_idx)
    gripper_box = gripper_boxes.get(frame_idx)

    distance_px = None
    iou = None
    proximity_score = 0.0
    overlap_score = 0.0
    if object_box is not None and gripper_box is not None:
        distance_px = box_edge_distance(object_box, gripper_box)
        iou = box_iou(object_box, gripper_box)
        proximity_score = math.exp(-distance_px / (0.12 * box_diag(object_box)))
        overlap_score = min(1.0, iou / 0.2)

    motion_score = target_motion_score(frame_idx, object_boxes, future_window)
    base_prior_score = max(0.0, 1.0 - abs(frame_idx - base_frame) / max(1, radius))
    score = (
        VISUAL_SCORE_WEIGHTS["proximity"] * proximity_score
        + VISUAL_SCORE_WEIGHTS["overlap"] * overlap_score
        + VISUAL_SCORE_WEIGHTS["target_motion"] * motion_score
        + VISUAL_SCORE_WEIGHTS["base_prior"] * base_prior_score
    )

    return {
        "frame": int(frame_idx),
        "score": float(score),
        "distance_px": distance_px,
        "iou": iou,
        "proximity_score": float(proximity_score),
        "overlap_score": float(overlap_score),
        "target_motion_score": float(motion_score),
        "base_prior_score": float(base_prior_score),
        "has_object_box": object_box is not None,
        "has_gripper_box": gripper_box is not None,
    }


def correct_contact_with_visual_boxes(
    rows: list[dict[str, Any]],
    frame_indices: list[int],
    frame_column: str,
    base_contact_frame: int | None,
    object_box_key: str,
    gripper_box_key: str,
    window_before: int,
    window_after: int,
    future_window: int,
    min_score_margin: float,
    min_backward_shift: int,
) -> tuple[int | None, dict[str, Any]]:
    if base_contact_frame is None:
        return None, {"enabled": True, "corrected": False, "reason": "missing_base_contact_frame"}

    object_boxes = {
        frame_idx: box
        for row in rows
        if (frame_idx := int(row[frame_column])) in frame_indices
        if (box := parse_box(row.get(object_box_key))) is not None
    }
    gripper_boxes = {
        frame_idx: box
        for row in rows
        if (frame_idx := int(row[frame_column])) in frame_indices
        if (box := parse_box(row.get(gripper_box_key))) is not None
    }
    if not object_boxes:
        return base_contact_frame, {
            "enabled": True,
            "corrected": False,
            "reason": "missing_object_boxes",
        }

    start_frame = base_contact_frame - window_before
    end_frame = base_contact_frame + window_after
    candidate_frames = [frame for frame in frame_indices if start_frame <= frame <= end_frame]
    if not candidate_frames:
        return base_contact_frame, {
            "enabled": True,
            "corrected": False,
            "reason": "empty_search_window",
        }

    radius = max(window_before, window_after, 1)
    candidates = [
        visual_contact_candidate(
            frame_idx=frame,
            base_frame=base_contact_frame,
            radius=radius,
            object_boxes=object_boxes,
            gripper_boxes=gripper_boxes,
            future_window=future_window,
        )
        for frame in candidate_frames
    ]
    eligible_candidates = [
        item for item in candidates if base_contact_frame - item["frame"] >= min_backward_shift
    ]
    best_candidate = max(eligible_candidates, key=lambda item: item["score"]) if eligible_candidates else None
    base_candidates = [item for item in candidates if item["frame"] == base_contact_frame]
    base_candidate = base_candidates[0] if base_candidates else None
    top_candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)[:5]

    corrected = True
    reason = "visual_score"
    final_candidate = best_candidate or base_candidate or max(candidates, key=lambda item: item["score"])
    if best_candidate is None:
        corrected = False
        reason = "min_backward_shift_not_met"
    elif base_candidate is not None and best_candidate["score"] < base_candidate["score"] + min_score_margin:
        corrected = False
        reason = "score_margin_not_met"
        final_candidate = base_candidate

    final_frame = int(final_candidate["frame"])
    return final_frame, {
        "enabled": True,
        "corrected": corrected and final_frame != base_contact_frame,
        "reason": reason,
        "bbox_source": "parquet",
        "window": [int(start_frame), int(end_frame)],
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible_candidates),
        "object_box_frame_count": len(object_boxes),
        "gripper_box_frame_count": len(gripper_boxes),
        "base_candidate": base_candidate,
        "best_candidate": best_candidate,
        "top_candidates": top_candidates,
        "selected_candidate": final_candidate,
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
    visual_contact_correction: bool,
    object_box_key: str,
    gripper_box_key: str,
    visual_window_before: int,
    visual_window_after: int,
    visual_future_window: int,
    visual_min_score_margin: float,
    visual_min_backward_shift: int,
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

    base_pred_contact_frame = first_frame_for_phase(frame_indices, phase_by_step, "contact")
    pred_contact_frame = base_pred_contact_frame
    visual_correction = {"enabled": False, "corrected": False, "reason": "disabled"}
    if visual_contact_correction:
        pred_contact_frame, visual_correction = correct_contact_with_visual_boxes(
            rows=rows,
            frame_indices=frame_indices,
            frame_column=frame_column,
            base_contact_frame=base_pred_contact_frame,
            object_box_key=object_box_key,
            gripper_box_key=gripper_box_key,
            window_before=visual_window_before,
            window_after=visual_window_after,
            future_window=visual_future_window,
            min_score_margin=visual_min_score_margin,
            min_backward_shift=visual_min_backward_shift,
        )
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
    base_frame_error = (
        int(base_pred_contact_frame - gt_contact_frame)
        if base_pred_contact_frame is not None and gt_contact_frame is not None
        else None
    )
    base_abs_frame_error = abs(base_frame_error) if base_frame_error is not None else None
    episode_length = len(frame_indices)

    return {
        "episode_id": parquet_path.stem,
        "parquet": str(parquet_path),
        "num_frames": episode_length,
        "gt_contact_frame": gt_contact_frame,
        "base_pred_contact_frame": base_pred_contact_frame,
        "base_frame_error": base_frame_error,
        "base_abs_frame_error": base_abs_frame_error,
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
        "visual_contact_correction": visual_correction,
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
    valid_base_frame = [row for row in results if row.get("base_abs_frame_error") is not None]
    base_abs_errors = [float(row["base_abs_frame_error"]) for row in valid_base_frame]
    base_signed_errors = [float(row["base_frame_error"]) for row in valid_base_frame]
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
        "base_contact_frame": {
            "mae_frames": mean(base_abs_errors),
            "median_abs_error_frames": median(base_abs_errors),
            "rmse_frames": rmse(base_signed_errors),
            "mean_signed_error_frames": mean(base_signed_errors),
            "max_abs_error_frames": max(base_abs_errors) if base_abs_errors else None,
            "accuracy_at_tolerance": {
                f"within_{tol}_frames": accuracy_at(base_abs_errors, tol) for tol in tolerances
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
        "base_pred_contact_frame",
        "base_frame_error",
        "base_abs_frame_error",
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
        "base_pred_contact_frame",
        "base_frame_error",
        "base_abs_frame_error",
        "pred_contact_frame",
        "frame_error",
        "abs_frame_error",
        "normalized_abs_frame_error",
        "tcp_position_l2_error",
        "tcp_rotation_l2_error",
        "tcp_pose6d_l2_error",
        "gripper_raw_min",
        "gripper_raw_max",
        "visual_corrected",
        "visual_reason",
        "visual_selected_score",
        "visual_base_score",
        "metric",
        "value",
    ]
    numeric_fields = [
        "num_frames",
        "gt_contact_frame",
        "base_pred_contact_frame",
        "base_frame_error",
        "base_abs_frame_error",
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
            visual = row.get("visual_contact_correction", {})
            selected = visual.get("selected_candidate") or {}
            base = visual.get("base_candidate") or {}
            output_row = {field: row.get(field) for field in fields}
            output_row.update(
                {
                    "visual_corrected": visual.get("corrected"),
                    "visual_reason": visual.get("reason"),
                    "visual_selected_score": selected.get("score"),
                    "visual_base_score": base.get("score"),
                }
            )
            writer.writerow(output_row)

        writer.writerow({})

        mean_row = {"episode_id": "MEAN"}
        for field in numeric_fields:
            values = [float(row[field]) for row in results if row.get(field) is not None]
            mean_row[field] = mean(values)
        writer.writerow(mean_row)

        writer.writerow({})
        frame_metrics = summary["contact_frame"]
        base_frame_metrics = summary["base_contact_frame"]
        pose_metrics = summary["contact_pose_vs_state_affordance"]
        metric_rows = [
            ("evaluated_episodes", summary["num_with_gt_and_prediction"]),
            ("total_episodes", summary["num_episodes"]),
            ("base_contact_frame_mae_frames", base_frame_metrics["mae_frames"]),
            ("base_contact_frame_rmse_frames", base_frame_metrics["rmse_frames"]),
            ("base_contact_frame_median_abs_error_frames", base_frame_metrics["median_abs_error_frames"]),
            ("base_contact_frame_mean_signed_error_frames", base_frame_metrics["mean_signed_error_frames"]),
            ("base_contact_frame_max_abs_error_frames", base_frame_metrics["max_abs_error_frames"]),
            ("contact_frame_mae_frames", frame_metrics["mae_frames"]),
            ("contact_frame_rmse_frames", frame_metrics["rmse_frames"]),
            ("contact_frame_median_abs_error_frames", frame_metrics["median_abs_error_frames"]),
            ("contact_frame_mean_signed_error_frames", frame_metrics["mean_signed_error_frames"]),
            ("contact_frame_max_abs_error_frames", frame_metrics["max_abs_error_frames"]),
            ("contact_frame_mean_normalized_abs_error", frame_metrics["mean_normalized_abs_error"]),
        ]
        metric_rows.extend(
            (f"base_{name}", value)
            for name, value in base_frame_metrics["accuracy_at_tolerance"].items()
        )
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


def output_tag(args: argparse.Namespace) -> str:
    tag = threshold_tag(args.grasp_threshold, args.contact_threshold)
    if args.visual_contact_correction:
        tag += "__bbox_visual"
    return tag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate predicted first-contact phase against annotation.contact_frame "
            "and predicted TCP pose against annotation.state_affordance."
        )
    )
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frame-column", default=DEFAULT_FRAME_COLUMN)
    parser.add_argument("--gripper-column", default=DEFAULT_GRIPPER_COLUMN)
    parser.add_argument("--pose-column", default=DEFAULT_POSE_COLUMN)
    parser.add_argument("--tcp-pose-column", default=DEFAULT_TCP_POSE_COLUMN)
    parser.add_argument("--gt-contact-column", default=DEFAULT_ANNOTATION_CONTACT_COLUMN)
    parser.add_argument("--gt-state-affordance-column", default=DEFAULT_STATE_AFFORDANCE_COLUMN)
    parser.add_argument("--grasp-threshold", type=float, default=0.4)
    parser.add_argument("--contact-threshold", type=float, default=0.9)
    parser.add_argument("--tolerances", default="0,1,2,3,4,5,6,7,8,9,10,15,20,30")
    parser.add_argument(
        "--visual-contact-correction",
        action="store_true",
        help="Rerank first-contact frame inside a local window using object/gripper bbox features.",
    )
    parser.add_argument("--object-box-key", default=DEFAULT_OBJECT_BOX_KEY)
    parser.add_argument("--gripper-box-key", default=DEFAULT_GRIPPER_BOX_KEY)
    parser.add_argument("--visual-window-before", type=int, default=15)
    parser.add_argument(
        "--visual-window-after",
        type=int,
        default=0,
        help="Frames after the base contact frame to consider. Default 0 keeps correction backward-only.",
    )
    parser.add_argument("--visual-future-window", type=int, default=3)
    parser.add_argument(
        "--visual-min-score-margin",
        type=float,
        default=0.12,
        help="Keep the base contact frame unless the best visual candidate beats it by this score margin.",
    )
    parser.add_argument(
        "--visual-min-backward-shift",
        type=int,
        default=3,
        help="Only allow bbox visual correction to move contact at least this many frames earlier.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquets = sorted(args.test_data.glob(DEFAULT_PARQUET_PATTERN))
    if not parquets:
        raise FileNotFoundError(f"No parquet files matched {args.test_data / DEFAULT_PARQUET_PATTERN}")

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
            visual_contact_correction=args.visual_contact_correction,
            object_box_key=args.object_box_key,
            gripper_box_key=args.gripper_box_key,
            visual_window_before=args.visual_window_before,
            visual_window_after=args.visual_window_after,
            visual_future_window=args.visual_future_window,
            visual_min_score_margin=args.visual_min_score_margin,
            visual_min_backward_shift=args.visual_min_backward_shift,
        )
        for parquet_path in parquets
    ]

    tolerances = parse_tolerances(args.tolerances)
    summary = summarize(results, tolerances)
    report = {
        "test_data": str(args.test_data),
        "parquet_pattern": DEFAULT_PARQUET_PATTERN,
        "grasp_threshold": args.grasp_threshold,
        "contact_threshold": args.contact_threshold,
        "visual_contact_correction": {
            "enabled": args.visual_contact_correction,
            "bbox_source": "parquet",
            "object_box_key": args.object_box_key,
            "gripper_box_key": args.gripper_box_key,
            "window_before": args.visual_window_before,
            "window_after": args.visual_window_after,
            "future_window": args.visual_future_window,
            "min_score_margin": args.visual_min_score_margin,
            "min_backward_shift": args.visual_min_backward_shift,
            "score_weights": VISUAL_SCORE_WEIGHTS,
        },
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
    tag = output_tag(args)
    report_path = args.output_dir / f"phase_segmentation_eval_report__{tag}.json"
    csv_path = args.output_dir / f"phase_segmentation_eval_per_episode__{tag}.csv"
    report_path.write_text(json.dumps(json_ready(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv_with_summary(csv_path, results, summary)

    frame_metrics = summary["contact_frame"]
    base_frame_metrics = summary["base_contact_frame"]
    pose_metrics = summary["contact_pose_vs_state_affordance"]
    print(f"Evaluated {summary['num_with_gt_and_prediction']}/{summary['num_episodes']} episodes")
    if args.visual_contact_correction:
        print(f"Base contact frame MAE: {base_frame_metrics['mae_frames']:.3f} frames")
    print(f"Contact frame MAE: {frame_metrics['mae_frames']:.3f} frames")
    print(f"Contact frame RMSE: {frame_metrics['rmse_frames']:.3f} frames")
    for name, value in frame_metrics["accuracy_at_tolerance"].items():
        print(f"{name}: {value:.3f}")
    print(f"Mean TCP position L2: {pose_metrics['mean_tcp_position_l2_m']:.6f} m")
    print(f"Wrote report: {report_path}")
    print(f"Wrote per-episode CSV: {csv_path}")


if __name__ == "__main__":
    main()
