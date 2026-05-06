#!/usr/bin/env python3
"""Generate per-frame gripper phases and first-contact pose for one episode."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_VIDEO = PROJECT_ROOT / "episode_000000.mp4"
DEFAULT_PARQUET = PROJECT_ROOT / "episode_000000.parquet"
DEFAULT_OUTPUT = SCRIPT_DIR / "episode_000000_gripper_phases.json"
DEFAULT_FRAME_COLUMN = "frame_index"
DEFAULT_GRIPPER_COLUMN = "other_information.observation_gripper_position"
DEFAULT_POSE_COLUMN = "other_information.observation_gripper_pose6d"
DEFAULT_TCP_POSE_COLUMN = "other_information.observation_tcp_pose6d"
DEFAULT_ANNOTATION_CONTACT_COLUMN = "annotation.contact_frame"


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


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        return json_ready(value.tolist())
    if hasattr(value, "item"):
        try:
            return json_ready(value.item())
        except Exception:
            pass
    return value


def probe_video_frame_count(video_path: Path) -> int | None:
    if shutil.which("ffprobe") is None:
        return None

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception:
        return None

    text = result.stdout.strip()
    return int(text) if text.isdigit() else None


def extract_series(rows: list[dict[str, Any]], frame_column: str, gripper_column: str) -> tuple[list[int], list[float]]:
    frame_indices = []
    gripper_positions = []
    for row in rows:
        if frame_column not in row:
            raise KeyError(f"Frame column '{frame_column}' not found in parquet")
        if gripper_column not in row:
            raise KeyError(f"Gripper column '{gripper_column}' not found in parquet")
        frame_indices.append(int(row[frame_column]))
        gripper_positions.append(float(row[gripper_column]))
    order = np.argsort(frame_indices)
    return [frame_indices[i] for i in order], [gripper_positions[i] for i in order]


def normalize_gripper(gripper_positions: list[float]) -> list[float]:
    arr = np.asarray(gripper_positions, dtype=float)
    min_value = float(arr.min())
    max_value = float(arr.max())
    return ((arr - min_value) / (max_value - min_value + 1e-6)).tolist()


def determine_grasp_phases(
    normalized_gripper_actions: list[float],
    grasp_threshold: float = 0.4,
    contact_threshold: float = 0.9,
) -> dict[str, Any]:
    """Robo2VLM-style phase segmentation from normalized gripper positions."""
    traj_length = len(normalized_gripper_actions)
    phases = []
    phase_ranges = {
        "pre_grasp": [],
        "immobilization": [],
        "contact": [],
        "detach": [],
        "post_grasp": [],
        "transition": [],
    }
    current_phase = "pre_grasp"
    phase_start_idx = 0
    prev_gripper_position = None

    for step_idx, gripper_position in enumerate(normalized_gripper_actions):
        if prev_gripper_position is None:
            prev_gripper_position = gripper_position

        gripper_closing = gripper_position > prev_gripper_position
        gripper_opening = gripper_position < prev_gripper_position

        if current_phase == "pre_grasp":
            if gripper_position > grasp_threshold and gripper_closing:
                if step_idx > phase_start_idx:
                    phase_ranges["pre_grasp"].append((phase_start_idx, step_idx - 1))
                    phases.append((phase_start_idx, step_idx - 1, "pre_grasp"))
                current_phase = "immobilization"
                phase_start_idx = step_idx

        elif current_phase == "immobilization":
            if gripper_position >= contact_threshold:
                if step_idx > phase_start_idx:
                    phase_ranges["immobilization"].append((phase_start_idx, step_idx - 1))
                    phases.append((phase_start_idx, step_idx - 1, "immobilization"))
                current_phase = "contact"
                phase_start_idx = step_idx
            elif gripper_position <= grasp_threshold and gripper_opening:
                if step_idx > phase_start_idx:
                    phase_ranges["immobilization"].append((phase_start_idx, step_idx - 1))
                    phases.append((phase_start_idx, step_idx - 1, "immobilization"))
                current_phase = "pre_grasp"
                phase_start_idx = step_idx

        elif current_phase == "contact":
            if gripper_position < contact_threshold and gripper_opening:
                if step_idx > phase_start_idx:
                    phase_ranges["contact"].append((phase_start_idx, step_idx - 1))
                    phases.append((phase_start_idx, step_idx - 1, "contact"))
                current_phase = "detach"
                phase_start_idx = step_idx
            elif gripper_position < contact_threshold and gripper_closing:
                phase_ranges["transition"].append((step_idx, step_idx))
                phases.append((step_idx, step_idx, "transition"))

        elif current_phase == "detach":
            if gripper_position <= grasp_threshold:
                if step_idx > phase_start_idx:
                    phase_ranges["detach"].append((phase_start_idx, step_idx - 1))
                    phases.append((phase_start_idx, step_idx - 1, "detach"))
                current_phase = "post_grasp"
                phase_start_idx = step_idx
            elif gripper_position >= contact_threshold and gripper_closing:
                if step_idx > phase_start_idx:
                    phase_ranges["detach"].append((phase_start_idx, step_idx - 1))
                    phases.append((phase_start_idx, step_idx - 1, "detach"))
                current_phase = "contact"
                phase_start_idx = step_idx

        elif current_phase == "post_grasp":
            if gripper_position > grasp_threshold and gripper_closing:
                if step_idx > phase_start_idx:
                    phase_ranges["post_grasp"].append((phase_start_idx, step_idx - 1))
                    phases.append((phase_start_idx, step_idx - 1, "post_grasp"))
                current_phase = "immobilization"
                phase_start_idx = step_idx

        prev_gripper_position = gripper_position

    if phase_start_idx < traj_length:
        phase_ranges[current_phase].append((phase_start_idx, traj_length - 1))
        phases.append((phase_start_idx, traj_length - 1, current_phase))

    return {"phases": phases, "phase_ranges": phase_ranges}


def build_phase_by_step(phases: dict[str, Any], length: int) -> list[str]:
    phase_by_step = ["unknown"] * length
    for start_idx, end_idx, phase_name in phases["phases"]:
        for step_idx in range(start_idx, end_idx + 1):
            if 0 <= step_idx < length:
                phase_by_step[step_idx] = phase_name
    return phase_by_step


def get_row_by_frame(rows: list[dict[str, Any]], frame_column: str) -> dict[int, dict[str, Any]]:
    return {int(row[frame_column]): row for row in rows}


def first_frame_for_phase(frame_indices: list[int], phase_by_step: list[str], phase: str) -> int | None:
    for frame_idx, phase_name in zip(frame_indices, phase_by_step):
        if phase_name == phase:
            return frame_idx
    return None


def get_pose(row: dict[str, Any], pose_column: str) -> list[float] | None:
    value = row.get(pose_column)
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    return arr.tolist()


def get_reference_contact_frame(rows: list[dict[str, Any]], column: str) -> int | None:
    for row in rows:
        value = row.get(column)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        try:
            return int(float(text))
        except ValueError:
            continue
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Robo2VLM-style per-frame grasp phases, first contact frame, "
            "and the gripper contact pose for a DROID/LeRobot episode."
        )
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame-column", default=DEFAULT_FRAME_COLUMN)
    parser.add_argument("--gripper-column", default=DEFAULT_GRIPPER_COLUMN)
    parser.add_argument("--pose-column", default=DEFAULT_POSE_COLUMN)
    parser.add_argument("--tcp-pose-column", default=DEFAULT_TCP_POSE_COLUMN)
    parser.add_argument("--annotation-contact-column", default=DEFAULT_ANNOTATION_CONTACT_COLUMN)
    parser.add_argument("--grasp-threshold", type=float, default=0.4)
    parser.add_argument("--contact-threshold", type=float, default=0.9)
    parser.add_argument(
        "--phase-alias-style",
        choices=("robo2vlm", "hyphen"),
        default="hyphen",
        help="Use Robo2VLM phase names or hyphenated display names.",
    )
    return parser.parse_args()


def phase_alias(phase: str, style: str) -> str:
    if style == "robo2vlm":
        return phase
    aliases = {
        "pre_grasp": "pre-grasp",
        "immobilization": "approach",
        "contact": "contact",
        "detach": "detach",
        "post_grasp": "post-grasp",
        "transition": "transition",
    }
    return aliases.get(phase, phase.replace("_", "-"))


def main() -> None:
    args = parse_args()
    rows = load_rows(args.parquet)
    if not rows:
        raise ValueError(f"{args.parquet} is empty")

    frame_indices, gripper_positions = extract_series(rows, args.frame_column, args.gripper_column)
    normalized_gripper = normalize_gripper(gripper_positions)
    phases = determine_grasp_phases(
        normalized_gripper_actions=normalized_gripper,
        grasp_threshold=args.grasp_threshold,
        contact_threshold=args.contact_threshold,
    )
    phase_by_step = build_phase_by_step(phases, len(frame_indices))
    row_by_frame = get_row_by_frame(rows, args.frame_column)

    first_contact_frame = first_frame_for_phase(frame_indices, phase_by_step, "contact")
    reference_contact_frame = get_reference_contact_frame(rows, args.annotation_contact_column)
    contact_pose = None
    contact_tcp_pose = None
    contact_row = None
    if first_contact_frame is not None:
        contact_row = row_by_frame[first_contact_frame]
        contact_pose = get_pose(contact_row, args.pose_column)
        contact_tcp_pose = get_pose(contact_row, args.tcp_pose_column)

    frame_records = []
    for step_idx, (frame_idx, raw_gripper, norm_gripper, phase_name) in enumerate(
        zip(frame_indices, gripper_positions, normalized_gripper, phase_by_step)
    ):
        frame_records.append(
            {
                "step_index": step_idx,
                "frame_index": frame_idx,
                "phase": phase_alias(phase_name, args.phase_alias_style),
                "robo2vlm_phase": phase_name,
                "gripper_position_raw": raw_gripper,
                "gripper_position_normalized": norm_gripper,
            }
        )

    payload = {
        "video": str(args.video),
        "parquet": str(args.parquet),
        "video_frame_count": probe_video_frame_count(args.video),
        "parquet_frame_count": len(rows),
        "frame_column": args.frame_column,
        "gripper_column": args.gripper_column,
        "pose_column": args.pose_column,
        "tcp_pose_column": args.tcp_pose_column,
        "annotation_contact_column": args.annotation_contact_column,
        "grasp_threshold": args.grasp_threshold,
        "contact_threshold": args.contact_threshold,
        "gripper_raw_min": min(gripper_positions),
        "gripper_raw_max": max(gripper_positions),
        "phase_ranges": {
            phase: [[int(start), int(end)] for start, end in ranges]
            for phase, ranges in phases["phase_ranges"].items()
        },
        "phases": [
            {"start_step": int(start), "end_step": int(end), "phase": phase}
            for start, end, phase in phases["phases"]
        ],
        "first_contact_frame_index": first_contact_frame,
        "reference_annotation_contact_frame_index": reference_contact_frame,
        "contact_pose": {
            "frame_index": first_contact_frame,
            "pose_column": args.pose_column,
            "gripper_pose6d": contact_pose,
            "tcp_pose_column": args.tcp_pose_column if contact_tcp_pose is not None else None,
            "tcp_pose6d": contact_tcp_pose,
            "gripper_position_raw": contact_row.get(args.gripper_column) if contact_row else None,
            "gripper_position_normalized": (
                normalized_gripper[frame_indices.index(first_contact_frame)]
                if first_contact_frame is not None
                else None
            ),
        }
        if first_contact_frame is not None
        else None,
        "frames": frame_records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(frame_records)} frame phase records to {args.output}")
    if first_contact_frame is None:
        print("No contact phase was detected.")
    else:
        print(f"First contact frame: {first_contact_frame}")
        print(f"Contact gripper pose6d: {contact_pose}")


if __name__ == "__main__":
    main()
