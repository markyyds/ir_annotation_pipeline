#!/usr/bin/env python3
"""Generate VLM-based subtask instructions without annotation/Q_annotation fields."""

from __future__ import annotations

import argparse
import base64
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_TEST_DATA = PROJECT_ROOT / "test_data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_FRAME_COLUMN = "frame_index"
DEFAULT_TASK_INSTRUCTION_COLUMN = "other_information.language_instruction_2"
DEFAULT_GRIPPER_POSE_COLUMN = "other_information.observation_gripper_pose6d"
DEFAULT_TCP_POSE_COLUMN = "other_information.observation_tcp_pose6d"


@dataclass
class Segment:
    segment_id: str
    start_frame: int
    end_frame: int
    subtask_instruction: str
    instruction_source: str


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


def parse_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text in {"", "[]", "-1"} else text


def sorted_rows(rows: list[dict[str, Any]], frame_column: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: int(row[frame_column]))


def get_task_instruction(rows: list[dict[str, Any]], column: str) -> str:
    if not rows:
        return ""
    text = parse_text(rows[0].get(column))
    if not text:
        raise KeyError(f"Task instruction column '{column}' is missing or empty")
    return text


def build_row_by_frame(rows: list[dict[str, Any]], frame_column: str) -> dict[int, dict[str, Any]]:
    return {int(row[frame_column]): row for row in rows}


def pose_at(row_by_frame: dict[int, dict[str, Any]], frame_idx: int, column: str) -> list[float] | None:
    row = row_by_frame.get(frame_idx)
    if row is None or column not in row:
        return None
    value = row[column]
    if value is None:
        return None
    return np.asarray(value, dtype=float).reshape(-1).tolist()


def sample_frame_indices(start_frame: int, end_frame: int, samples: int) -> list[int]:
    if samples <= 1 or start_frame == end_frame:
        return [start_frame]
    return sorted(set(np.linspace(start_frame, end_frame, samples, dtype=int).tolist()))


def open_video_reader(video_path: Path):
    try:
        import imageio.v2 as imageio

        return imageio.get_reader(str(video_path))
    except ValueError as exc:
        raise RuntimeError(
            "Could not open video with imageio. Install video support with: "
            "python -m pip install 'imageio[ffmpeg]'"
        ) from exc


def save_frame(reader, frame_idx: int, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(reader.get_data(frame_idx)).convert("RGB").save(output_path, quality=95)
    return output_path


def clamp_frame(frame_idx: int, min_frame: int, max_frame: int) -> int:
    return max(min_frame, min(max_frame, int(frame_idx)))


def normalize_segments(raw_segments: list[dict[str, Any]], first_frame: int, last_frame: int) -> list[Segment]:
    segments: list[Segment] = []
    for raw in raw_segments:
        try:
            start_frame = clamp_frame(int(raw["start_frame"]), first_frame, last_frame)
            end_frame = clamp_frame(int(raw["end_frame"]), first_frame, last_frame)
        except Exception:
            continue
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame
        instruction = parse_text(raw.get("subtask_instruction")) or "complete this part of the task"
        segments.append(
            Segment(
                segment_id=f"segment_{len(segments):03d}",
                start_frame=start_frame,
                end_frame=end_frame,
                subtask_instruction=instruction,
                instruction_source="vlm",
            )
        )

    segments.sort(key=lambda segment: (segment.start_frame, segment.end_frame))
    cleaned: list[Segment] = []
    cursor = first_frame
    for segment in segments:
        start_frame = max(segment.start_frame, cursor)
        end_frame = max(start_frame, segment.end_frame)
        if start_frame > last_frame:
            break
        cleaned.append(
            Segment(
                segment_id=f"segment_{len(cleaned):03d}",
                start_frame=start_frame,
                end_frame=min(end_frame, last_frame),
                subtask_instruction=segment.subtask_instruction,
                instruction_source=segment.instruction_source,
            )
        )
        cursor = min(end_frame + 1, last_frame + 1)

    if cleaned and cleaned[0].start_frame > first_frame:
        cleaned[0].start_frame = first_frame
    if cleaned and cleaned[-1].end_frame < last_frame:
        cleaned[-1].end_frame = last_frame
    return cleaned


def fallback_segments(task_instruction: str, first_frame: int, last_frame: int) -> list[Segment]:
    return [
        Segment(
            segment_id="segment_000",
            start_frame=first_frame,
            end_frame=last_frame,
            subtask_instruction=task_instruction,
            instruction_source="single_segment_fallback",
        )
    ]


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def extract_json_object(text: str) -> dict[str, Any]:
    text = strip_thinking(text)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def encode_image_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def assistant_text(response: dict[str, Any]) -> str:
    message = response["choices"][0]["message"]
    content = message.get("content")
    if content:
        return str(content)
    reasoning = message.get("reasoning")
    if reasoning:
        return str(reasoning)
    return ""


def build_planning_prompt(
    task_instruction: str,
    sampled_frames: list[tuple[int, Path]],
    first_frame: int,
    last_frame: int,
) -> str:
    frame_list = ", ".join(str(frame_idx) for frame_idx, _ in sampled_frames)
    return (
        "You are a robot manipulation video annotator. "
        "You will see ordered frames sampled from one episode. "
        "Use only the visual evidence and the task instruction. "
        "Do not assume hidden labels or parquet annotations. "
        "Return JSON only with this exact schema:\n"
        "{\"subtask_segments\": ["
        "{\"start_frame\": 0, \"end_frame\": 10, "
        "\"subtask_instruction\": \"short imperative phrase\"}"
        "]}\n"
        f"Task instruction: {task_instruction}\n"
        f"Episode frame range: {first_frame} to {last_frame}\n"
        f"Sampled frame indices in order: {frame_list}\n"
        "Make 1 to 5 non-overlapping contiguous segments. "
        "Use actual frame indices, not ordinal sample numbers. "
        "The first segment must start at the episode's first frame. "
        "The final segment must end at the episode's last frame."
    )


class VLLMSubtaskPlanner:
    """Planner for a local vLLM OpenAI-compatible VLM endpoint."""

    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str,
        max_tokens: int,
        temperature: float,
    ):
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(self, messages: list[dict[str, Any]]) -> str:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"VLM API request failed with HTTP {exc.code}: {body}") from exc
        return assistant_text(data)

    def plan(
        self,
        task_instruction: str,
        sampled_frames: list[tuple[int, Path]],
        first_frame: int,
        last_frame: int,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": build_planning_prompt(task_instruction, sampled_frames, first_frame, last_frame),
            }
        ]
        for _, path in sampled_frames:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image_data_url(path)},
                }
            )
        raw = self.chat([{"role": "user", "content": content}])
        parsed = extract_json_object(raw)
        segments = parsed.get("subtask_segments", [])
        return segments if isinstance(segments, list) else []


def maybe_make_vlm(args: argparse.Namespace) -> VLLMSubtaskPlanner | None:
    if not args.use_vlm:
        return None
    return VLLMSubtaskPlanner(
        model_id=args.vlm_model,
        base_url=args.vlm_base_url,
        api_key=args.vlm_api_key,
        max_tokens=args.vlm_max_new_tokens,
        temperature=args.vlm_temperature,
    )


def process_episode(
    video_path: Path,
    parquet_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    vlm: VLLMSubtaskPlanner | None,
) -> dict[str, Any]:
    episode_id = parquet_path.stem
    episode_dir = output_dir / episode_id
    frame_dir = episode_dir / "frames"
    subgoal_dir = episode_dir / "subgoal_images"
    rows = sorted_rows(load_rows(parquet_path), args.frame_column)
    if not rows:
        raise ValueError(f"{parquet_path} is empty")

    first_frame = int(rows[0][args.frame_column])
    last_frame = int(rows[-1][args.frame_column])
    row_by_frame = build_row_by_frame(rows, args.frame_column)
    task_instruction = get_task_instruction(rows, args.task_instruction_column)

    reader = open_video_reader(video_path)
    try:
        sampled_indices = sample_frame_indices(first_frame, last_frame, args.vlm_frame_samples)
        sampled_frames = [
            (
                frame_idx,
                save_frame(reader, frame_idx, frame_dir / "vlm_episode_samples" / f"frame_{frame_idx:06d}.jpg"),
            )
            for frame_idx in sampled_indices
        ]

        if vlm is None:
            segments = fallback_segments(task_instruction, first_frame, last_frame)
        else:
            raw_segments = vlm.plan(task_instruction, sampled_frames, first_frame, last_frame)
            segments = normalize_segments(raw_segments, first_frame, last_frame)
            if not segments:
                segments = fallback_segments(task_instruction, first_frame, last_frame)

        segment_records = []
        for segment in segments:
            subgoal_frame = segment.end_frame
            subgoal_image_path = save_frame(
                reader,
                subgoal_frame,
                subgoal_dir / f"{segment.segment_id}_subgoal_frame_{subgoal_frame:06d}.jpg",
            )
            segment_records.append(
                {
                    "segment_id": segment.segment_id,
                    "frame_indices": [segment.start_frame, segment.end_frame],
                    "start_frame": segment.start_frame,
                    "end_frame": segment.end_frame,
                    "subtask_instruction": segment.subtask_instruction,
                    "instruction_source": segment.instruction_source,
                    "subgoal_image_frame": subgoal_frame,
                    "subgoal_image_path": str(subgoal_image_path),
                    "subgoal_image_gripper_pose": pose_at(row_by_frame, subgoal_frame, args.gripper_pose_column),
                    "subgoal_image_tcp_pose": pose_at(row_by_frame, subgoal_frame, args.tcp_pose_column),
                }
            )

        goal_image_path = save_frame(reader, last_frame, episode_dir / f"goal_frame_{last_frame:06d}.jpg")
    finally:
        reader.close()

    payload = {
        "episode_id": episode_id,
        "video": str(video_path),
        "parquet": str(parquet_path),
        "task_instruction_column": args.task_instruction_column,
        "task_instruction": task_instruction,
        "annotation_fields_used": False,
        "segment_source": "vlm" if vlm is not None else "single_segment_fallback",
        "vlm_sample_frame_indices": sampled_indices,
        "vlm_sample_frame_paths": [str(path) for _, path in sampled_frames],
        "subtask_segments": segment_records,
        "goal_image_frame": last_frame,
        "goal_image_path": str(goal_image_path),
        "goal_image_gripper_pose": pose_at(row_by_frame, last_frame, args.gripper_pose_column),
        "goal_image_tcp_pose": pose_at(row_by_frame, last_frame, args.tcp_pose_column),
    }
    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "subtask_instruction_outputs.json").write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate subtask annotations using only video and "
            "other_information.language_instruction_2 from parquet."
        )
    )
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern", default="episode_*.parquet")
    parser.add_argument("--frame-column", default=DEFAULT_FRAME_COLUMN)
    parser.add_argument("--task-instruction-column", default=DEFAULT_TASK_INSTRUCTION_COLUMN)
    parser.add_argument("--gripper-pose-column", default=DEFAULT_GRIPPER_POSE_COLUMN)
    parser.add_argument("--tcp-pose-column", default=DEFAULT_TCP_POSE_COLUMN)
    parser.add_argument("--vlm-frame-samples", type=int, default=12)
    parser.add_argument("--use-vlm", action="store_true")
    parser.add_argument("--vlm-model", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--vlm-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--vlm-api-key", default="EMPTY")
    parser.add_argument("--vlm-max-new-tokens", type=int, default=512)
    parser.add_argument("--vlm-temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.vlm_frame_samples <= 0:
        raise ValueError("--vlm-frame-samples must be positive")
    parquets = sorted(args.test_data.glob(args.pattern))
    if not parquets:
        raise FileNotFoundError(f"No parquets matched {args.test_data / args.pattern}")

    vlm = maybe_make_vlm(args)
    summaries = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for parquet_path in parquets:
        video_path = args.test_data / f"{parquet_path.stem}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video for {parquet_path}: {video_path}")
        print(f"[{parquet_path.stem}] generating subtask instructions")
        payload = process_episode(video_path, parquet_path, args.output_dir, args, vlm)
        summaries.append(
            {
                "episode_id": payload["episode_id"],
                "num_segments": len(payload["subtask_segments"]),
                "task_instruction": payload["task_instruction"],
                "segment_source": payload["segment_source"],
                "annotation_fields_used": False,
                "output_path": str(args.output_dir / payload["episode_id"] / "subtask_instruction_outputs.json"),
            }
        )

    summary_path = args.output_dir / "subtask_instruction_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
