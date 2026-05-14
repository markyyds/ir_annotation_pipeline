#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from video_object_detection_mapper import common, vlm


DEFAULT_TEST_DATA = PROJECT_ROOT / "test_data"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"
DEFAULT_TASK_INSTRUCTION_COLUMN = "other_information.language_instruction_2"


def run_vlm_stage(args: argparse.Namespace) -> dict:
    np, torch, imageio, Image, _ImageDraw = common.load_common_modules()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    task_instruction = common.get_task_instruction(args.parquet, args.task_instruction_column)
    first_frame_path, last_frame_path, frame_width, frame_height, first_idx, last_idx = common.save_context_frames(
        args,
        imageio,
        Image,
    )

    started = time.perf_counter()
    target = vlm.extract_target_and_referring_expression(
        vlm.build_vllm_json_client(args),
        task_instruction,
        first_frame_path,
        last_frame_path,
        frame_width,
        frame_height,
    )
    timing = {"vlm_seconds": time.perf_counter() - started}

    payload = {
        "status": "ok",
        "stage": "vlm_target_referring",
        "video_path": str(args.video),
        "parquet_path": str(args.parquet),
        "task_instruction": task_instruction,
        "vlm_model": args.vlm_model,
        "vllm_base_url": args.vllm_base_url,
        "generation": {
            "max_tokens": args.vlm_max_new_tokens,
            "temperature": args.vlm_temperature,
            "top_p": args.vlm_top_p,
            "top_k": args.vlm_top_k,
            "min_p": args.vlm_min_p,
            "presence_penalty": args.vlm_presence_penalty,
            "repetition_penalty": args.vlm_repetition_penalty,
        },
        "frame_context": {
            "first_frame_path": str(first_frame_path),
            "last_frame_path": str(last_frame_path),
            "first_video_frame_index": first_idx,
            "last_video_frame_index": last_idx,
            "width": frame_width,
            "height": frame_height,
        },
        "vlm_target": target,
        "timing_seconds": timing,
    }
    output_json = args.output_dir / "vlm_stage.json"
    output_json.write_text(json.dumps(common.json_ready(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote VLM stage output: {output_json}")
    print(f"Target object: {target['target_object']}")
    print(f"Referring expression: {target['referring_expression']}")
    return {"output_json": output_json, "payload": payload}


def safe_name(value: str) -> str:
    safe = []
    for char in str(value):
        if char.isalnum() or char in ("-", "_", "."):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "none"


def default_output_dir(vlm_model: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / safe_name(vlm_model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run only the VLM target/referring-expression stage and save it for reuse.")
    parser.add_argument("--video", type=Path, default=DEFAULT_TEST_DATA / "episode_000000.mp4")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_TEST_DATA / "episode_000000.parquet")
    parser.add_argument("--output-dir", type=Path, help="Defaults to video_object_detection_mapper/outputs/{vlm_model}.")
    parser.add_argument("--task-instruction-column", default=DEFAULT_TASK_INSTRUCTION_COLUMN)
    parser.add_argument("--first-video-frame-index", type=int, default=0)
    parser.add_argument("--last-video-frame-index", type=int, default=-1)

    parser.add_argument("--vlm-model", default=os.environ.get("MODEL_NAME", "qwen3-max"))
    parser.add_argument("--vllm-base-url", default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--vllm-api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--vlm-max-new-tokens", type=int, default=2048)
    parser.add_argument("--vlm-temperature", type=float, default=0.0)
    parser.add_argument("--vlm-top-p", type=float, default=0.95)
    parser.add_argument("--vlm-top-k", type=int, default=20)
    parser.add_argument("--vlm-min-p", type=float, default=0.0)
    parser.add_argument("--vlm-presence-penalty", type=float, default=0.0)
    parser.add_argument("--vlm-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--vlm-timeout", type=int, default=300)
    parser.add_argument("--print-raw-response", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.vlm_model)
    run_vlm_stage(args)


if __name__ == "__main__":
    main()
