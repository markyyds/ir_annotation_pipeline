#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from video_object_detection_mapper import common, vlm


DEFAULT_TEST_DATA = PROJECT_ROOT / "test_data"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"
DEFAULT_TASK_INSTRUCTION_COLUMN = "other_information.language_instruction_2"


def print_timer(stage: str, seconds: float) -> None:
    print(f"[timer] {stage}: {seconds:.3f}s", flush=True)


def run_vlm_stage(args: argparse.Namespace) -> dict:
    _np, _torch, imageio, Image, _ImageDraw = common.load_common_modules()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    task_instruction = common.get_task_instruction(args.parquet, args.task_instruction_column)
    started = time.perf_counter()
    first_frame_path, last_frame_path, frame_width, frame_height, first_idx, last_idx = common.save_context_frames(
        args,
        imageio,
        Image,
    )
    print_timer("context_frames", time.perf_counter() - started)

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
    print_timer("vlm", timing["vlm_seconds"])

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


def episode_args(base_args: argparse.Namespace, parquet_path: Path, output_root: Path) -> argparse.Namespace:
    args = argparse.Namespace(**vars(base_args))
    args.parquet = parquet_path
    args.video = parquet_path.with_suffix(".mp4")
    args.output_dir = output_root / parquet_path.stem
    return args


def run_batch(args: argparse.Namespace) -> None:
    if args.single_episode:
        if args.output_dir is None:
            args.output_dir = default_output_dir(args.vlm_model)
        run_vlm_stage(args)
        return

    output_root = args.output_dir or default_output_dir(args.vlm_model)
    output_root.mkdir(parents=True, exist_ok=True)
    parquets = sorted(args.test_data.glob(args.pattern))
    if not parquets:
        raise FileNotFoundError(f"No parquet files matched: {args.test_data / args.pattern}")

    summary_path = output_root / "vlm_stage_summary.jsonl"
    with summary_path.open("w", encoding="utf-8") as handle:
        for parquet_path in parquets:
            ep_args = episode_args(args, parquet_path, output_root)
            record: dict[str, Any] = {"episode_id": parquet_path.stem, "parquet_path": str(parquet_path), "video_path": str(ep_args.video)}
            print(f"[{parquet_path.stem}] running VLM stage")
            try:
                result = run_vlm_stage(ep_args)
                payload = result["payload"]
                record.update(
                    {
                        "status": "ok",
                        "output_json": str(result["output_json"]),
                        "target_object": payload["vlm_target"]["target_object"],
                        "referring_expression": payload["vlm_target"]["referring_expression"],
                        "vlm_seconds": payload["timing_seconds"]["vlm_seconds"],
                    }
                )
            except Exception as exc:
                if args.stop_on_error:
                    raise
                record.update({"status": "skipped", "error": str(exc)})
                print(f"[{parquet_path.stem}] skipped: {exc}")
            handle.write(json.dumps(common.json_ready(record), ensure_ascii=False) + "\n")
            handle.flush()
    print(f"Wrote VLM stage summary: {summary_path}")


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
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA, help="Directory with episode_*.parquet and matching .mp4 files.")
    parser.add_argument("--pattern", default="episode_*.parquet")
    parser.add_argument("--single-episode", action="store_true", help="Use --video/--parquet instead of iterating --test-data.")
    parser.add_argument("--stop-on-error", action=argparse.BooleanOptionalAction, default=False)
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
    run_batch(args)


if __name__ == "__main__":
    main()
