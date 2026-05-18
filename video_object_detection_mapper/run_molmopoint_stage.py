#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from video_object_detection_mapper import common, molmopoint


DEFAULT_MOLMOPOINT_MODEL = "allenai/MolmoPoint-8B"


def load_vlm_stage(vlm_stage_json: Path) -> tuple[dict, dict]:
    payload = json.loads(vlm_stage_json.read_text(encoding="utf-8"))
    target = payload.get("vlm_target") if isinstance(payload, dict) else None
    frame_context = payload.get("frame_context") if isinstance(payload, dict) else None
    if not isinstance(target, dict):
        raise RuntimeError(f"Invalid VLM stage JSON: missing vlm_target in {vlm_stage_json}")
    if not isinstance(frame_context, dict):
        raise RuntimeError(f"Invalid VLM stage JSON: missing frame_context in {vlm_stage_json}")

    target_object = str(target.get("target_object") or target.get("object") or target.get("target") or "").strip()
    referring_expression = str(target.get("referring_expression") or target.get("referring") or target_object).strip()
    if not target_object:
        raise RuntimeError(f"Invalid VLM stage JSON: missing target_object in {vlm_stage_json}")
    if not referring_expression:
        referring_expression = target_object

    normalized_target = dict(target)
    normalized_target["target_object"] = target_object
    normalized_target["referring_expression"] = referring_expression
    return payload, normalized_target


def run_molmopoint_stage(args: argparse.Namespace) -> dict:
    import numpy as np
    import torch
    from PIL import Image, ImageDraw

    device = common.auto_device(torch, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vlm_stage_payload, target = load_vlm_stage(args.vlm_stage_json)
    frame_context = vlm_stage_payload["frame_context"]
    first_frame_path = Path(frame_context["first_frame_path"])
    width = int(frame_context["width"])
    height = int(frame_context["height"])

    model, processor, model_dir, device_map = molmopoint.load_model_and_processor(args, device)
    started = time.perf_counter()
    point_payload = molmopoint.run_pointing(
        args,
        np,
        torch,
        Image,
        ImageDraw,
        model,
        processor,
        first_frame_path,
        target["referring_expression"],
        width,
        height,
        device,
    )
    timing = {"molmopoint_seconds": time.perf_counter() - started}
    point_payload["model_dir"] = model_dir
    point_payload["device_map"] = device_map

    payload = {
        "status": "ok",
        "stage": "molmopoint",
        "vlm_stage_json": str(args.vlm_stage_json),
        "video_path": vlm_stage_payload.get("video_path"),
        "parquet_path": vlm_stage_payload.get("parquet_path"),
        "frame_context": frame_context,
        "vlm_target": target,
        "molmopoint": point_payload,
        "timing_seconds": timing,
    }
    output_json = args.output_dir / "molmopoint_stage.json"
    output_json.write_text(json.dumps(common.json_ready(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote MolmoPoint stage output: {output_json}")
    print(f"Point: {point_payload['center_xy']}")
    return {"output_json": output_json, "payload": payload}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MolmoPoint stage in the MolmoPoint environment.")
    parser.add_argument("--vlm-stage-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--molmopoint-model", default=DEFAULT_MOLMOPOINT_MODEL)
    parser.add_argument("--molmopoint-cache-dir", type=Path)
    parser.add_argument("--molmopoint-device-map", default="auto")
    parser.add_argument("--molmopoint-max-new-tokens", type=int, default=200)
    parser.add_argument("--molmopoint-prompt-template", default="Point to {referring_expression}")
    parser.add_argument("--save-molmopoint-visualization", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    run_molmopoint_stage(parse_args())


if __name__ == "__main__":
    main()
