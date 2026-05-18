#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from video_object_detection_mapper import common, sam3_candidates, sam3_tracking, siglip_selector, vlm


DEFAULT_TEST_DATA = PROJECT_ROOT / "test_data"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"
DEFAULT_TASK_INSTRUCTION_COLUMN = "other_information.language_instruction_2"
DEFAULT_MOLMOPOINT_MODEL = "allenai/MolmoPoint-8B"
DEFAULT_SAM3_MODEL = "facebook/sam3"
DEFAULT_SIGLIP_MODEL = "google/siglip-base-patch16-224"
DEFAULT_MOLMOPOINT_PYTHON = PROJECT_ROOT / ".venv-molmopoint" / "bin" / "python"
DEFAULT_OUTPUT_SIZE = [320, 180]


class RuntimeContext:
    def __init__(self, args: argparse.Namespace):
        self.np, self.torch, self.imageio, self.Image, self.ImageDraw = common.load_common_modules()
        self.device = common.auto_device(self.torch, args.device)
        self._vlm_client = None
        self._sam3_candidate = None
        self._siglip = None
        self._sam3_video = None

    def vlm_client(self, args):
        if self._vlm_client is None:
            self._vlm_client = vlm.build_vllm_json_client(args)
        return self._vlm_client

    def sam3_candidate(self, args):
        key = (args.sam3_candidate_model, args.sam3_candidate_cache_dir, args.sam3_candidate_torch_dtype, self.device)
        if self._sam3_candidate is None or self._sam3_candidate[0] != key:
            model, processor, model_dir = sam3_candidates.load_tracker(args, self.torch, self.device)
            self._sam3_candidate = (key, model, processor, model_dir)
        return self._sam3_candidate[1], self._sam3_candidate[2], self._sam3_candidate[3]

    def siglip(self, args):
        key = (args.siglip_model, args.siglip_torch_dtype, self.device)
        if self._siglip is None or self._siglip[0] != key:
            model, processor = siglip_selector.load_siglip(args, self.torch, self.device)
            self._siglip = (key, model, processor)
        return self._siglip[1], self._siglip[2]

    def sam3_video(self, args):
        key = (
            args.sam3_video_model,
            args.sam3_video_cache_dir,
            args.sam3_video_checkpoint,
            args.sam3_video_version,
            self.device,
        )
        if self._sam3_video is None or self._sam3_video[0] != key:
            predictor, metadata = sam3_tracking.load_video_predictor(args, self.torch, self.device)
            self._sam3_video = (key, predictor, metadata)
        return self._sam3_video[1], self._sam3_video[2]


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


def output_coordinate_size(args: argparse.Namespace, frame_width: int, frame_height: int) -> tuple[int, int]:
    if args.output_coordinate_system == "input":
        return frame_width, frame_height
    return int(args.bbox_output_size[0]), int(args.bbox_output_size[1])


def materialize_tracking_outputs(args, context: RuntimeContext, frame_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_frames = []
    mask_dir = args.output_dir / "tracking_masks"
    bbox_dir = args.output_dir / "tracking_bboxes"
    bbox_output_width, bbox_output_height = output_coordinate_size(args, args.frame_width, args.frame_height)
    reader = None
    if args.save_tracking_bbox_images:
        bbox_dir.mkdir(parents=True, exist_ok=True)
        reader = context.imageio.get_reader(str(args.video))
    try:
        for record in frame_records:
            frame_idx = int(record["frame_index"])
            bbox_raw = record.get("bbox_xyxy")
            output_record = {
                "frame_index": frame_idx,
                "bbox_xyxy_raw": bbox_raw,
                "bbox_xyxy": common.scale_box(
                    bbox_raw,
                    args.frame_width,
                    args.frame_height,
                    bbox_output_width,
                    bbox_output_height,
                ),
            }
            if args.save_tracking_masks:
                mask_path = mask_dir / f"frame_{frame_idx:06d}_mask.png"
                common.save_mask_png(context.Image, context.np, record["mask"], mask_path)
                output_record["mask_path"] = str(mask_path)
            if reader is not None:
                try:
                    image = context.Image.fromarray(reader.get_data(frame_idx)).convert("RGB")
                    draw = context.ImageDraw.Draw(image)
                    bbox = bbox_raw
                    if bbox is not None:
                        draw.rectangle(tuple(bbox), outline="red", width=3)
                    image_path = bbox_dir / f"frame_{frame_idx:06d}_bbox.jpg"
                    image.save(image_path)
                    output_record["bbox_image_path"] = str(image_path)
                except Exception:
                    pass
            output_frames.append(output_record)
    finally:
        if reader is not None:
            reader.close()
    return output_frames


def load_vlm_stage_target(vlm_stage_json: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(vlm_stage_json.read_text(encoding="utf-8"))
    target = payload.get("vlm_target") if isinstance(payload, dict) else None
    if target is None and isinstance(payload, dict):
        target = payload
    if not isinstance(target, dict):
        raise RuntimeError(f"Invalid VLM stage JSON: missing object payload in {vlm_stage_json}")

    target_object = str(target.get("target_object") or target.get("object") or target.get("target") or "").strip()
    referring_expression = str(target.get("referring_expression") or target.get("referring") or target_object).strip()
    if not target_object:
        raise RuntimeError(f"Invalid VLM stage JSON: missing target_object in {vlm_stage_json}")
    if not referring_expression:
        referring_expression = target_object

    normalized_target = dict(target)
    normalized_target["target_object"] = target_object
    normalized_target["referring_expression"] = referring_expression
    normalized_target["loaded_from_vlm_stage_json"] = str(vlm_stage_json)
    if "model" not in normalized_target and isinstance(payload, dict) and payload.get("vlm_model"):
        normalized_target["model"] = payload["vlm_model"]
    return normalized_target, payload


def write_inline_vlm_stage_json(
    args: argparse.Namespace,
    instruction: str,
    target: dict[str, Any],
    first_frame_path: Path,
    last_frame_path: Path,
    frame_width: int,
    frame_height: int,
    first_idx: int,
    last_idx: int,
) -> Path:
    payload = {
        "status": "ok",
        "stage": "vlm_target_referring",
        "video_path": str(args.video),
        "parquet_path": str(args.parquet),
        "task_instruction": instruction,
        "vlm_model": args.vlm_model,
        "frame_context": {
            "first_frame_path": str(first_frame_path),
            "last_frame_path": str(last_frame_path),
            "first_video_frame_index": first_idx,
            "last_video_frame_index": last_idx,
            "width": frame_width,
            "height": frame_height,
        },
        "vlm_target": target,
        "timing_seconds": {"vlm_seconds": 0.0},
    }
    output_json = args.output_dir / "vlm_stage.json"
    output_json.write_text(json.dumps(common.json_ready(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_json


def load_molmopoint_stage(molmopoint_stage_json: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(molmopoint_stage_json.read_text(encoding="utf-8"))
    point_payload = payload.get("molmopoint") if isinstance(payload, dict) else None
    if not isinstance(point_payload, dict):
        raise RuntimeError(f"Invalid MolmoPoint stage JSON: missing molmopoint payload in {molmopoint_stage_json}")
    if "center_xy" not in point_payload:
        raise RuntimeError(f"Invalid MolmoPoint stage JSON: missing center_xy in {molmopoint_stage_json}")
    point_payload = dict(point_payload)
    point_payload["loaded_from_molmopoint_stage_json"] = str(molmopoint_stage_json)
    return point_payload, payload


def run_molmopoint_subprocess(args: argparse.Namespace, vlm_stage_json: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    output_json = args.output_dir / "molmopoint_stage.json"
    cmd = [
        str(args.molmopoint_python),
        str(SCRIPT_DIR / "run_molmopoint_stage.py"),
        "--vlm-stage-json",
        str(vlm_stage_json),
        "--output-dir",
        str(args.output_dir),
        "--device",
        str(args.molmopoint_device),
        "--molmopoint-model",
        str(args.molmopoint_model),
        "--molmopoint-device-map",
        str(args.molmopoint_device_map),
        "--molmopoint-max-new-tokens",
        str(args.molmopoint_max_new_tokens),
        "--molmopoint-prompt-template",
        str(args.molmopoint_prompt_template),
    ]
    if args.molmopoint_cache_dir is not None:
        cmd.extend(["--molmopoint-cache-dir", str(args.molmopoint_cache_dir)])
    cmd.append("--save-molmopoint-visualization" if args.save_molmopoint_visualization else "--no-save-molmopoint-visualization")
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    point_payload, stage_payload = load_molmopoint_stage(output_json)
    return point_payload, stage_payload, output_json


def episode_args(base_args: argparse.Namespace, parquet_path: Path, output_root: Path) -> argparse.Namespace:
    args = argparse.Namespace(**vars(base_args))
    args.parquet = parquet_path
    args.video = parquet_path.with_suffix(".mp4")
    args.output_dir = output_root / parquet_path.stem
    if args.vlm_stage_json is None:
        stage_root = args.vlm_stage_root or output_root
        candidate = stage_root / parquet_path.stem / "vlm_stage.json"
        if candidate.exists():
            args.vlm_stage_json = candidate
        elif args.vlm_stage_root is not None:
            raise FileNotFoundError(f"Missing cached VLM stage JSON: {candidate}")
    if args.molmopoint_stage_json is None:
        stage_root = args.molmopoint_stage_root or output_root
        candidate = stage_root / parquet_path.stem / "molmopoint_stage.json"
        if candidate.exists():
            args.molmopoint_stage_json = candidate
        elif args.molmopoint_stage_root is not None:
            raise FileNotFoundError(f"Missing cached MolmoPoint stage JSON: {candidate}")
    return args


def run_batch(args: argparse.Namespace) -> None:
    if args.single_episode:
        if args.output_dir is None:
            args.output_dir = default_output_dir(args.vlm_model)
        run_pipeline(args)
        return

    output_root = args.output_dir or default_output_dir(args.vlm_model)
    output_root.mkdir(parents=True, exist_ok=True)
    parquets = sorted(args.test_data.glob(args.pattern))
    if not parquets:
        raise FileNotFoundError(f"No parquet files matched: {args.test_data / args.pattern}")

    context = RuntimeContext(args)
    summary_path = output_root / "video_object_detection_summary.jsonl"
    with summary_path.open("w", encoding="utf-8") as handle:
        for parquet_path in parquets:
            ep_args = episode_args(args, parquet_path, output_root)
            record: dict[str, Any] = {"episode_id": parquet_path.stem, "parquet_path": str(parquet_path), "video_path": str(ep_args.video)}
            print(f"[{parquet_path.stem}] running object detection pipeline")
            try:
                result = run_pipeline(ep_args, context)
                payload = result["payload"]
                record.update(
                    {
                        "status": "ok",
                        "output_json": str(result["output_json"]),
                        "vlm_stage_json": payload.get("vlm_stage_json"),
                        "molmopoint_stage_json": payload.get("molmopoint_stage_json"),
                        "target_object": payload["vlm_target"]["target_object"],
                        "referring_expression": payload["vlm_target"]["referring_expression"],
                        "selected_bbox_xyxy": payload.get("selected_bbox_xyxy"),
                    }
                )
                evaluation_summary = ((payload.get("evaluation") or {}).get("summary") or {})
                timing_summary = payload.get("timing_seconds") or {}
                for key in (
                    "num_frames",
                    "num_valid_pairs",
                    "mean_iou",
                    "success_rate_iou_0_5",
                    "mean_center_distance_px",
                    "mean_normalized_center_distance",
                    "mean_bbox_l1_px",
                ):
                    record[key] = evaluation_summary.get(key)
                for key in (
                    "vlm_seconds",
                    "molmopoint_seconds",
                    "sam3_candidate_seconds",
                    "siglip_seconds",
                    "sam3_tracking_seconds",
                    "total_model_seconds",
                ):
                    record[key] = timing_summary.get(key)
            except Exception as exc:
                if args.stop_on_error:
                    raise
                record.update({"status": "skipped", "error": str(exc)})
                print(f"[{parquet_path.stem}] skipped: {exc}")
            handle.write(json.dumps(common.json_ready(record), ensure_ascii=False) + "\n")
            handle.flush()
    print(f"Wrote object detection summary: {summary_path}")


def run_pipeline(args: argparse.Namespace, context: RuntimeContext | None = None) -> dict[str, Any]:
    context = context or RuntimeContext(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}

    instruction = common.get_task_instruction(args.parquet, args.task_instruction_column)
    first_frame_path, last_frame_path, frame_width, frame_height, first_idx, last_idx = common.save_context_frames(
        args,
        context.imageio,
        context.Image,
    )
    args.frame_width = frame_width
    args.frame_height = frame_height
    bbox_output_width, bbox_output_height = output_coordinate_size(args, frame_width, frame_height)
    first_image = context.Image.open(first_frame_path).convert("RGB")

    vlm_stage_payload = None
    if args.vlm_stage_json:
        target, vlm_stage_payload = load_vlm_stage_target(args.vlm_stage_json)
        timing["vlm_seconds"] = 0.0
    else:
        started = time.perf_counter()
        target = vlm.extract_target_and_referring_expression(
            context.vlm_client(args),
            instruction,
            first_frame_path,
            last_frame_path,
            frame_width,
            frame_height,
        )
        timing["vlm_seconds"] = time.perf_counter() - started

    molmopoint_stage_payload = None
    if args.molmopoint_stage_json:
        point_payload, molmopoint_stage_payload = load_molmopoint_stage(args.molmopoint_stage_json)
        timing["molmopoint_seconds"] = 0.0
    else:
        vlm_stage_for_molmopoint = args.vlm_stage_json or write_inline_vlm_stage_json(
            args,
            instruction,
            target,
            first_frame_path,
            last_frame_path,
            frame_width,
            frame_height,
            first_idx,
            last_idx,
        )
        if args.vlm_stage_json is None:
            args.vlm_stage_json = vlm_stage_for_molmopoint
        if vlm_stage_payload is None:
            _target_from_stage, vlm_stage_payload = load_vlm_stage_target(vlm_stage_for_molmopoint)
        started = time.perf_counter()
        point_payload, molmopoint_stage_payload, args.molmopoint_stage_json = run_molmopoint_subprocess(
            args,
            vlm_stage_for_molmopoint,
        )
        timing["molmopoint_seconds"] = time.perf_counter() - started

    sam3_model, sam3_processor, sam3_model_dir = context.sam3_candidate(args)
    started = time.perf_counter()
    masks, scores, sam3_prompt_metadata = sam3_candidates.run_point_prompt(
        args,
        context.np,
        context.torch,
        sam3_model,
        sam3_processor,
        first_image,
        point_payload["center_xy"],
        context.device,
    )
    detections = sam3_candidates.save_candidates(
        args,
        context.Image,
        context.ImageDraw,
        context.np,
        first_image,
        masks,
        scores,
        point_payload["center_xy"],
    )
    visible_masks = masks[: args.sam3_candidate_max_masks]
    visible_scores = scores[: args.sam3_candidate_max_masks]
    timing["sam3_candidate_seconds"] = time.perf_counter() - started

    siglip_model, siglip_processor = context.siglip(args)
    started = time.perf_counter()
    candidate_rankings, selected_detection = siglip_selector.rank_candidates(
        args,
        context.np,
        context.torch,
        context.Image,
        first_image,
        detections,
        visible_masks,
        visible_scores,
        point_payload["center_xy"],
        target["target_object"],
        siglip_model,
        siglip_processor,
        context.device,
    )
    timing["siglip_seconds"] = time.perf_counter() - started
    if selected_detection is None:
        raise RuntimeError("SigLIP did not select a valid SAM3 candidate mask.")

    selected_idx = int(selected_detection["index"])
    selected_mask = visible_masks[selected_idx]
    selected_bbox = selected_detection.get("bbox_from_mask")
    selected_mask_path = args.output_dir / "selected_mask.png"
    selected_overlay_path = args.output_dir / "selected_overlay.jpg"
    common.save_mask_png(context.Image, context.np, selected_mask, selected_mask_path)
    common.save_overlay(
        context.Image,
        context.ImageDraw,
        context.np,
        first_image,
        selected_mask,
        point_payload["center_xy"],
        selected_bbox,
        selected_overlay_path,
    )
    selected_detection["selected_mask_path"] = str(selected_mask_path)
    selected_detection["selected_overlay_path"] = str(selected_overlay_path)

    tracking_payload = None
    evaluation = None
    if args.run_tracking:
        predictor, sam3_video_metadata = context.sam3_video(args)
        started = time.perf_counter()
        frame_records, tracking_metadata = sam3_tracking.track_video(
            context.np,
            context.torch,
            predictor,
            args.video,
            selected_bbox,
            selected_mask,
            point_payload["center_xy"],
            frame_width,
            frame_height,
            args.sam3_video_obj_id,
        )
        tracking_frames = materialize_tracking_outputs(args, context, frame_records)
        timing["sam3_tracking_seconds"] = time.perf_counter() - started
        tracking_payload = {
            **sam3_video_metadata,
            **tracking_metadata,
            "num_frames": len(tracking_frames),
            "frames": tracking_frames,
        }
        if not args.skip_evaluation:
            raw_gt_by_frame = common.load_ground_truth_boxes(args.parquet, args.frame_column, args.gt_box_column)
            gt_by_frame = {
                frame_idx: common.scale_box(box, frame_width, frame_height, bbox_output_width, bbox_output_height)
                for frame_idx, box in raw_gt_by_frame.items()
            }
            evaluation = common.evaluate_bboxes(tracking_frames, gt_by_frame, bbox_output_width, bbox_output_height)

    timing["total_model_seconds"] = sum(timing.values())
    selected_bbox_output = common.scale_box(selected_bbox, frame_width, frame_height, bbox_output_width, bbox_output_height)
    payload = {
        "status": "ok",
        "video_path": str(args.video),
        "parquet_path": str(args.parquet),
        "task_instruction": instruction,
        "vlm_model": args.vlm_model,
        "frame_context": {
            "first_frame_path": str(first_frame_path),
            "last_frame_path": str(last_frame_path),
            "first_video_frame_index": first_idx,
            "last_video_frame_index": last_idx,
            "width": frame_width,
            "height": frame_height,
            "bbox_output_width": bbox_output_width,
            "bbox_output_height": bbox_output_height,
            "output_coordinate_system": args.output_coordinate_system,
        },
        "vlm_stage_json": str(args.vlm_stage_json) if args.vlm_stage_json else None,
        "vlm_stage_payload": vlm_stage_payload,
        "vlm_target": target,
        "molmopoint_stage_json": str(args.molmopoint_stage_json) if args.molmopoint_stage_json else None,
        "molmopoint_stage_payload": molmopoint_stage_payload,
        "molmopoint": point_payload,
        "sam3_candidates": {
            "model": args.sam3_candidate_model,
            "model_dir": sam3_model_dir,
            "num_candidate_masks_total": len(masks),
            "num_candidate_masks_scored": len(visible_masks),
            "prompt_metadata": sam3_prompt_metadata,
            "detections": detections,
        },
        "siglip_selection": {
            "model": args.siglip_model,
            "target_object": target["target_object"],
            "candidate_rankings": candidate_rankings,
            "selected_detection": selected_detection,
            "score_weights": {
                "masked_crop_siglip": args.siglip_masked_weight,
                "context_crop_siglip": args.siglip_context_weight,
                "sam3_score": args.siglip_sam3_score_weight,
                "point_inside": args.siglip_point_inside_weight,
            },
        },
        "selected_bbox_xyxy_raw": selected_bbox,
        "selected_bbox_xyxy": selected_bbox_output,
        "tracking": tracking_payload,
        "evaluation": evaluation,
        "timing_seconds": timing,
    }
    output_json = args.output_dir / "video_object_detection.json"
    output_json.write_text(json.dumps(common.json_ready(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote object detection output: {output_json}")
    print(f"Selected bbox: {selected_bbox}")
    return {"output_json": output_json, "payload": payload}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modular VLM/MolmoPoint/SAM3/SigLIP video object detection pipeline.")
    parser.add_argument("--video", type=Path, default=DEFAULT_TEST_DATA / "episode_000000.mp4")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_TEST_DATA / "episode_000000.parquet")
    parser.add_argument("--output-dir", type=Path, help="Defaults to video_object_detection_mapper/outputs/{vlm_model}.")
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA, help="Directory with episode_*.parquet and matching .mp4 files.")
    parser.add_argument("--pattern", default="episode_*.parquet")
    parser.add_argument("--single-episode", action="store_true", help="Use --video/--parquet instead of iterating --test-data.")
    parser.add_argument("--stop-on-error", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--task-instruction-column", default=DEFAULT_TASK_INSTRUCTION_COLUMN)
    parser.add_argument("--frame-column", default="frame_index")
    parser.add_argument("--gt-box-column", default="annotation.object_box")
    parser.add_argument("--first-video-frame-index", type=int, default=0)
    parser.add_argument("--last-video-frame-index", type=int, default=-1)
    parser.add_argument("--output-coordinate-system", choices=("320x180", "input"), default="320x180")
    parser.add_argument("--bbox-output-size", type=int, nargs=2, default=DEFAULT_OUTPUT_SIZE, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--device", default="auto")

    parser.add_argument("--vlm-stage-json", type=Path, help="Reuse a saved run_vlm_stage.py output and skip the VLM request.")
    parser.add_argument(
        "--vlm-stage-root",
        type=Path,
        help="Batch cache root containing {episode_id}/vlm_stage.json. Defaults to the pipeline output root.",
    )
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

    parser.add_argument(
        "--molmopoint-python",
        type=Path,
        default=DEFAULT_MOLMOPOINT_PYTHON,
        help="Python executable for the MolmoPoint environment. Defaults to .venv-molmopoint/bin/python.",
    )
    parser.add_argument("--molmopoint-device", default="auto", help="Device passed to the MolmoPoint subprocess.")
    parser.add_argument("--molmopoint-stage-json", type=Path, help="Reuse a saved MolmoPoint stage output and skip the MolmoPoint subprocess.")
    parser.add_argument(
        "--molmopoint-stage-root",
        type=Path,
        help="Batch cache root containing {episode_id}/molmopoint_stage.json. Defaults to the pipeline output root.",
    )
    parser.add_argument("--molmopoint-model", default=DEFAULT_MOLMOPOINT_MODEL)
    parser.add_argument("--molmopoint-cache-dir", type=Path)
    parser.add_argument("--molmopoint-device-map", default="auto")
    parser.add_argument("--molmopoint-max-new-tokens", type=int, default=200)
    parser.add_argument("--molmopoint-prompt-template", default="Point to {referring_expression}")
    parser.add_argument("--save-molmopoint-visualization", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--sam3-candidate-model", default=DEFAULT_SAM3_MODEL)
    parser.add_argument("--sam3-candidate-cache-dir", type=Path)
    parser.add_argument("--sam3-candidate-torch-dtype", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--sam3-candidate-multimask-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sam3-candidate-binarize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sam3-candidate-max-masks", type=int, default=8)

    parser.add_argument("--siglip-model", default=DEFAULT_SIGLIP_MODEL)
    parser.add_argument("--siglip-torch-dtype", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--crop-padding", type=float, default=0.35)
    parser.add_argument("--masked-fill", type=int, default=128)
    parser.add_argument("--min-mask-area-fraction", type=float, default=0.001)
    parser.add_argument("--max-mask-area-fraction", type=float, default=0.8)
    parser.add_argument("--save-candidate-crops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--siglip-masked-weight", type=float, default=0.60)
    parser.add_argument("--siglip-context-weight", type=float, default=0.30)
    parser.add_argument("--siglip-sam3-score-weight", type=float, default=0.10)
    parser.add_argument("--siglip-point-inside-weight", type=float, default=0.0)

    parser.add_argument("--run-tracking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sam3-video-model", default=DEFAULT_SAM3_MODEL)
    parser.add_argument("--sam3-video-cache-dir", type=Path)
    parser.add_argument("--sam3-video-checkpoint", type=Path)
    parser.add_argument("--sam3-video-version", default="3")
    parser.add_argument("--sam3-video-obj-id", type=int, default=0)
    parser.add_argument("--sam3-video-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sam3-video-warm-up", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sam3-video-async-loading-frames", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-tracking-masks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-tracking-bbox-images", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_batch(args)


if __name__ == "__main__":
    main()
