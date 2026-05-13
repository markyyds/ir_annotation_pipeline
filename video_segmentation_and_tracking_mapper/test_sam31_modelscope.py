#!/usr/bin/env python3
"""Smoke-test SAM3 tracker candidate masks from a first-frame point.

This script downloads `facebook/sam3.1` from ModelScope, extracts the first
frame of a video, sends a positive point prompt to the SAM3 tracker through
Transformers, and writes all returned candidate masks with bboxes/overlays.

The model is loaded only from the local ModelScope snapshot path. We pass
`local_files_only=True` to Transformers so it does not fall back to Hugging Face.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "test_data/episode_000000.mp4"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "sam31_modelscope_test"
DEFAULT_MODEL_ID = "facebook/sam3.1"
VIDEO_SUFFIXES: frozenset[str] = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv"}
)


def _require(module_name: str, install_hint: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise RuntimeError(f"Missing dependency '{module_name}'. Install it with: {install_hint}") from exc


def _load_torch():
    return _require("torch", "python -m pip install torch torchvision")


def _load_pil():
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'Pillow'. Install it with: python -m pip install pillow") from exc
    return Image, ImageDraw


def _auto_device(torch_module) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def _download_model(model_id: str, cache_dir: Path | None) -> Path:
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'modelscope'. Install it with: python -m pip install modelscope") from exc

    kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return Path(snapshot_download(model_id, **kwargs))


def _load_transformers_tracker(model_dir: Path, device: str, torch_dtype: str):
    try:
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'transformers' with Sam3TrackerModel/Sam3TrackerProcessor. "
            "Install a SAM3-tracker-capable version, for example:\n"
            "  python -m pip install 'transformers>=4.57.1' accelerate"
        ) from exc

    torch = _load_torch()
    dtype = {
        "auto": "auto",
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[torch_dtype]
    model = Sam3TrackerModel.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=dtype if dtype != "auto" else None,
    )
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    processor = Sam3TrackerProcessor.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )
    return model, processor


def _validate_video_path(input_path: Path) -> None:
    if not input_path.exists():
        raise RuntimeError(f"Input video does not exist: {input_path}")
    suffix = input_path.suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise RuntimeError(
            f"Input must be a video file (allowed suffixes: {sorted(VIDEO_SUFFIXES)}); "
            f"got '{suffix or '<no suffix>'}' for: {input_path}"
        )


def _load_first_frame(input_path: Path):
    Image, _ImageDraw = _load_pil()
    cv2 = _require("cv2", "python -m pip install opencv-python")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video with OpenCV: {input_path}")
    try:
        ok, frame_bgr = cap.read()
    finally:
        cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to read first frame from video: {input_path}")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def _to_numpy(value: Any):
    torch = _load_torch()
    np = _require("numpy", "python -m pip install numpy")
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        if getattr(value, "dtype", None) == torch.bfloat16:
            value = value.float()
        return value.numpy()
    return np.asarray(value)


def _move_inputs_to_device(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _mask_to_bbox(mask: Any) -> list[float] | None:
    np = _require("numpy", "python -m pip install numpy")
    mask_bool = np.asarray(mask).squeeze() > 0
    if mask_bool.ndim != 2 or not mask_bool.any():
        return None
    ys, xs = np.where(mask_bool)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _candidate_masks_from_output(processor, outputs, original_sizes, binarize: bool):
    np = _require("numpy", "python -m pip install numpy")
    if not hasattr(outputs, "pred_masks"):
        available = sorted(name for name in dir(outputs) if not name.startswith("_"))
        raise RuntimeError(f"SAM3 tracker output does not expose pred_masks. Available fields: {available}")

    try:
        processed = processor.post_process_masks(
            outputs.pred_masks.cpu(),
            original_sizes.cpu() if hasattr(original_sizes, "cpu") else original_sizes,
            binarize=binarize,
        )[0]
    except TypeError:
        processed = processor.post_process_masks(
            outputs.pred_masks.cpu(),
            original_sizes.cpu() if hasattr(original_sizes, "cpu") else original_sizes,
        )[0]
    masks_np = _to_numpy(processed)

    masks_np = np.asarray(masks_np).squeeze()
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    elif masks_np.ndim == 4:
        masks_np = masks_np.reshape((-1,) + masks_np.shape[-2:])
    elif masks_np.ndim > 3:
        masks_np = masks_np.reshape((-1,) + masks_np.shape[-2:])
    return [masks_np[idx] for idx in range(len(masks_np))]


def _scores_from_output(outputs) -> list[float | None]:
    for name in ("iou_scores", "pred_iou_scores", "scores"):
        value = getattr(outputs, name, None)
        if value is not None:
            scores = _to_numpy(value).reshape(-1).tolist()
            return [float(score) for score in scores]
    return []


def _run_tracker_point_prompt(
    model,
    processor,
    image,
    point_xy: list[float],
    device: str,
    multimask_output: bool,
    binarize: bool,
) -> tuple[list[Any], list[float | None], dict[str, Any]]:
    torch = _load_torch()
    inputs = processor(
        images=image,
        input_points=[[[[float(point_xy[0]), float(point_xy[1])]]]],
        input_labels=[[[1]]],
        return_tensors="pt",
    )
    inputs = _move_inputs_to_device(dict(inputs), device)
    with torch.inference_mode():
        try:
            outputs = model(**inputs, multimask_output=multimask_output)
        except TypeError:
            outputs = model(**inputs)

    masks = _candidate_masks_from_output(processor, outputs, inputs["original_sizes"], binarize)
    scores = _scores_from_output(outputs)
    metadata = {
        "input_point_xy": point_xy,
        "input_points_shape": list(inputs["input_points"].shape) if "input_points" in inputs else None,
        "original_sizes": _to_numpy(inputs["original_sizes"]).tolist() if "original_sizes" in inputs else None,
        "output_fields": sorted(name for name in dir(outputs) if not name.startswith("_")),
    }
    return masks, scores, metadata


def _save_mask_png(mask, output_path: Path) -> None:
    Image, _ImageDraw = _load_pil()
    np = _require("numpy", "python -m pip install numpy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(mask).squeeze()
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}")
    Image.fromarray((mask > 0).astype("uint8") * 255).save(output_path)


def _save_overlay(image, mask, point_xy: list[float], bbox: list[float] | None, output_path: Path) -> None:
    Image, ImageDraw = _load_pil()
    np = _require("numpy", "python -m pip install numpy")

    overlay = np.asarray(image).copy()
    mask_bool = np.asarray(mask).squeeze() > 0
    if mask_bool.ndim == 2:
        color = np.array([46, 204, 113], dtype=np.uint8)
        overlay[mask_bool] = (0.55 * overlay[mask_bool] + 0.45 * color).astype(np.uint8)
    annotated = Image.fromarray(overlay)
    draw = ImageDraw.Draw(annotated)
    if bbox is not None:
        draw.rectangle(tuple(bbox), outline="red", width=3)
    x, y = point_xy
    radius = 7
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="yellow", outline="black", width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM3 tracker candidate masks from a first-frame point.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to a VIDEO file. The first frame is used.")
    parser.add_argument("--point", type=float, nargs=2, metavar=("X", "Y"), help="Point in first-frame pixels. Defaults to frame center.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="ModelScope model id.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional ModelScope cache directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None, choices=("cpu", "cuda", "mps"))
    parser.add_argument("--torch-dtype", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--multimask-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--binarize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-masks", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_video_path(args.input)
    torch = _load_torch()
    device = args.device or _auto_device(torch)

    model_dir = _download_model(args.model_id, args.cache_dir)
    image = _load_first_frame(args.input)
    width, height = image.size
    point_xy = [float(args.point[0]), float(args.point[1])] if args.point else [width / 2.0, height / 2.0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image.save(args.output_dir / "input_frame.jpg")

    model, processor = _load_transformers_tracker(model_dir, device, args.torch_dtype)
    masks, scores, prompt_metadata = _run_tracker_point_prompt(
        model=model,
        processor=processor,
        image=image,
        point_xy=point_xy,
        device=device,
        multimask_output=args.multimask_output,
        binarize=args.binarize,
    )

    detections = []
    for idx, mask in enumerate(masks[: args.max_masks]):
        bbox = _mask_to_bbox(mask)
        mask_path = args.output_dir / f"candidate_mask_{idx:02d}.png"
        overlay_path = args.output_dir / f"candidate_overlay_{idx:02d}.jpg"
        _save_mask_png(mask, mask_path)
        _save_overlay(image, mask, point_xy, bbox, overlay_path)
        detections.append(
            {
                "index": idx,
                "score": scores[idx] if idx < len(scores) else None,
                "bbox_from_mask": bbox,
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
            }
        )

    metadata = {
        "model_id": args.model_id,
        "model_source": "modelscope",
        "modelscope_url": f"https://modelscope.cn/models/{args.model_id}",
        "model_dir": str(model_dir),
        "device": device,
        "input_video": str(args.input),
        "frame_index": 0,
        "frame_size": [width, height],
        "point_xy": point_xy,
        "multimask_output": args.multimask_output,
        "binarize": args.binarize,
        "num_candidate_masks": len(masks),
        "prompt_metadata": prompt_metadata,
        "detections": detections,
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote SAM3 tracker candidate mask outputs to: {args.output_dir}")
    print(f"ModelScope model dir: {model_dir}")
    print(f"Point: {point_xy}")
    print(f"Candidate masks: {len(masks)}")


if __name__ == "__main__":
    main()
