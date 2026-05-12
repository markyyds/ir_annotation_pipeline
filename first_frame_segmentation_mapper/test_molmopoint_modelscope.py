#!/usr/bin/env python3
"""Smoke-test MolmoPoint-8B from ModelScope on the first frame of a video.

The script downloads `allenai/MolmoPoint-8B` from ModelScope, extracts the
first frame from a local .mp4, asks MolmoPoint to point to the requested target,
and writes an annotated image with the returned point(s).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = PROJECT_ROOT / "test_data/episode_000000.mp4"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "molmopoint_modelscope_test"
DEFAULT_MODEL_ID = "allenai/MolmoPoint-8B"


def _require(module_name: str, install_hint: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise RuntimeError(f"Missing dependency '{module_name}'. Install it with: {install_hint}") from exc


def _download_model(model_id: str, cache_dir: Path | None) -> Path:
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'modelscope'. Install it with:\n"
            "  python -m pip install modelscope"
        ) from exc

    kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return Path(snapshot_download(model_id, **kwargs))


def _auto_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _extract_first_frame(video_path: Path, output_path: Path) -> None:
    cv2 = _require("cv2", "python -m pip install opencv-python")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to read the first frame from: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame_bgr):
        raise RuntimeError(f"Failed to write first frame: {output_path}")


def _load_transformers():
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'transformers'. MolmoPoint recommends transformers==4.57.1:\n"
            "  python -m pip install transformers==4.57.1"
        ) from exc
    return AutoModelForImageTextToText, AutoProcessor


def _load_pil():
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'Pillow'. Install it with: python -m pip install pillow") from exc
    return Image, ImageDraw


def _load_model_and_processor(model_dir: Path, device: str, device_map: str | None):
    AutoModelForImageTextToText, AutoProcessor = _load_transformers()

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": "auto",
    }
    if device_map:
        model_kwargs["device_map"] = device_map

    model = AutoModelForImageTextToText.from_pretrained(str(model_dir), **model_kwargs)
    if not device_map and hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    processor = AutoProcessor.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        padding_side="left",
    )
    return model, processor


def _move_inputs_to_device(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def _run_pointing(
    model,
    processor,
    frame_path: Path,
    prompt: str,
    device: str,
    max_new_tokens: int,
) -> tuple[str, list[list[Any]]]:
    torch = _require("torch", "python -m pip install torch torchvision")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": str(frame_path)},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
        return_pointing_metadata=True,
    )
    metadata = inputs.pop("metadata")
    inputs = _move_inputs_to_device(inputs, device)

    autocast_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else _null_context()
    )
    with torch.inference_mode(), autocast_context:
        output = model.generate(
            **inputs,
            logits_processor=model.build_logit_processor_from_inputs(inputs),
            max_new_tokens=max_new_tokens,
        )

    generated_tokens = output[:, inputs["input_ids"].size(1) :]
    generated_text = processor.post_process_image_text_to_text(
        generated_tokens,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]
    points = model.extract_image_points(
        generated_text,
        metadata["token_pooling"],
        metadata["subpatch_mapping"],
        metadata["image_sizes"],
    )
    return generated_text, points


def _null_context():
    class _NullContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    return _NullContext()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if hasattr(value, "item"):
        try:
            return _json_ready(value.item())
        except Exception:
            pass
    return value


def _normalize_points(points: list[list[Any]]) -> list[dict[str, Any]]:
    normalized = []
    for idx, point in enumerate(points):
        values = list(point)
        if len(values) < 4:
            continue
        object_id, image_num, x, y = values[:4]
        normalized.append(
            {
                "index": idx,
                "object_id": int(object_id),
                "image_num": int(image_num),
                "x": float(x),
                "y": float(y),
            }
        )
    return normalized


def _annotate_points(frame_path: Path, points: list[dict[str, Any]], output_path: Path) -> None:
    Image, ImageDraw = _load_pil()

    image = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = ["#ff3b30", "#34c759", "#007aff", "#ffcc00", "#af52de", "#ff9500"]
    radius = max(6, round(min(image.size) * 0.015))

    for point in points:
        x = point["x"]
        y = point["y"]
        color = colors[point["index"] % len(colors)]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=3)
        draw.line((x - radius * 2, y, x + radius * 2, y), fill="white", width=2)
        draw.line((x, y - radius * 2, x, y + radius * 2), fill="white", width=2)
        label = f"{point['index']} ({round(x)}, {round(y)})"
        draw.text((x + radius + 4, max(0, y - radius - 4)), label, fill=color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MolmoPoint-8B pointing on a video's first frame.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--prompt", default="Point to the target object")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Passed to Transformers from_pretrained. Use 'none' to disable sharding.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch = _require("torch", "python -m pip install torch torchvision")

    device = _auto_device(torch) if args.device == "auto" else args.device
    device_map = None if args.device_map.lower() in {"", "none", "null"} else args.device_map

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = args.output_dir / "first_frame.jpg"
    annotated_path = args.output_dir / "first_frame_pointing.jpg"
    result_path = args.output_dir / "pointing_result.json"

    _extract_first_frame(args.video, frame_path)
    model_dir = _download_model(args.model_id, args.cache_dir)
    model, processor = _load_model_and_processor(model_dir, device, device_map)
    generated_text, raw_points = _run_pointing(
        model=model,
        processor=processor,
        frame_path=frame_path,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )
    points = _normalize_points(raw_points)
    _annotate_points(frame_path, points, annotated_path)

    result = {
        "model_id": args.model_id,
        "model_dir": str(model_dir),
        "video": str(args.video),
        "frame_path": str(frame_path),
        "annotated_path": str(annotated_path),
        "prompt": args.prompt,
        "device": device,
        "device_map": device_map,
        "generated_text": generated_text,
        "points": points,
        "raw_points": _json_ready(raw_points),
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote first frame: {frame_path}")
    print(f"Wrote annotated image: {annotated_path}")
    print(f"Wrote result JSON: {result_path}")
    print(f"Points: {points}")


if __name__ == "__main__":
    main()
