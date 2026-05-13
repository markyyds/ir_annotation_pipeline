#!/usr/bin/env python3
"""Smoke-test SAM 3.1 ModelScope image text prompts on a video's first frame.

This script downloads facebook/sam3.1 from ModelScope, extracts the first
frame of a video, and feeds the requested text prompt to SAM3.
"""

from __future__ import annotations

import argparse
import inspect
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


def _load_sam3_image_builder():
    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'sam3'. Install the official package:\n"
            "  python -m pip install git+https://github.com/facebookresearch/sam3.git"
        ) from exc
    return build_sam3_image_model, Sam3Processor


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


def _find_checkpoint(model_dir: Path) -> Path | None:
    candidates = []
    for suffix in ("*.pt", "*.pth", "*.ckpt", "*.bin", "*.safetensors"):
        candidates.extend(model_dir.rglob(suffix))
    if not candidates:
        return None

    def priority(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        score = 0
        if "sam3.1" in name or "sam31" in name:
            score += 4
        if "image" in name:
            score += 2
        if name.endswith((".pt", ".pth")):
            score += 1
        return (-score, len(path.parts), str(path))

    return sorted(candidates, key=priority)[0]


def _build_image_model(builder, checkpoint_path: Path | None, device: str):
    if checkpoint_path is None:
        raise RuntimeError(
            "ModelScope download did not contain a SAM3 checkpoint. "
            "Refusing to call the SAM3 builder without a local checkpoint because "
            "that may fall back to Hugging Face. Pass --checkpoint explicitly if needed."
        )

    signature = inspect.signature(builder)
    kwargs: dict[str, Any] = {}
    checkpoint_arg_set = False
    for name in ("checkpoint_path", "ckpt_path", "checkpoint"):
        if name in signature.parameters:
            kwargs[name] = str(checkpoint_path)
            checkpoint_arg_set = True
            break
    if not checkpoint_arg_set:
        raise RuntimeError(
            "SAM3 image builder does not expose a known checkpoint argument "
            f"(signature: {signature}). Refusing to use a non-ModelScope fallback."
        )
    if "load_from_HF" in signature.parameters:
        kwargs["load_from_HF"] = False
    if "load_from_hf" in signature.parameters:
        kwargs["load_from_hf"] = False
    if "device" in signature.parameters:
        kwargs["device"] = device

    model = builder(**kwargs)
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return model, kwargs


def _build_processor(Sam3Processor, model, confidence_threshold: float):
    signature = inspect.signature(Sam3Processor)
    kwargs: dict[str, Any] = {}
    if "confidence_threshold" in signature.parameters:
        kwargs["confidence_threshold"] = confidence_threshold
    elif "conf_threshold" in signature.parameters:
        kwargs["conf_threshold"] = confidence_threshold
    return Sam3Processor(model, **kwargs)


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


def _first_present(mapping: dict[str, Any], names: tuple[str, ...]):
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _run_text_prompt(processor, image, prompt: str) -> dict[str, Any]:
    torch = _load_torch()
    autocast_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else _null_context()
    )
    with torch.inference_mode(), autocast_context:
        state = processor.set_image(image)
        try:
            return processor.set_text_prompt(state=state, prompt=prompt)
        except TypeError:
            return processor.set_text_prompt(prompt=prompt, state=state)


def _null_context():
    class _NullContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    return _NullContext()


def _mask_to_bbox(mask: Any) -> list[float] | None:
    np = _require("numpy", "python -m pip install numpy")
    mask_bool = np.asarray(mask).squeeze() > 0
    if mask_bool.ndim != 2 or not mask_bool.any():
        return None
    ys, xs = np.where(mask_bool)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _records_from_output(output: dict[str, Any], max_masks: int) -> tuple[list[Any], list[dict[str, Any]]]:
    masks = _to_numpy(_first_present(output, ("masks", "pred_masks")))
    boxes = _to_numpy(_first_present(output, ("boxes", "pred_boxes")))
    scores = _to_numpy(_first_present(output, ("scores", "pred_scores", "iou_scores")))
    labels = _first_present(output, ("labels", "text_labels"))

    if masks is None:
        raise RuntimeError(f"SAM3 output did not contain masks. Keys: {sorted(output.keys())}")

    masks = masks.squeeze(1) if getattr(masks, "ndim", 0) == 4 and masks.shape[1] == 1 else masks
    mask_list = [masks[idx] for idx in range(min(len(masks), max_masks))]

    records = []
    for idx, mask in enumerate(mask_list):
        record: dict[str, Any] = {"index": idx, "bbox_from_mask": _mask_to_bbox(mask)}
        if scores is not None and len(scores) > idx:
            record["score"] = float(scores[idx])
        if boxes is not None and len(boxes) > idx:
            record["box"] = [float(v) for v in boxes[idx].reshape(-1).tolist()]
        if labels is not None and len(labels) > idx:
            record["label"] = str(labels[idx])
        records.append(record)
    return mask_list, records


def _save_mask_png(mask, output_path: Path) -> None:
    Image, _ImageDraw = _load_pil()
    np = _require("numpy", "python -m pip install numpy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(mask).squeeze()
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}")
    Image.fromarray((mask > 0).astype("uint8") * 255).save(output_path)


def _save_overlay(image, masks, output_path: Path) -> None:
    Image, _ImageDraw = _load_pil()
    np = _require("numpy", "python -m pip install numpy")

    overlay = np.asarray(image).copy()
    colors = np.array(
        [
            [46, 204, 113],
            [52, 152, 219],
            [241, 196, 15],
            [231, 76, 60],
            [155, 89, 182],
        ],
        dtype=np.uint8,
    )
    for idx, mask in enumerate(masks):
        mask_bool = np.asarray(mask).squeeze() > 0
        if mask_bool.ndim != 2:
            continue
        color = colors[idx % len(colors)]
        overlay[mask_bool] = (0.55 * overlay[mask_bool] + 0.45 * color).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM 3.1 ModelScope text-prompt smoke test.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to a VIDEO file. The first frame is used.")
    parser.add_argument("--prompt", default="object", help="Text prompt passed to SAM3.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="ModelScope model id.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional ModelScope cache directory.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional explicit checkpoint path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--confidence-threshold", type=float, default=0.0001)
    parser.add_argument("--max-masks", type=int, default=8)
    parser.add_argument("--device", default=None, choices=("cpu", "cuda", "mps"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_video_path(args.input)
    torch = _load_torch()
    build_sam3_image_model, Sam3Processor = _load_sam3_image_builder()

    device = args.device or _auto_device(torch)
    model_dir = _download_model(args.model_id, args.cache_dir)
    checkpoint_path = args.checkpoint or _find_checkpoint(model_dir)

    image = _load_first_frame(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image.save(args.output_dir / "input_frame.jpg")

    model, build_kwargs = _build_image_model(build_sam3_image_model, checkpoint_path, device)
    processor = _build_processor(Sam3Processor, model, args.confidence_threshold)
    output = _run_text_prompt(processor, image, args.prompt)

    masks, records = _records_from_output(output, args.max_masks)
    for idx, mask in enumerate(masks):
        _save_mask_png(mask, args.output_dir / f"mask_{idx:02d}.png")
    _save_overlay(image, masks, args.output_dir / "overlay.jpg")

    metadata = {
        "model_id": args.model_id,
        "model_source": "modelscope",
        "modelscope_url": f"https://modelscope.cn/models/{args.model_id}",
        "model_dir": str(model_dir),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "build_kwargs": {key: str(value) for key, value in build_kwargs.items()},
        "device": device,
        "input_video": str(args.input),
        "frame_index": 0,
        "prompt": args.prompt,
        "confidence_threshold": args.confidence_threshold,
        "num_masks": len(masks),
        "detections": records,
        "output_keys": sorted(output.keys()),
    }
    (args.output_dir / "result.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote SAM3.1 text-prompt outputs to: {args.output_dir}")
    print(f"Input video: {args.input} (used frame 0)")
    print(f"Prompt: {args.prompt}")
    print(f"Masks: {len(masks)}")


if __name__ == "__main__":
    main()
