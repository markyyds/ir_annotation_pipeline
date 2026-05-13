#!/usr/bin/env python3
"""Smoke-test SAM 3.1 video point prompting from the ModelScope checkpoint.

This test follows the point-prompt path we want in the main mapper:

1. Download facebook/sam3.1 with ModelScope.
2. Build a SAM3 video predictor.
3. Start a video session from an .mp4.
4. Add a positive point prompt on frame 0.
5. Propagate the object through the video and write masks/bboxes/overlays.

The input MUST be a video file. The point is accepted in absolute first-frame
pixel coordinates and normalized before it is sent to SAM3.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Iterable


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


def _load_sam3_video_builder():
    try:
        from sam3.model_builder import build_sam3_video_predictor
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'sam3' or a SAM3 build without video predictor support. Install the official package:\n"
            "  python -m pip install git+https://github.com/facebookresearch/sam3.git"
        ) from exc
    return build_sam3_video_predictor


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
        if "video" in name or "tracker" in name:
            score += 3
        if name.endswith((".pt", ".pth")):
            score += 1
        return (-score, len(path.parts), str(path))

    return sorted(candidates, key=priority)[0]


def _build_video_predictor(builder, checkpoint_path: Path | None, device: str, args: argparse.Namespace):
    torch = _load_torch()
    signature = inspect.signature(builder)
    kwargs: dict[str, Any] = {}
    if checkpoint_path is not None:
        for name in ("checkpoint_path", "ckpt_path", "checkpoint"):
            if name in signature.parameters:
                kwargs[name] = str(checkpoint_path)
                break
        if "load_from_HF" in signature.parameters:
            kwargs["load_from_HF"] = False
        if "load_from_hf" in signature.parameters:
            kwargs["load_from_hf"] = False
    if "version" in signature.parameters:
        kwargs["version"] = args.version
    if "gpus_to_use" in signature.parameters and device.startswith("cuda"):
        kwargs["gpus_to_use"] = range(torch.cuda.device_count())
    if "compile" in signature.parameters:
        kwargs["compile"] = args.compile
    if "warm_up" in signature.parameters:
        kwargs["warm_up"] = args.warm_up
    if "async_loading_frames" in signature.parameters:
        kwargs["async_loading_frames"] = args.async_loading_frames
    return builder(**kwargs), kwargs


def _validate_video_path(input_path: Path) -> None:
    if not input_path.exists():
        raise RuntimeError(f"Input video does not exist: {input_path}")
    suffix = input_path.suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise RuntimeError(
            f"Input must be a video file (allowed suffixes: {sorted(VIDEO_SUFFIXES)}); "
            f"got '{suffix or '<no suffix>'}' for: {input_path}"
        )


def _read_frame(video_path: Path, frame_index: int):
    imageio = _require("imageio.v2", "python -m pip install 'imageio[ffmpeg]'")
    reader = imageio.get_reader(str(video_path))
    try:
        return reader.get_data(frame_index)
    finally:
        reader.close()


def _frame_size(video_path: Path) -> tuple[int, int]:
    frame = _read_frame(video_path, 0)
    height, width = frame.shape[:2]
    return int(width), int(height)


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


def _response_outputs(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        if isinstance(response.get("outputs"), dict):
            return response["outputs"]
        return response
    if hasattr(response, "outputs") and isinstance(response.outputs, dict):
        return response.outputs
    return {}


def _response_session_id(response: Any) -> str:
    if isinstance(response, dict) and "session_id" in response:
        return str(response["session_id"])
    if hasattr(response, "session_id"):
        return str(response.session_id)
    raise RuntimeError(f"SAM3 start_session response did not contain session_id: {response}")


def _mask_to_bbox(mask: Any) -> list[float] | None:
    np = _require("numpy", "python -m pip install numpy")
    mask_bool = np.asarray(mask).squeeze() > 0
    if mask_bool.ndim != 2 or not mask_bool.any():
        return None
    ys, xs = np.where(mask_bool)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _extract_mask(outputs: dict[str, Any], obj_id: int):
    np = _require("numpy", "python -m pip install numpy")
    masks = _to_numpy(_first_present(outputs, ("out_binary_masks", "pred_masks", "masks")))
    if masks is None:
        return None
    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 4 and masks.shape[0] == 1:
        masks = masks[0]
    if masks.ndim == 2:
        return masks
    if masks.ndim < 2:
        return None

    obj_ids = _first_present(outputs, ("out_obj_ids", "obj_ids", "object_ids"))
    if obj_ids is not None:
        obj_ids = [int(item) for item in np.asarray(_to_numpy(obj_ids)).reshape(-1).tolist()]
        if obj_id in obj_ids:
            return masks[obj_ids.index(obj_id)]
    return masks[0]


def _iter_stream(stream: Any) -> Iterable[Any]:
    if stream is None:
        return []
    return stream


def _save_mask(mask: Any, path: Path) -> None:
    Image, _ImageDraw = _load_pil()
    np = _require("numpy", "python -m pip install numpy")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(mask).squeeze() > 0).astype("uint8") * 255).save(path)


def _save_overlay(video_path: Path, frame_index: int, mask: Any, point_xy: list[float] | None, bbox: list[float] | None, path: Path) -> None:
    Image, ImageDraw = _load_pil()
    np = _require("numpy", "python -m pip install numpy")

    image = Image.fromarray(_read_frame(video_path, frame_index)).convert("RGB")
    overlay = np.asarray(image).copy()
    mask_bool = np.asarray(mask).squeeze() > 0
    if mask_bool.ndim == 2:
        color = np.array([46, 204, 113], dtype=np.uint8)
        overlay[mask_bool] = (0.55 * overlay[mask_bool] + 0.45 * color).astype(np.uint8)
    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    if bbox is not None:
        draw.rectangle(tuple(bbox), outline="red", width=3)
    if point_xy is not None:
        x, y = point_xy
        r = 7
        draw.ellipse((x - r, y - r, x + r, y + r), fill="yellow", outline="black", width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _run_point_prompt_tracking(
    predictor,
    video_path: Path,
    point_xy: list[float],
    width: int,
    height: int,
    obj_id: int,
    max_frames: int,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch = _load_torch()
    point_norm = [
        max(0.0, min(float(point_xy[0]) / max(1.0, float(width)), 1.0)),
        max(0.0, min(float(point_xy[1]) / max(1.0, float(height)), 1.0)),
    ]
    points_tensor = torch.tensor([point_norm], dtype=torch.float32)
    labels_tensor = torch.tensor([1], dtype=torch.int32)

    session_response = predictor.handle_request(
        request={"type": "start_session", "resource_path": str(video_path)}
    )
    session_id = _response_session_id(session_response)
    records: list[dict[str, Any]] = []
    try:
        add_response = predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "obj_id": int(obj_id),
                "points": points_tensor,
                "point_labels": labels_tensor,
                "rel_coordinates": True,
            }
        )
        add_outputs = _response_outputs(add_response)
        initial_mask = _extract_mask(add_outputs, obj_id)
        if initial_mask is not None:
            bbox = _mask_to_bbox(initial_mask)
            mask_path = output_dir / "masks" / "frame_000000_mask.png"
            overlay_path = output_dir / "overlays" / "frame_000000_overlay.jpg"
            _save_mask(initial_mask, mask_path)
            _save_overlay(video_path, 0, initial_mask, point_xy, bbox, overlay_path)
            records.append(
                {
                    "frame_index": 0,
                    "bbox_xyxy": bbox,
                    "mask_path": str(mask_path),
                    "overlay_path": str(overlay_path),
                }
            )

        stream = predictor.handle_stream_request(
            request={"type": "propagate_in_video", "session_id": session_id}
        )
        seen = {0} if initial_mask is not None else set()
        for fallback_idx, response in enumerate(_iter_stream(stream)):
            outputs = _response_outputs(response)
            frame_idx = int(_first_present(outputs, ("frame_index", "frame_idx")) or fallback_idx)
            if frame_idx in seen:
                continue
            if len(records) >= max_frames:
                break
            mask = _extract_mask(outputs, obj_id)
            if mask is None:
                continue
            bbox = _mask_to_bbox(mask)
            mask_path = output_dir / "masks" / f"frame_{frame_idx:06d}_mask.png"
            overlay_path = output_dir / "overlays" / f"frame_{frame_idx:06d}_overlay.jpg"
            _save_mask(mask, mask_path)
            _save_overlay(video_path, frame_idx, mask, None, bbox, overlay_path)
            records.append(
                {
                    "frame_index": frame_idx,
                    "bbox_xyxy": bbox,
                    "mask_path": str(mask_path),
                    "overlay_path": str(overlay_path),
                }
            )
            seen.add(frame_idx)
    finally:
        try:
            predictor.handle_request(request={"type": "close_session", "session_id": session_id})
        except Exception:
            pass

    return records, {"session_id": session_id, "point_xy": point_xy, "point_norm": point_norm}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM 3.1 ModelScope video point-prompt smoke test.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to a VIDEO file.")
    parser.add_argument("--point", type=float, nargs=2, metavar=("X", "Y"), help="Positive point in first-frame pixels. Defaults to frame center.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="ModelScope model id.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional ModelScope cache directory.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional explicit checkpoint path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None, choices=("cpu", "cuda", "mps"))
    parser.add_argument("--version", default="3.1")
    parser.add_argument("--obj-id", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warm-up", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--async-loading-frames", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_video_path(args.input)
    torch = _load_torch()
    builder = _load_sam3_video_builder()

    device = args.device or _auto_device(torch)
    width, height = _frame_size(args.input)
    point_xy = [float(args.point[0]), float(args.point[1])] if args.point else [width / 2.0, height / 2.0]

    model_dir = _download_model(args.model_id, args.cache_dir)
    checkpoint_path = args.checkpoint or _find_checkpoint(model_dir)
    predictor, build_kwargs = _build_video_predictor(builder, checkpoint_path, device, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, prompt_metadata = _run_point_prompt_tracking(
        predictor=predictor,
        video_path=args.input,
        point_xy=point_xy,
        width=width,
        height=height,
        obj_id=args.obj_id,
        max_frames=args.max_frames,
        output_dir=args.output_dir,
    )

    metadata = {
        "model_id": args.model_id,
        "model_dir": str(model_dir),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "build_kwargs": {key: str(value) for key, value in build_kwargs.items()},
        "device": device,
        "input_video": str(args.input),
        "frame_size": [width, height],
        "obj_id": args.obj_id,
        "prompt": prompt_metadata,
        "num_frames": len(records),
        "frames": records,
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote SAM3.1 point-prompt outputs to: {args.output_dir}")
    print(f"Point: {point_xy} pixels")
    print(f"Tracked frames: {len(records)}")


if __name__ == "__main__":
    main()
