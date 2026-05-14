from __future__ import annotations

import ast
import base64
import json
import re
from pathlib import Path
from typing import Any


def require(module_name: str, install_hint: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise RuntimeError(f"Missing dependency '{module_name}'. Install it with: {install_hint}") from exc


def load_common_modules():
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image, ImageDraw

    return np, torch, imageio, Image, ImageDraw


def null_context():
    class _NullContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    return _NullContext()


def auto_device(torch, requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_dtype(torch, name: str):
    return {
        "auto": None,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[name]


def snapshot_download_model(model_id: str, cache_dir: Path | None) -> Path:
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'modelscope'. Install it with: python -m pip install modelscope") from exc
    kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return Path(snapshot_download(model_id, **kwargs))


def move_inputs_to_device(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def tensor_to_numpy(np, torch, value: Any):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        if getattr(value, "dtype", None) == torch.bfloat16:
            value = value.float()
        return value.numpy()
    return np.asarray(value)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return json_ready(value.tolist())
    if hasattr(value, "item"):
        try:
            return json_ready(value.item())
        except Exception:
            pass
    return value


def image_part(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower().lstrip(".") or "jpeg"
    mime = "jpeg" if suffix == "jpg" else suffix
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{data}"}}


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def assistant_content(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, list):
        return "\n".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    return str(content)


def assistant_reasoning(message: dict[str, Any]) -> str:
    return str(message.get("reasoning_content") or message.get("reasoning") or "")


def parse_vlm_json(raw_content: str) -> dict[str, Any]:
    text = raw_content.strip()
    for marker in ("```json", "```", "JSON:", "Response:"):
        text = text.replace(marker, "")
    text = text.strip()
    try:
        result = json.loads(text, strict=False)
    except Exception:
        try:
            result = ast.literal_eval(text)
        except Exception:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                return {}
            try:
                result = json.loads(match.group(0), strict=False)
            except Exception:
                return {}
    if isinstance(result, list) and result:
        result = result[0]
    return result if isinstance(result, dict) else {}


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


def parse_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text in {"", "[]", "-1", "None", "nan"} else text


def get_task_instruction(parquet_path: Path, task_instruction_column: str) -> str:
    rows = load_rows(parquet_path)
    if not rows:
        raise RuntimeError(f"Empty parquet: {parquet_path}")
    text = parse_text(rows[0].get(task_instruction_column))
    if not text:
        raise KeyError(f"Task instruction column '{task_instruction_column}' is missing or empty")
    return text


def read_video_frame(imageio, video_path: Path, frame_index: int):
    reader = imageio.get_reader(str(video_path))
    try:
        return reader.get_data(frame_index)
    finally:
        reader.close()


def video_frame_count(imageio, video_path: Path) -> int | None:
    try:
        reader = imageio.get_reader(str(video_path))
        try:
            count = reader.count_frames()
            return int(count) if count and count > 0 else None
        finally:
            reader.close()
    except Exception:
        return None


def save_rgb_image(Image, frame_rgb, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame_rgb).convert("RGB").save(output_path)


def save_context_frames(args, imageio, Image) -> tuple[Path, Path, int, int, int, int]:
    first_frame = read_video_frame(imageio, args.video, args.first_video_frame_index)
    frame_height, frame_width = first_frame.shape[:2]
    count = video_frame_count(imageio, args.video)
    last_index = args.last_video_frame_index if args.last_video_frame_index >= 0 else (count - 1 if count else args.first_video_frame_index)
    last_frame = read_video_frame(imageio, args.video, last_index)
    first_path = args.output_dir / "first_frame.jpg"
    last_path = args.output_dir / "last_frame.jpg"
    save_rgb_image(Image, first_frame, first_path)
    save_rgb_image(Image, last_frame, last_path)
    return first_path, last_path, frame_width, frame_height, args.first_video_frame_index, int(last_index)


def valid_box_or_none(value: Any, width: int, height: int) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", value)
        value = [float(num) for num in nums[:4]]
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    x1, y1, x2, y2 = [float(item) for item in value[:4]]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0.0, min(x1, float(width)))
    x2 = max(0.0, min(x2, float(width)))
    y1 = max(0.0, min(y1, float(height)))
    y2 = max(0.0, min(y2, float(height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def normalize_point(value: Any, width: int, height: int) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", value)
        value = [float(num) for num in nums[:2]]
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x, y = float(value[0]), float(value[1])
    return [max(0.0, min(x, float(width))), max(0.0, min(y, float(height)))]


def mask_to_bbox(np, mask: Any) -> list[float] | None:
    mask_bool = np.asarray(mask).squeeze() > 0
    if mask_bool.ndim != 2 or not mask_bool.any():
        return None
    ys, xs = np.where(mask_bool)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def mask_area_fraction(np, mask: Any) -> float:
    mask_bool = np.asarray(mask).squeeze() > 0
    if mask_bool.ndim != 2 or mask_bool.size == 0:
        return 0.0
    return float(mask_bool.mean())


def point_inside_mask(np, mask: Any, point_xy: list[float]) -> bool:
    mask_bool = np.asarray(mask).squeeze() > 0
    if mask_bool.ndim != 2:
        return False
    x = int(round(float(point_xy[0])))
    y = int(round(float(point_xy[1])))
    if y < 0 or y >= mask_bool.shape[0] or x < 0 or x >= mask_bool.shape[1]:
        return False
    return bool(mask_bool[y, x])


def scale_box(box: Any, src_width: int, src_height: int, dst_width: int, dst_height: int) -> list[float] | None:
    box = valid_box_or_none(box, src_width, src_height)
    if box is None:
        return None
    sx = float(dst_width) / max(1.0, float(src_width))
    sy = float(dst_height) / max(1.0, float(src_height))
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]


def save_mask_png(Image, np, mask: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = (np.asarray(mask).squeeze() > 0).astype("uint8") * 255
    Image.fromarray(mask_u8).save(output_path)


def save_overlay(Image, ImageDraw, np, image, mask: Any, point_xy: list[float] | None, bbox: list[float] | None, output_path: Path) -> None:
    overlay = np.asarray(image.convert("RGB")).copy()
    mask_bool = np.asarray(mask).squeeze() > 0
    if mask_bool.ndim == 2:
        color = np.array([46, 204, 113], dtype=np.uint8)
        overlay[mask_bool] = (0.55 * overlay[mask_bool] + 0.45 * color).astype(np.uint8)
    annotated = Image.fromarray(overlay)
    draw = ImageDraw.Draw(annotated)
    if bbox is not None:
        draw.rectangle(tuple(bbox), outline="red", width=3)
    if point_xy is not None:
        x, y = point_xy
        radius = 7
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="yellow", outline="black", width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)


def annotate_points(Image, ImageDraw, frame_path: Path, points: list[dict[str, Any]], selected_index: int | None, output_path: Path) -> None:
    image = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = ["#ff3b30", "#34c759", "#007aff", "#ffcc00", "#af52de", "#ff9500"]
    radius = max(6, round(min(image.size) * 0.015))
    for point in points:
        x, y = point["center_xy"]
        idx = int(point["index"])
        color = "#ffcc00" if selected_index is not None and idx == selected_index else colors[idx % len(colors)]
        width = 5 if selected_index is not None and idx == selected_index else 3
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=width)
        draw.line((x - radius * 2, y, x + radius * 2, y), fill="white", width=2)
        draw.line((x, y - radius * 2, x, y + radius * 2), fill="white", width=2)
        draw.text((x + radius + 4, max(0, y - radius - 4)), f"{idx} ({round(x)}, {round(y)})", fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def expanded_box(box: list[float], width: int, height: int, padding: float) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * padding
    pad_y = bh * padding
    return [
        int(max(0, round(x1 - pad_x))),
        int(max(0, round(y1 - pad_y))),
        int(min(width, round(x2 + pad_x))),
        int(min(height, round(y2 + pad_y))),
    ]


def crop_candidate_views(Image, np, image, mask: Any, bbox: list[float], crop_padding: float, masked_fill: int):
    width, height = image.size
    tight = expanded_box(bbox, width, height, 0.0)
    context = expanded_box(bbox, width, height, crop_padding)
    image_arr = np.asarray(image.convert("RGB"))
    mask_bool = np.asarray(mask).squeeze() > 0
    fill = np.full_like(image_arr, int(masked_fill), dtype=np.uint8)
    masked_arr = np.where(mask_bool[..., None], image_arr, fill)
    return Image.fromarray(masked_arr).crop(tuple(tight)), image.crop(tuple(context)).convert("RGB")
