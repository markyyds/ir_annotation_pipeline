#!/usr/bin/env python3
"""Standalone VLM target-object grounding for test_data videos/parquets."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import json
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_TEST_DATA = PROJECT_ROOT / "test_data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_FRAME_COLUMN = "frame_index"
DEFAULT_TASK_INSTRUCTION_COLUMN = "other_information.language_instruction_2"

SYSTEM_PROMPT = "You are a precise robotic manipulation perception assistant. Return only valid JSON."

INPUT_TEMPLATE = """
Given the robot language instruction below and the first frame of the video,
identify the target object being manipulated and localize it in the image.

Instruction: {instruction}

Return exactly this JSON schema:
{{
  "target_object": "short object name",
  "bbox": [x1, y1, x2, y2],
  "center": [cx, cy],
  "confidence": 0.0
}}

Use absolute pixel coordinates in the image coordinate system, with the origin
at the top-left corner. If the target object is not visible, use null for
"bbox" and "center".
""".strip()


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
    return "" if text in {"", "[]", "-1", "None", "nan"} else text


def get_task_instruction(rows: list[dict[str, Any]], column: str) -> str:
    if not rows:
        return ""
    text = parse_text(rows[0].get(column))
    if not text:
        raise KeyError(f"Task instruction column '{column}' is missing or empty")
    return text


def open_video_reader(video_path: Path):
    try:
        import imageio.v2 as imageio

        return imageio.get_reader(str(video_path))
    except ValueError as exc:
        raise RuntimeError(
            "Could not open video with imageio. Install video support with: "
            "python -m pip install 'imageio[ffmpeg]'"
        ) from exc


def read_frame_with_ffmpeg(video_path: Path, frame_idx: int, output_path: Path) -> Image.Image:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{frame_idx})",
        "-frames:v",
        "1",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return Image.open(output_path).convert("RGB")


def read_frame(video_path: Path, frame_idx: int, ffmpeg_output_path: Path) -> Image.Image:
    try:
        reader = open_video_reader(video_path)
        try:
            return Image.fromarray(reader.get_data(frame_idx)).convert("RGB")
        finally:
            reader.close()
    except RuntimeError:
        pass

    try:
        import cv2
    except ImportError:
        return read_frame_with_ffmpeg(video_path, frame_idx, ffmpeg_output_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return read_frame_with_ffmpeg(video_path, frame_idx, ffmpeg_output_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        return read_frame_with_ffmpeg(video_path, frame_idx, ffmpeg_output_path)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb).convert("RGB")


def resize_image(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale >= 1.0:
        return image
    return image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)


def save_first_frame(
    video_path: Path,
    frame_idx: int,
    output_path: Path,
    max_side: int,
    quality: int,
) -> tuple[Path, int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = read_frame(video_path, frame_idx, output_path)
    width, height = image.size
    resized = resize_image(image, max_side)
    resized.save(output_path, quality=quality)
    return output_path, width, height


def encode_image_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def strip_json_markers(raw_output: str) -> str:
    text = raw_output.strip()
    for marker in ["```json", "```", "JSON:", "Response:"]:
        text = text.replace(marker, "")
    return text.strip()


def parse_output(raw_output: str) -> dict[str, Any]:
    json_str = strip_json_markers(raw_output)
    try:
        result = json.loads(json_str, strict=False)
    except Exception:
        try:
            result = ast.literal_eval(json_str)
        except Exception:
            match = re.search(r"\{.*\}", json_str, flags=re.S)
            if not match:
                return {}
            try:
                result = json.loads(match.group(0), strict=False)
            except Exception:
                return {}

    if isinstance(result, list) and result:
        result = result[0]
    return result if isinstance(result, dict) else {}


def normalize_bbox(bbox: Any, width: int, height: int) -> list[float] | None:
    if bbox is None:
        return None
    if isinstance(bbox, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", bbox)
        bbox = [float(num) for num in nums[:4]]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
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


def normalize_center(center: Any, width: int, height: int) -> list[float] | None:
    if center is None:
        return None
    if isinstance(center, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", center)
        center = [float(num) for num in nums[:2]]
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        return None
    cx, cy = [float(v) for v in center]
    return [max(0.0, min(cx, float(width))), max(0.0, min(cy, float(height)))]


def postprocess_result(parsed: dict[str, Any], width: int, height: int, frame_path: Path) -> dict[str, Any]:
    target_object = parsed.get("target_object") or parsed.get("object") or parsed.get("target") or parsed.get("label")
    bbox = normalize_bbox(parsed.get("bbox") or parsed.get("bbox_2d") or parsed.get("box"), width, height)
    if bbox is None:
        center = normalize_center(parsed.get("center") or parsed.get("center_xy"), width, height)
    else:
        center = [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
    result = {
        "target_object": str(target_object).strip() if target_object else "",
        "bbox_xyxy": bbox,
        "center_xy": center,
        "frame_size": [width, height],
        "first_frame_path": str(frame_path),
        "raw_response": parsed,
    }
    if "confidence" in parsed:
        result["confidence"] = parsed["confidence"]
    return result


def assistant_text(response: dict[str, Any]) -> str:
    message = response["choices"][0]["message"]
    content = message.get("content")
    if content:
        return str(content)
    reasoning = message.get("reasoning")
    if reasoning:
        return str(reasoning)
    return ""


def get_vllm_models(base_url: str, api_key: str) -> list[str]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    models = payload.get("data", [])
    return [str(model["id"]) for model in models if model.get("id")]


class VLLMTargetGrounder:
    def __init__(self, model_id: str, base_url: str, api_key: str, max_tokens: int, temperature: float, timeout: int):
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

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
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 404:
                try:
                    models = get_vllm_models(self.base_url, self.api_key)
                except Exception:
                    models = []
                available = ", ".join(models) if models else "unknown"
                raise RuntimeError(
                    f"VLM API request failed with HTTP 404. Requested model `{self.model_id}` "
                    f"was not found. Available models: {available}."
                ) from exc
            raise RuntimeError(f"VLM API request failed with HTTP {exc.code}: {body}") from exc
        return assistant_text(data)

    def ground(self, instruction: str, frame_path: Path) -> dict[str, Any]:
        prompt = INPUT_TEMPLATE.format(instruction=instruction)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": encode_image_data_url(frame_path)}},
                ],
            },
        ]
        return parse_output(self.chat(messages))


def maybe_make_grounder(args: argparse.Namespace) -> VLLMTargetGrounder | None:
    if not args.use_vlm:
        return None
    model_id = args.vlm_model
    if model_id == "auto":
        models = get_vllm_models(args.vlm_base_url, args.vlm_api_key)
        if not models:
            raise RuntimeError(f"No served models found at {args.vlm_base_url.rstrip('/')}/models")
        model_id = models[0]
        print(f"[vllm] using served model from /models: {model_id}")
    return VLLMTargetGrounder(
        model_id=model_id,
        base_url=args.vlm_base_url,
        api_key=args.vlm_api_key,
        max_tokens=args.vlm_max_new_tokens,
        temperature=args.vlm_temperature,
        timeout=args.vlm_timeout,
    )


def draw_overlay(frame_path: Path, result: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    bbox = result.get("bbox_xyxy")
    if bbox is not None:
        draw.rectangle(tuple(bbox), outline=(255, 40, 40), width=4)
        label = result.get("target_object") or "target"
        draw.text((bbox[0], max(0, bbox[1] - 16)), label, fill=(255, 40, 40))
    center = result.get("center_xy")
    if center is not None:
        cx, cy = center
        r = 5
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(40, 200, 255))
    image.save(output_path, quality=95)
    return output_path


def process_episode(parquet_path: Path, video_path: Path, output_dir: Path, args: argparse.Namespace, grounder) -> dict[str, Any]:
    rows = load_rows(parquet_path)
    if not rows:
        raise ValueError(f"{parquet_path} is empty")
    rows = sorted(rows, key=lambda row: int(row[args.frame_column]))
    first_frame = int(rows[0][args.frame_column])
    instruction = get_task_instruction(rows, args.task_instruction_column)

    episode_id = parquet_path.stem
    episode_dir = output_dir / episode_id
    frame_path = episode_dir / "first_frame.jpg"
    overlay_path = episode_dir / "target_object_grounding_overlay.jpg"

    frame_path, width, height = save_first_frame(
        video_path,
        first_frame,
        frame_path,
        max_side=args.vlm_image_max_side,
        quality=args.vlm_image_quality,
    )

    if grounder is None:
        parsed = {}
        result = {
            "target_object": "",
            "bbox_xyxy": None,
            "center_xy": None,
            "frame_size": [width, height],
            "first_frame_path": str(frame_path),
            "raw_response": {},
            "error": "vlm_disabled",
        }
    else:
        parsed = grounder.ground(instruction, frame_path)
        result = postprocess_result(parsed, width, height, frame_path)

    draw_overlay(frame_path, result, overlay_path)
    payload = {
        "episode_id": episode_id,
        "video": str(video_path),
        "parquet": str(parquet_path),
        "task_instruction_column": args.task_instruction_column,
        "task_instruction": instruction,
        "first_frame_index": first_frame,
        "annotation_fields_used": False,
        "target_object_grounding": result,
        "overlay_path": str(overlay_path),
    }
    episode_dir.mkdir(parents=True, exist_ok=True)
    output_path = episode_dir / "target_object_grounding.json"
    output_path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["episode_id", "target_object", "bbox_xyxy", "center_xy", "confidence", "output_path"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone first-frame target-object grounding using video and parquet task instruction."
    )
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern", default="episode_*.parquet")
    parser.add_argument("--frame-column", default=DEFAULT_FRAME_COLUMN)
    parser.add_argument("--task-instruction-column", default=DEFAULT_TASK_INSTRUCTION_COLUMN)
    parser.add_argument("--use-vlm", action="store_true")
    parser.add_argument("--vlm-model", default="auto", help="Served vLLM model id. Use auto for first /v1/models entry.")
    parser.add_argument("--vlm-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--vlm-api-key", default="EMPTY")
    parser.add_argument("--vlm-max-new-tokens", type=int, default=512)
    parser.add_argument("--vlm-temperature", type=float, default=0.0)
    parser.add_argument("--vlm-timeout", type=int, default=300)
    parser.add_argument("--vlm-image-max-side", type=int, default=768)
    parser.add_argument("--vlm-image-quality", type=int, default=90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (1 <= args.vlm_image_quality <= 100):
        raise ValueError("--vlm-image-quality must be between 1 and 100")
    parquets = sorted(args.test_data.glob(args.pattern))
    if not parquets:
        raise FileNotFoundError(f"No parquets matched {args.test_data / args.pattern}")

    grounder = maybe_make_grounder(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for parquet_path in parquets:
        video_path = args.test_data / f"{parquet_path.stem}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video for {parquet_path}: {video_path}")
        print(f"[{parquet_path.stem}] grounding target object")
        payload = process_episode(parquet_path, video_path, args.output_dir, args, grounder)
        result = payload["target_object_grounding"]
        output_path = args.output_dir / payload["episode_id"] / "target_object_grounding.json"
        summaries.append(
            {
                "episode_id": payload["episode_id"],
                "target_object": result.get("target_object", ""),
                "bbox_xyxy": json.dumps(result.get("bbox_xyxy")),
                "center_xy": json.dumps(result.get("center_xy")),
                "confidence": result.get("confidence"),
                "output_path": str(output_path),
            }
        )

    summary_json = args.output_dir / "target_object_grounding_summary.json"
    summary_csv = args.output_dir / "target_object_grounding_summary.csv"
    summary_json.write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_summary_csv(summary_csv, summaries)
    print(f"Wrote summary JSON: {summary_json}")
    print(f"Wrote summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
