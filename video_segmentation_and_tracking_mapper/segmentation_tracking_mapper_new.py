#!/usr/bin/env python3
"""Clean VLM/MolmoPoint/SAM video target segmentation mapper."""

from __future__ import annotations

import argparse
import csv
import inspect
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

import grounded_yoloe_sam2_video as base
from model_wrappers import ChatGenerationConfig, OpenAIChatClient, VLLMChatClient, image_part, text_part


DEFAULT_TEST_DATA = PROJECT_ROOT / "test_data"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "segmentation_tracking"
DEFAULT_GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-base"
DEFAULT_SAM2_MODEL = "facebook/sam2.1-hiera-large"
DEFAULT_SAM3_MODEL = "facebook/sam3.1"
DEFAULT_MOLMOPOINT_MODEL = "allenai/MolmoPoint-8B"
DEFAULT_TASK_INSTRUCTION_COLUMN = "other_information.language_instruction_2"
DEFAULT_OUTPUT_SIZE = [320, 180]
SYSTEM_PROMPT = "You are a precise robotic manipulation perception assistant. Return only valid JSON."

DIRECT_GROUNDING_PROMPT = """
Given the robot instruction, the first frame, and the final frame of the video,
identify the target object the robot should directly manipulate and localize it
in the FIRST frame.

Instruction: {instruction}
First-frame image size: {width} x {height} pixels

Return exactly this JSON schema:
{{
  "target_object": "short object name",
  "center": [cx, cy],
  "bbox": [x1, y1, x2, y2],
  "confidence": 0.0
}}

Use absolute pixel coordinates in the FIRST-frame image coordinate system.
Use the final frame only as temporal context. If the target object is not
visible in the first frame, use null for "center" and "bbox".
Return only valid JSON.

Keep your reasoning concise and precise.
""".strip()

TARGET_OBJECT_PROMPT = """
Given the robot instruction, the first frame, and the final frame of the video,
extract only the target object that the robot should directly manipulate.

Instruction: {instruction}

Return exactly this JSON schema:
{{
  "target_object": "short object name",
  "confidence": 0.0
}}

Use the final frame only as temporal context. Return only valid JSON.
""".strip()

POINT_TARGET_OBJECT_PROMPT = """
Given the robot instruction and the first frame image, extract only the target
object that the robot should directly manipulate.

Instruction: {instruction}

Return exactly this JSON schema:
{{
  "target_object": "short object name",
  "confidence": 0.0
}}

Do not return a point, bbox, or any localization. Return only valid JSON.
""".strip()

SELECT_BBOX_PROMPT = """
Given the robot instruction, the extracted target object, the first frame, the
final frame, and numbered candidate boxes from GroundingDINO, select the single
candidate that best localizes the target object in the FIRST frame.

Instruction: {instruction}
Target object: {target_object}
First-frame image size: {width} x {height} pixels

Candidate boxes are absolute xyxy coordinates in the first frame:
{candidates_json}

You will receive the first frame, the final frame, and then one annotated first
frame per candidate. Each annotated image contains exactly one candidate box.

Return exactly this JSON schema:
{{
  "selected_index": 0,
  "target_object": "{target_object}",
  "bbox": [x1, y1, x2, y2],
  "confidence": 0.0,
  "reason": "short reason"
}}

If none of the candidates matches the target object, use null for selected_index
and bbox. Return only valid JSON.

Keep your reasoning concise and precise.
""".strip()


class ChatJSONClient:
    def __init__(self, chat_client):
        self.chat_client = chat_client

    def chat_json(self, prompt: str, image_paths: list[Path] | Path) -> tuple[dict[str, Any], str, str]:
        if isinstance(image_paths, Path):
            image_paths = [image_paths]
        content = [text_part(prompt)]
        print(f"[ChatJSONClient] prompt: {prompt}")
        for image_path in image_paths:
            content.append(image_part(image_path))
        message = self.chat_client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
        )
        raw_content = base.assistant_content(message)
        reasoning = base.assistant_reasoning(message)
        return base.parse_vlm_json(raw_content), raw_content, reasoning


def snapshot_download_model(model_id: str, cache_dir: Path | None) -> Path:
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'modelscope'. Install it with: python -m pip install modelscope"
        ) from exc

    kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return Path(snapshot_download(model_id, **kwargs))


def first_present(mapping: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


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


def move_inputs_to_device(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def find_checkpoint(model_dir: Path) -> Path | None:
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
        if "image" in name or "video" in name:
            score += 2
        if name.endswith((".pt", ".pth")):
            score += 1
        return (-score, len(path.parts), str(path))

    return sorted(candidates, key=priority)[0]


class RuntimeContext:
    def __init__(self, args: argparse.Namespace):
        self.np, self.torch, self.imageio, self.pil_image, self.pil_draw = base.load_common_modules()
        self.device = base.auto_device(self.torch, args.device)
        self._model_client: ChatJSONClient | None = None
        self._grounding_dino: tuple[Any, Any, str, str] | None = None
        self._sam2: tuple[Any, Any, str, Any, str] | None = None
        self._sam3: tuple[Any, Any, str, Path | None, str] | None = None
        self._molmopoint: tuple[Any, Any, str, str | None, str] | None = None

    def model_client(self, args: argparse.Namespace) -> ChatJSONClient:
        if self._model_client is not None:
            return self._model_client
        generation = ChatGenerationConfig(
            max_tokens=args.vlm_max_new_tokens,
            temperature=args.vlm_temperature,
            top_p=args.vlm_top_p,
            top_k=args.vlm_top_k,
            min_p=args.vlm_min_p,
            presence_penalty=args.vlm_presence_penalty,
            repetition_penalty=args.vlm_repetition_penalty,
        )
        if args.model_backend == "openai":
            chat_client = OpenAIChatClient(
                model=args.vlm_model,
                url=args.openai_url,
                api_key=args.openai_api_key,
                generation=generation,
                timeout=args.vlm_timeout,
                include_extended_sampling=False,
                print_raw_response=args.print_raw_response,
            )
        else:
            chat_client = VLLMChatClient(
                model=args.vlm_model,
                base_url=args.vllm_base_url,
                api_key=args.vllm_api_key,
                generation=generation,
                timeout=args.vlm_timeout,
            )
        self._model_client = ChatJSONClient(chat_client)
        return self._model_client

    def grounding_dino(self, model_id: str):
        if self._grounding_dino is None or self._grounding_dino[2] != model_id or self._grounding_dino[3] != self.device:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
            model.eval()
            self._grounding_dino = (model, processor, model_id, self.device)
        return self._grounding_dino[0], self._grounding_dino[1]

    def sam2(self, model_id: str, dtype_name: str):
        dtype = base.torch_dtype(self.torch, dtype_name)
        if (
            self._sam2 is None
            or self._sam2[2] != model_id
            or self._sam2[3] != dtype
            or self._sam2[4] != self.device
        ):
            from transformers import Sam2VideoModel, Sam2VideoProcessor

            processor = Sam2VideoProcessor.from_pretrained(model_id)
            model = Sam2VideoModel.from_pretrained(model_id, torch_dtype=dtype)
            model.to(self.device)
            model.eval()
            self._sam2 = (model, processor, model_id, dtype, self.device)
        return self._sam2[0], self._sam2[1], self._sam2[3]

    def molmopoint(self, args: argparse.Namespace):
        device_map = None if args.molmopoint_device_map.lower() in {"", "none", "null"} else args.molmopoint_device_map
        cache_dir = args.molmopoint_cache_dir
        if (
            self._molmopoint is None
            or self._molmopoint[2] != args.molmopoint_model
            or self._molmopoint[3] != device_map
            or self._molmopoint[4] != self.device
        ):
            from transformers import AutoModelForImageTextToText, AutoProcessor

            model_dir = snapshot_download_model(args.molmopoint_model, cache_dir)
            model_kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "dtype": "auto",
            }
            if device_map:
                model_kwargs["device_map"] = device_map
            model = AutoModelForImageTextToText.from_pretrained(str(model_dir), **model_kwargs)
            if not device_map:
                model.to(self.device)
            model.eval()
            processor = AutoProcessor.from_pretrained(
                str(model_dir),
                trust_remote_code=True,
                padding_side="left",
            )
            self._molmopoint = (model, processor, args.molmopoint_model, device_map, self.device)
        return self._molmopoint[0], self._molmopoint[1]

    def sam3(self, args: argparse.Namespace):
        checkpoint = args.sam3_checkpoint
        checkpoint_key = str(checkpoint) if checkpoint is not None else ""
        if (
            self._sam3 is None
            or self._sam3[2] != args.sam3_model
            or self._sam3[3] != checkpoint_key
            or self._sam3[4] != self.device
        ):
            try:
                from sam3.model.sam3_image_processor import Sam3Processor
                from sam3.model_builder import build_sam3_image_model
            except ImportError as exc:
                raise RuntimeError(
                    "Missing dependency 'sam3'. Install it with: "
                    "python -m pip install git+https://github.com/facebookresearch/sam3.git"
                ) from exc

            model_dir = snapshot_download_model(args.sam3_model, args.sam3_cache_dir)
            checkpoint_path = checkpoint or find_checkpoint(model_dir)
            signature = inspect.signature(build_sam3_image_model)
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
            if "device" in signature.parameters:
                kwargs["device"] = self.device

            model = build_sam3_image_model(**kwargs)
            if hasattr(model, "to"):
                model = model.to(self.device)
            model.eval()

            processor_signature = inspect.signature(Sam3Processor)
            processor_kwargs: dict[str, Any] = {}
            if "confidence_threshold" in processor_signature.parameters:
                processor_kwargs["confidence_threshold"] = args.sam3_confidence_threshold
            elif "conf_threshold" in processor_signature.parameters:
                processor_kwargs["conf_threshold"] = args.sam3_confidence_threshold
            processor = Sam3Processor(model, **processor_kwargs)
            self._sam3 = (model, processor, args.sam3_model, checkpoint_key, self.device)
        return self._sam3[0], self._sam3[1]


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


def save_context_frames(args: argparse.Namespace, context: RuntimeContext) -> tuple[Path, Path, int, int, int, int]:
    first_frame = read_video_frame(context.imageio, args.video, args.first_video_frame_index)
    frame_height, frame_width = first_frame.shape[:2]
    count = video_frame_count(context.imageio, args.video)
    last_index = args.last_video_frame_index if args.last_video_frame_index >= 0 else (count - 1 if count else args.first_video_frame_index)
    last_frame = read_video_frame(context.imageio, args.video, last_index)
    first_path = args.output_dir / "first_frame.jpg"
    last_path = args.output_dir / "last_frame.jpg"
    base.save_rgb_image(context.pil_image, first_frame, first_path)
    base.save_rgb_image(context.pil_image, last_frame, last_path)
    return first_path, last_path, frame_width, frame_height, args.first_video_frame_index, int(last_index)


def extract_direct_grounding(
    client: ChatJSONClient,
    instruction: str,
    first_frame_path: Path,
    last_frame_path: Path,
    width: int,
    height: int,
) -> dict[str, Any]:
    prompt = DIRECT_GROUNDING_PROMPT.format(instruction=instruction, width=width, height=height)
    parsed, raw_content, reasoning = client.chat_json(prompt, [first_frame_path, last_frame_path])
    target_object = str(parsed.get("target_object") or parsed.get("object") or parsed.get("target") or "").strip()
    bbox = base.valid_box_or_none(parsed.get("bbox") or parsed.get("bbox_xyxy") or parsed.get("box"), width, height)
    if bbox is not None:
        center = [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
    else:
        center = normalize_point(parsed.get("center") or parsed.get("center_xy"), width, height)
    if not target_object:
        raise RuntimeError(f"VLM did not return target_object. Raw response: {raw_content}")
    if center is None:
        raise RuntimeError(f"VLM did not return usable center point. Raw response: {raw_content}")
    return {
        "target_object": target_object,
        "bbox_xyxy": bbox,
        "center_xy": center,
        "confidence": parsed.get("confidence"),
        "raw_response": parsed,
        "raw_content": raw_content,
        "reasoning": reasoning,
    }


def extract_target_object(
    client: ChatJSONClient,
    instruction: str,
    first_frame_path: Path,
    last_frame_path: Path,
) -> dict[str, Any]:
    prompt = TARGET_OBJECT_PROMPT.format(instruction=instruction)
    parsed, raw_content, reasoning = client.chat_json(prompt, [first_frame_path, last_frame_path])
    target_object = str(parsed.get("target_object") or parsed.get("object") or parsed.get("target") or "").strip()
    if not target_object:
        raise RuntimeError(f"VLM did not return target_object. Raw response: {raw_content}")
    return {
        "target_object": target_object,
        "confidence": parsed.get("confidence"),
        "raw_response": parsed,
        "raw_content": raw_content,
        "reasoning": reasoning,
    }


def extract_target_object_first_frame(
    client: ChatJSONClient,
    instruction: str,
    first_frame_path: Path,
) -> dict[str, Any]:
    prompt = POINT_TARGET_OBJECT_PROMPT.format(instruction=instruction)
    parsed, raw_content, reasoning = client.chat_json(prompt, first_frame_path)
    target_object = str(parsed.get("target_object") or parsed.get("object") or parsed.get("target") or "").strip()
    if not target_object:
        raise RuntimeError(f"VLM did not return target_object. Raw response: {raw_content}")
    return {
        "target_object": target_object,
        "confidence": parsed.get("confidence"),
        "raw_response": parsed,
        "raw_content": raw_content,
        "reasoning": reasoning,
    }


def normalize_point(value: Any, width: int, height: int) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        import re

        nums = re.findall(r"-?\d+(?:\.\d+)?", value)
        value = [float(num) for num in nums[:2]]
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x, y = float(value[0]), float(value[1])
    return [max(0.0, min(x, float(width))), max(0.0, min(y, float(height)))]


def normalize_molmopoint_points(points: Any, width: int, height: int) -> list[dict[str, Any]]:
    normalized = []
    for idx, point in enumerate(points or []):
        values = list(point)
        if len(values) < 4:
            continue
        object_id, image_num, x, y = values[:4]
        center = normalize_point([x, y], width, height)
        if center is None:
            continue
        normalized.append(
            {
                "index": idx,
                "object_id": int(object_id),
                "image_num": int(image_num),
                "center_xy": center,
            }
        )
    return normalized


def run_molmopoint_center_point(
    context: RuntimeContext,
    args: argparse.Namespace,
    model,
    processor,
    first_frame_path: Path,
    target_object: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    prompt = args.molmopoint_prompt_template.format(target_object=target_object)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": str(first_frame_path)},
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
    inputs = move_inputs_to_device(inputs, context.device)
    autocast_context = (
        context.torch.autocast("cuda", dtype=context.torch.bfloat16)
        if context.device.startswith("cuda")
        else base.null_context()
    )
    with context.torch.inference_mode(), autocast_context:
        output = model.generate(
            **inputs,
            logits_processor=model.build_logit_processor_from_inputs(inputs),
            max_new_tokens=args.molmopoint_max_new_tokens,
        )
    generated_tokens = output[:, inputs["input_ids"].size(1) :]
    generated_text = processor.post_process_image_text_to_text(
        generated_tokens,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]
    raw_points = model.extract_image_points(
        generated_text,
        metadata["token_pooling"],
        metadata["subpatch_mapping"],
        metadata["image_sizes"],
    )
    points = normalize_molmopoint_points(raw_points, width, height)
    if not points:
        raise RuntimeError(f"MolmoPoint returned no usable point for '{target_object}'. Raw text: {generated_text}")
    selected_point = points[0]
    return {
        "model": args.molmopoint_model,
        "prompt": prompt,
        "target_object": target_object,
        "center_xy": selected_point["center_xy"],
        "selected_point": selected_point,
        "points": points,
        "raw_points": raw_points,
        "generated_text": generated_text,
    }


def select_detection_with_model(
    client: ChatJSONClient,
    instruction: str,
    target_object: str,
    first_frame_path: Path,
    last_frame_path: Path,
    detections: list[dict[str, Any]],
    candidate_image_paths: list[str],
    width: int,
    height: int,
) -> dict[str, Any]:
    candidate_records = [
        {
            "index": int(det["candidate_index"]),
            "label": det.get("label") or target_object,
            "score": round(float(det.get("score") or 0.0), 4),
            "bbox_xyxy": [round(float(v), 2) for v in det["box_xyxy"]],
        }
        for det in detections
    ]
    prompt = SELECT_BBOX_PROMPT.format(
        instruction=instruction,
        target_object=target_object,
        width=width,
        height=height,
        candidates_json=json.dumps(candidate_records, ensure_ascii=False, indent=2),
    )
    parsed, raw_content, reasoning = client.chat_json(
        prompt,
        [first_frame_path, last_frame_path] + [Path(path) for path in candidate_image_paths],
    )
    selected_index = parsed.get("selected_index")
    try:
        selected_index = None if selected_index is None else int(selected_index)
    except Exception:
        selected_index = None
    selected = None
    if selected_index is not None:
        selected = next((det for det in detections if int(det["candidate_index"]) == selected_index), None)
    bbox = base.valid_box_or_none(parsed.get("bbox"), width, height)
    if bbox is None and selected is not None:
        bbox = selected["box_xyxy"]
    return {
        "selected_index": selected_index,
        "bbox_xyxy": bbox,
        "confidence": parsed.get("confidence"),
        "reason": parsed.get("reason"),
        "raw_response": parsed,
        "raw_content": raw_content,
        "reasoning": reasoning,
        "selected_detection": selected,
    }


def track_with_sam2_prompt(
    np,
    torch,
    model,
    processor,
    video_path: Path,
    dtype,
    device: str,
    binarize: bool,
    input_box: list[float] | None = None,
    input_point: list[float] | None = None,
) -> list[dict[str, Any]]:
    import transformers

    video_frames, _ = transformers.video_utils.load_video(str(video_path))
    inference_session = processor.init_video_session(video=video_frames, inference_device=device, dtype=dtype)
    kwargs: dict[str, Any] = {
        "inference_session": inference_session,
        "frame_idx": 0,
        "obj_ids": [0],
    }
    if input_point is not None:
        kwargs["input_points"] = [[[[float(input_point[0]), float(input_point[1])]]]]
        kwargs["input_labels"] = [[[1]]]
    elif input_box is not None:
        kwargs["input_boxes"] = [[[int(round(v)) for v in input_box]]]
    else:
        raise RuntimeError("SAM2 prompt requires either input_point or input_box")
    processor.add_inputs_to_inference_session(**kwargs)

    autocast_context = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else base.null_context()
    frame_records = []
    with torch.inference_mode(), autocast_context:
        _ = model(inference_session=inference_session, frame_idx=0)
        for fallback_frame_idx, output in enumerate(model.propagate_in_video_iterator(inference_session)):
            masks = processor.post_process_masks(
                [output.pred_masks],
                original_sizes=[[inference_session.video_height, inference_session.video_width]],
                binarize=binarize,
            )[0]
            mask = np.squeeze(masks[0])
            frame_records.append(
                {
                    "frame_index": int(getattr(output, "frame_idx", fallback_frame_idx)),
                    "mask": mask,
                    "bbox_xyxy": base.mask_to_bbox(np, mask),
                }
            )
    return frame_records


def run_sam3_text_prompt(torch, processor, image, prompt: str) -> dict[str, Any]:
    autocast_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else base.null_context()
    )
    with autocast_context:
        state = processor.set_image(image)
        try:
            return processor.set_text_prompt(state=state, prompt=prompt)
        except TypeError:
            return processor.set_text_prompt(prompt=prompt, state=state)


def sam3_records_from_output(np, torch, output: dict[str, Any]) -> list[dict[str, Any]]:
    masks = tensor_to_numpy(np, torch, first_present(output, ("masks", "pred_masks")))
    boxes = tensor_to_numpy(np, torch, first_present(output, ("boxes", "pred_boxes")))
    scores = tensor_to_numpy(np, torch, first_present(output, ("scores", "pred_scores", "iou_scores")))
    if masks is None:
        return []
    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    records = []
    for idx in range(len(masks)):
        mask = np.asarray(masks[idx]).squeeze()
        if mask.ndim != 2:
            continue
        bbox = base.mask_to_bbox(np, mask)
        if bbox is None and boxes is not None and len(boxes) > idx:
            bbox = [float(v) for v in np.asarray(boxes[idx]).reshape(-1).tolist()[:4]]
        score = None
        if scores is not None and len(scores) > idx:
            score = float(np.asarray(scores[idx]).reshape(-1)[0])
        records.append({"index": idx, "mask": mask, "bbox_xyxy": bbox, "score": score})
    return records


def box_center_distance(box: list[float] | None, point: list[float]) -> float:
    if box is None:
        return float("inf")
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = (float(box[1]) + float(box[3])) / 2.0
    return ((cx - float(point[0])) ** 2 + (cy - float(point[1])) ** 2) ** 0.5


def mask_contains_point(np, mask: Any, point: list[float]) -> bool:
    mask = np.asarray(mask).squeeze() > 0
    if mask.ndim != 2:
        return False
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    if y < 0 or y >= mask.shape[0] or x < 0 or x >= mask.shape[1]:
        return False
    return bool(mask[y, x])


def select_sam3_record(np, records: list[dict[str, Any]], point: list[float] | None) -> dict[str, Any] | None:
    if not records:
        return None
    if point is None:
        return max(records, key=lambda item: float(item.get("score") or 0.0))
    containing = [record for record in records if mask_contains_point(np, record["mask"], point)]
    candidates = containing or records
    return min(
        candidates,
        key=lambda item: (
            0 if item in containing else 1,
            box_center_distance(item.get("bbox_xyxy"), point),
            -float(item.get("score") or 0.0),
        ),
    )


def track_with_sam3_prompt(
    np,
    torch,
    processor,
    Image,
    imageio,
    video_path: Path,
    target_object: str,
    input_point: list[float],
    max_frames: int | None = None,
) -> list[dict[str, Any]]:
    frame_records = []
    selection_point = input_point
    reader = imageio.get_reader(str(video_path))
    try:
        for frame_idx, frame_rgb in enumerate(reader):
            if max_frames is not None and frame_idx >= max_frames:
                break
            image = Image.fromarray(frame_rgb).convert("RGB")
            with torch.inference_mode():
                output = run_sam3_text_prompt(torch, processor, image, target_object)
            records = sam3_records_from_output(np, torch, output)
            selected = select_sam3_record(np, records, selection_point)
            if selected is None:
                continue
            bbox = selected.get("bbox_xyxy")
            frame_records.append(
                {
                    "frame_index": int(frame_idx),
                    "mask": selected["mask"],
                    "bbox_xyxy": bbox,
                    "sam3_score": selected.get("score"),
                    "sam3_mask_index": selected.get("index"),
                }
            )
            if bbox is not None:
                selection_point = [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
    finally:
        reader.close()

    return frame_records


def output_coordinate_size(args: argparse.Namespace, frame_width: int, frame_height: int) -> tuple[int, int]:
    if args.output_coordinate_system == "input":
        return frame_width, frame_height
    return int(args.bbox_output_size[0]), int(args.bbox_output_size[1])


def materialize_frame_outputs(
    args: argparse.Namespace,
    context: RuntimeContext,
    raw_frame_records: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
    bbox_output_width: int,
    bbox_output_height: int,
) -> list[dict[str, Any]]:
    output_frames = []
    mask_dir = args.output_dir / "masks"
    for record in raw_frame_records:
        output_record = {
            "frame_index": int(record["frame_index"]),
            "bbox_xyxy_raw": record.get("bbox_xyxy"),
            "bbox_xyxy": base.scale_box(
                record.get("bbox_xyxy"),
                frame_width,
                frame_height,
                bbox_output_width,
                bbox_output_height,
            ),
        }
        if not args.no_save_masks:
            mask_path = mask_dir / f"frame_{int(record['frame_index']):06d}_target_mask.png"
            base.save_mask_png(context.pil_image, context.np, record["mask"], mask_path)
            output_record["mask_path"] = str(mask_path)
        output_frames.append(output_record)
    if not args.no_save_bbox_images:
        base.save_bbox_frames(context.imageio, context.pil_image, context.pil_draw, args.video, output_frames, args.output_dir / "bboxes")
    return output_frames


def run_pipeline(args: argparse.Namespace, context: RuntimeContext | None = None) -> dict[str, Any]:
    context = context or RuntimeContext(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}
    task_instruction = base.get_task_instruction(args.parquet, args.frame_column, args.task_instruction_column)
    first_frame_path, last_frame_path, frame_width, frame_height, first_idx, last_idx = save_context_frames(args, context)
    bbox_output_width, bbox_output_height = output_coordinate_size(args, frame_width, frame_height)
    client = context.model_client(args)

    target: dict[str, Any]
    selected: dict[str, Any]
    direct_grounding = None
    grounding_dino_payload = None
    vlm_selection = None
    molmopoint_payload = None

    if args.pipeline == "vlm_point_sam2":
        started = time.perf_counter()
        target = extract_target_object_first_frame(client, task_instruction, first_frame_path)
        timing["vlm_target_object_seconds"] = time.perf_counter() - started

        molmopoint_model, molmopoint_processor = context.molmopoint(args)
        started = time.perf_counter()
        molmopoint_payload = run_molmopoint_center_point(
            context,
            args,
            molmopoint_model,
            molmopoint_processor,
            first_frame_path,
            target["target_object"],
            frame_width,
            frame_height,
        )
        timing["molmopoint_seconds"] = time.perf_counter() - started
        selected = {
            "candidate_index": 0,
            "selection_method": "molmopoint_center_point",
            "target_object": target["target_object"],
            "box_xyxy": None,
            "center_xy": molmopoint_payload["center_xy"],
            "score": target.get("confidence"),
        }
        sam_point = molmopoint_payload["center_xy"]
        sam_box = None
    else:
        started = time.perf_counter()
        target = extract_target_object(client, task_instruction, first_frame_path, last_frame_path)
        timing["vlm_target_object_seconds"] = time.perf_counter() - started
        object_prompt = args.object_prompt or target["target_object"]
        gdino_model, gdino_processor = context.grounding_dino(args.grounding_dino_model)
        started = time.perf_counter()
        detections = base.run_grounding_dino(
            context.torch,
            context.pil_image,
            gdino_model,
            gdino_processor,
            first_frame_path,
            object_prompt,
            args.grounding_dino_box_threshold,
            args.grounding_dino_text_threshold,
            context.device,
        )
        candidates = base.normalize_detections(detections, frame_width, frame_height)
        timing["grounding_dino_seconds"] = time.perf_counter() - started
        if not candidates:
            raise RuntimeError(f"GroundingDINO produced no valid detections for prompt '{object_prompt}'")
        candidate_image_paths = base.draw_candidate_images(context.pil_image, context.pil_draw, first_frame_path, candidates, args.output_dir / "candidate_images")
        started = time.perf_counter()
        vlm_selection = select_detection_with_model(
            client,
            task_instruction,
            object_prompt,
            first_frame_path,
            last_frame_path,
            candidates,
            candidate_image_paths,
            frame_width,
            frame_height,
        )
        timing["vlm_bbox_selection_seconds"] = time.perf_counter() - started
        selected_detection = vlm_selection.get("selected_detection")
        selected_box = base.valid_box_or_none(vlm_selection.get("bbox_xyxy"), frame_width, frame_height)
        if selected_detection is None or selected_box is None:
            selected = base.choose_highest_confidence_detection(candidates)
            selection_fallback_reason = f"vlm_candidate_selection_failed: {vlm_selection.get('raw_content')}"
        else:
            selected = dict(selected_detection)
            selected["box_xyxy"] = selected_box
            selected["selection_method"] = "vlm_selected_candidate"
            selection_fallback_reason = None
        grounding_dino_payload = {
            "model": args.grounding_dino_model,
            "prompt": object_prompt,
            "box_threshold": args.grounding_dino_box_threshold,
            "text_threshold": args.grounding_dino_text_threshold,
            "candidates_raw": candidates,
            "candidates": [
                base.with_output_bbox(item, frame_width, frame_height, bbox_output_width, bbox_output_height)
                for item in candidates
            ],
            "candidate_image_paths": candidate_image_paths,
        }
        sam_point = None
        sam_box = selected["box_xyxy"]

    started = time.perf_counter()
    if args.sam_backend == "sam2":
        sam2_model, sam2_processor, sam2_dtype = context.sam2(args.sam2_model, args.torch_dtype)
        raw_frame_records = track_with_sam2_prompt(
            context.np,
            context.torch,
            sam2_model,
            sam2_processor,
            args.video,
            sam2_dtype,
            context.device,
            binarize=True,
            input_box=sam_box,
            input_point=sam_point,
        )
    else:
        if sam_point is None:
            raise RuntimeError("SAM3 backend currently requires a point prompt.")
        _sam3_model, sam3_processor = context.sam3(args)
        raw_frame_records = track_with_sam3_prompt(
            context.np,
            context.torch,
            sam3_processor,
            context.pil_image,
            context.imageio,
            args.video,
            target["target_object"],
            sam_point,
            max_frames=args.sam3_max_frames,
        )
    timing["sam_seconds"] = time.perf_counter() - started
    timing["sam2_seconds"] = timing["sam_seconds"] if args.sam_backend == "sam2" else 0.0
    timing["sam3_seconds"] = timing["sam_seconds"] if args.sam_backend == "sam3" else 0.0
    timing["vlm_total_seconds"] = timing.get("vlm_direct_grounding_seconds", 0.0) + timing.get("vlm_target_object_seconds", 0.0) + timing.get("vlm_bbox_selection_seconds", 0.0)
    timing["total_model_seconds"] = timing["vlm_total_seconds"] + timing.get("molmopoint_seconds", 0.0) + timing.get("grounding_dino_seconds", 0.0) + timing["sam_seconds"]

    output_frames = materialize_frame_outputs(args, context, raw_frame_records, frame_width, frame_height, bbox_output_width, bbox_output_height)
    evaluation = None
    if not args.skip_evaluation:
        raw_gt_by_frame = base.load_ground_truth_boxes(args.parquet, args.frame_column, args.gt_box_column)
        gt_by_frame = {
            frame_idx: base.scale_box(box, frame_width, frame_height, bbox_output_width, bbox_output_height)
            for frame_idx, box in raw_gt_by_frame.items()
        }
        evaluation = base.evaluate_bboxes(output_frames, gt_by_frame, bbox_output_width, bbox_output_height)

    payload = {
        "status": "ok",
        "pipeline": args.pipeline,
        "model_preset": args.model_preset,
        "model_backend": args.model_backend,
        "video_path": str(args.video),
        "parquet_path": str(args.parquet),
        "task_instruction": task_instruction,
        "first_frame": {
            "path": str(first_frame_path),
            "width": frame_width,
            "height": frame_height,
            "first_video_frame_index": first_idx,
            "last_frame_path": str(last_frame_path),
            "last_video_frame_index": last_idx,
            "bbox_output_width": bbox_output_width,
            "bbox_output_height": bbox_output_height,
            "output_coordinate_system": args.output_coordinate_system,
        },
        "vlm_target": target,
        "vlm_direct_grounding": direct_grounding,
        "molmopoint": molmopoint_payload,
        "grounding_dino": grounding_dino_payload,
        "vlm_selection": vlm_selection,
        "selected_prompt": base.with_output_bbox(selected, frame_width, frame_height, bbox_output_width, bbox_output_height)
        if selected.get("box_xyxy") is not None else selected,
        "selected_prompt_raw": selected,
        "timing_seconds": timing,
        "sam2": {
            "model": args.sam2_model,
            "torch_dtype": args.torch_dtype,
            "device": context.device,
            "num_frames": len(output_frames),
            "frames": output_frames,
        },
        "sam": {
            "backend": args.sam_backend,
            "model": args.sam2_model if args.sam_backend == "sam2" else args.sam3_model,
            "torch_dtype": args.torch_dtype if args.sam_backend == "sam2" else None,
            "device": context.device,
            "num_frames": len(output_frames),
            "frames": output_frames,
        },
        "evaluation": evaluation,
    }
    if args.pipeline == "vlm_dino_vlm_sam2":
        payload["selection_fallback_reason"] = locals().get("selection_fallback_reason")
    output_json = args.output_dir / "segmentation_tracking.json"
    output_json.write_text(json.dumps(base.json_ready(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote tracking output: {output_json}")
    print(f"Tracked frames: {len(output_frames)}")
    return {"output_json": output_json, "payload": payload}


def write_skipped_output(args: argparse.Namespace, reason: str) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "segmentation_tracking.json"
    payload = {
        "status": "skipped",
        "skip_reason": reason,
        "pipeline": args.pipeline,
        "model_preset": args.model_preset,
        "model_backend": args.model_backend,
        "video_path": str(args.video),
        "parquet_path": str(args.parquet),
        "sam2": {"frames": [], "num_frames": 0},
        "evaluation": None,
    }
    output_json.write_text(json.dumps(base.json_ready(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_json


SUMMARY_FIELDS = [
    "episode_id",
    "status",
    "model_preset",
    "pipeline",
    "model_backend",
    "num_frames",
    "num_valid_pairs",
    "mean_iou",
    "success_rate_iou_0_5",
    "mean_center_distance_px",
    "mean_normalized_center_distance",
    "mean_bbox_l1_px",
    "vlm_direct_grounding_seconds",
    "vlm_target_object_seconds",
    "vlm_bbox_selection_seconds",
    "vlm_total_seconds",
    "molmopoint_seconds",
    "grounding_dino_seconds",
    "sam_seconds",
    "sam2_seconds",
    "sam3_seconds",
    "total_model_seconds",
]


def summary_row(episode_id: str, status: str, args: argparse.Namespace, payload: dict[str, Any] | None) -> dict[str, Any]:
    row = {field: "" for field in SUMMARY_FIELDS}
    row["episode_id"] = episode_id
    row["status"] = status
    row["model_preset"] = args.model_preset
    row["pipeline"] = args.pipeline
    row["model_backend"] = args.model_backend
    if not payload:
        return row
    evaluation_summary = ((payload.get("evaluation") or {}).get("summary") or {})
    timing = payload.get("timing_seconds") or {}
    for key in SUMMARY_FIELDS[5:12]:
        row[key] = "" if evaluation_summary.get(key) is None else evaluation_summary.get(key)
    for key in SUMMARY_FIELDS[12:]:
        row[key] = "" if timing.get(key) is None else timing.get(key)
    return row


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    def format_value(value: Any) -> str:
        return f"{value:.10g}" if isinstance(value, float) else str(value)

    mean_row = {field: "" for field in SUMMARY_FIELDS}
    mean_row["episode_id"] = "MEAN"
    mean_row["status"] = "ok_only"
    for field in SUMMARY_FIELDS[5:]:
        values = [float(row[field]) for row in rows if row.get("status") == "ok" and row.get(field) not in ("", None)]
        mean_row[field] = sum(values) / len(values) if values else ""
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(SUMMARY_FIELDS) + "\n")
        for row in rows + [mean_row]:
            handle.write("\t".join(format_value(row.get(field, "")) for field in SUMMARY_FIELDS) + "\n")


def format_summary_value(value: Any) -> str:
    return f"{value:.10g}" if isinstance(value, float) else str(value)


def write_summary_header(handle) -> None:
    handle.write("\t".join(SUMMARY_FIELDS) + "\n")


def write_summary_row(handle, row: dict[str, Any]) -> None:
    handle.write("\t".join(format_summary_value(row.get(field, "")) for field in SUMMARY_FIELDS) + "\n")
    handle.flush()


def mean_summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mean_row = {field: "" for field in SUMMARY_FIELDS}
    mean_row["episode_id"] = "MEAN"
    mean_row["status"] = "ok_only"
    for field in SUMMARY_FIELDS[5:]:
        values = [float(row[field]) for row in rows if row.get("status") == "ok" and row.get(field) not in ("", None)]
        mean_row[field] = sum(values) / len(values) if values else ""
    return mean_row


def safe_name(value: Any) -> str:
    text = str(value).strip() if value is not None else "none"
    safe = []
    for char in text:
        if char.isalnum() or char in ("-", "_", "."):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "none"


def parse_model_preset(model_preset: str) -> tuple[str, str]:
    if model_preset.startswith("openai-"):
        return "openai", model_preset.removeprefix("openai-")
    if model_preset.startswith("vllm-"):
        return "vllm", model_preset.removeprefix("vllm-")
    raise ValueError("--model-preset must look like openai-{model_name} or vllm-{model_name}")


def apply_model_preset(args: argparse.Namespace) -> None:
    if not args.model_preset:
        args.model_preset = f"{args.model_backend}-{args.vlm_model}"
        return
    backend, model = parse_model_preset(args.model_preset)
    args.model_backend = backend
    args.vlm_model = model


def run_label(args: argparse.Namespace) -> str:
    return f"{safe_name(args.model_preset)}__{safe_name(args.pipeline)}__{safe_name(args.sam_backend)}"


def default_output_root(args: argparse.Namespace) -> Path:
    return DEFAULT_OUTPUT_ROOT / safe_name(args.model_preset) / safe_name(args.pipeline) / safe_name(args.sam_backend)


def episode_args(base_args: argparse.Namespace, parquet_path: Path, output_root: Path) -> argparse.Namespace:
    args = argparse.Namespace(**vars(base_args))
    args.parquet = parquet_path
    args.video = parquet_path.with_suffix(".mp4")
    args.output_dir = output_root / parquet_path.stem
    return args


def run_batch(args: argparse.Namespace) -> None:
    if args.test_data is None:
        run_pipeline(args)
        return
    output_root = args.output_root or default_output_root(args)
    output_root.mkdir(parents=True, exist_ok=True)
    parquets = sorted(args.test_data.glob(args.pattern))
    if not parquets:
        raise FileNotFoundError(f"No parquet files matched: {args.test_data / args.pattern}")
    context = RuntimeContext(args)
    rows = []
    summary_path = output_root / f"run_all_summary__{run_label(args)}.tsv"
    with summary_path.open("w", encoding="utf-8") as handle:
        write_summary_header(handle)
        for parquet_path in parquets:
            ep_args = episode_args(args, parquet_path, output_root)
            print(f"[{parquet_path.stem}] running {args.pipeline}")
            try:
                result = run_pipeline(ep_args, context)
                status = "ok"
                payload = result["payload"]
            except Exception as exc:
                if args.fail_on_skip or args.stop_on_error:
                    raise
                write_skipped_output(ep_args, str(exc))
                print(f"[{parquet_path.stem}] skipped: {exc}")
                status = "skipped"
                payload = None
            row = summary_row(parquet_path.stem, status, ep_args, payload)
            rows.append(row)
            write_summary_row(handle, row)
        write_summary_row(handle, mean_summary_row(rows))
    print(f"Wrote summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean target-object video segmentation and tracking mapper.")
    parser.add_argument("--pipeline", choices=("vlm_point_sam2", "vlm_dino_vlm_sam2"), default="vlm_point_sam2")
    parser.add_argument("--model-preset", help="Model preset: openai-{model_name} or vllm-{model_name}. Overrides --model-backend and --vlm-model.")
    parser.add_argument("--model-backend", choices=("openai", "vllm"), default="openai")
    parser.add_argument("--video", type=Path, default=DEFAULT_TEST_DATA / "episode_000000.mp4")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_TEST_DATA / "episode_000000.parquet")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "episode_000000")
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA, help="Batch mode directory with episode_*.parquet and matching .mp4 files.")
    parser.add_argument("--dataset-name", help=argparse.SUPPRESS)
    parser.add_argument("--pattern", default="episode_*.parquet")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--stop-on-error", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fail-on-skip", action="store_true")
    parser.add_argument("--task-instruction-column", default=DEFAULT_TASK_INSTRUCTION_COLUMN)
    parser.add_argument("--frame-column", default="frame_index")
    parser.add_argument("--gt-box-column", default="annotation.object_box")
    parser.add_argument("--first-video-frame-index", type=int, default=0)
    parser.add_argument("--last-video-frame-index", type=int, default=-1)
    parser.add_argument("--output-coordinate-system", choices=("320x180", "input"), default="320x180")
    parser.add_argument("--bbox-output-size", type=int, nargs=2, default=DEFAULT_OUTPUT_SIZE, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--no-save-masks", action="store_true")
    parser.add_argument("--no-save-bbox-images", action="store_true")
    parser.add_argument("--object-prompt")
    parser.add_argument("--grounding-dino-model", default=DEFAULT_GROUNDING_DINO_MODEL)
    parser.add_argument("--grounding-dino-box-threshold", type=float, default=0.08)
    parser.add_argument("--grounding-dino-text-threshold", type=float, default=0.08)
    parser.add_argument("--sam-backend", choices=("sam2", "sam3"), default="sam2")
    parser.add_argument("--sam2-model", default=DEFAULT_SAM2_MODEL)
    parser.add_argument("--sam3-model", default=DEFAULT_SAM3_MODEL)
    parser.add_argument("--sam3-cache-dir", type=Path)
    parser.add_argument("--sam3-checkpoint", type=Path)
    parser.add_argument("--sam3-confidence-threshold", type=float, default=0.0001)
    parser.add_argument("--sam3-max-frames", type=int, default=None)
    parser.add_argument("--torch-dtype", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--molmopoint-model", default=DEFAULT_MOLMOPOINT_MODEL)
    parser.add_argument("--molmopoint-cache-dir", type=Path)
    parser.add_argument("--molmopoint-device-map", default="auto")
    parser.add_argument("--molmopoint-max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--molmopoint-prompt-template",
        default="Point to the {target_object}",
    )
    parser.add_argument("--vlm-model", default=os.environ.get("MODEL_NAME", "qwen3-max"))
    parser.add_argument("--openai-url", default=os.environ.get("OPENAI_CHAT_COMPLETIONS_URL", "https://eval.dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"))
    parser.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--vllm-base-url", default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--vllm-api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--vlm-max-new-tokens", type=int, default=16384)
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
    apply_model_preset(args)
    try:
        run_batch(args)
    except Exception as exc:
        if args.fail_on_skip:
            raise
        output_json = write_skipped_output(args, str(exc))
        print(f"Skipped episode: {exc}")
        print(f"Wrote skipped output: {output_json}")


if __name__ == "__main__":
    main()
