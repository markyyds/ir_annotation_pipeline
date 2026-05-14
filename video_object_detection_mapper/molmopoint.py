from __future__ import annotations

from pathlib import Path
from typing import Any

from video_object_detection_mapper import common


def load_model_and_processor(args, device: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device_map = None if args.molmopoint_device_map.lower() in {"", "none", "null"} else args.molmopoint_device_map
    model_dir = common.snapshot_download_model(args.molmopoint_model, args.molmopoint_cache_dir)
    model_kwargs: dict[str, Any] = {"trust_remote_code": True, "dtype": "auto"}
    if device_map:
        model_kwargs["device_map"] = device_map
    model = AutoModelForImageTextToText.from_pretrained(str(model_dir), **model_kwargs)
    if not device_map:
        model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True, padding_side="left")
    return model, processor, str(model_dir), device_map


def normalize_points(points: Any, width: int, height: int) -> list[dict[str, Any]]:
    normalized = []
    for idx, point in enumerate(points or []):
        values = list(point)
        if len(values) < 4:
            continue
        object_id, image_num, x, y = values[:4]
        center = common.normalize_point([x, y], width, height)
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


def run_pointing(
    args,
    np,
    torch,
    Image,
    ImageDraw,
    model,
    processor,
    first_frame_path: Path,
    referring_expression: str,
    width: int,
    height: int,
    device: str,
) -> dict[str, Any]:
    prompt = args.molmopoint_prompt_template.format(referring_expression=referring_expression)
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
    inputs = common.move_inputs_to_device(inputs, device)
    autocast_context = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else common.null_context()
    with torch.inference_mode(), autocast_context:
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
    points = normalize_points(raw_points, width, height)
    if not points:
        raise RuntimeError(f"MolmoPoint returned no usable point for '{referring_expression}'. Raw text: {generated_text}")
    selected_point = points[0]
    annotated_path = args.output_dir / "molmopoint_first_frame_points.jpg"
    if args.save_molmopoint_visualization:
        common.annotate_points(Image, ImageDraw, first_frame_path, points, int(selected_point["index"]), annotated_path)
    return {
        "model": args.molmopoint_model,
        "prompt": prompt,
        "referring_expression": referring_expression,
        "center_xy": selected_point["center_xy"],
        "selected_point": selected_point,
        "points": points,
        "raw_points": common.json_ready(raw_points),
        "generated_text": generated_text,
        "annotated_path": str(annotated_path) if args.save_molmopoint_visualization else None,
    }
