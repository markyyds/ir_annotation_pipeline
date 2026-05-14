from __future__ import annotations

from pathlib import Path
from typing import Any

from video_object_detection_mapper import common


def load_tracker(args, torch, device: str):
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor

    model_dir = common.snapshot_download_model(args.sam3_candidate_model, args.sam3_candidate_cache_dir)
    dtype = common.torch_dtype(torch, args.sam3_candidate_torch_dtype)
    model_kwargs: dict[str, Any] = {"local_files_only": True}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = Sam3TrackerModel.from_pretrained(str(model_dir), **model_kwargs)
    model.to(device)
    model.eval()
    processor = Sam3TrackerProcessor.from_pretrained(str(model_dir), local_files_only=True)
    return model, processor, str(model_dir)


def candidate_masks_from_output(np, torch, processor, outputs, original_sizes, binarize: bool):
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
    masks_np = common.tensor_to_numpy(np, torch, processed)
    masks_np = np.asarray(masks_np).squeeze()
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    elif masks_np.ndim > 3:
        masks_np = masks_np.reshape((-1,) + masks_np.shape[-2:])
    return [masks_np[idx] for idx in range(len(masks_np))]


def scores_from_output(np, torch, outputs) -> list[float | None]:
    for name in ("iou_scores", "pred_iou_scores", "scores"):
        value = getattr(outputs, name, None)
        if value is not None:
            scores = common.tensor_to_numpy(np, torch, value).reshape(-1).tolist()
            return [float(score) for score in scores]
    return []


def run_point_prompt(
    args,
    np,
    torch,
    model,
    processor,
    image,
    point_xy: list[float],
    device: str,
) -> tuple[list[Any], list[float | None], dict[str, Any]]:
    inputs = processor(
        images=image,
        input_points=[[[[float(point_xy[0]), float(point_xy[1])]]]],
        input_labels=[[[1]]],
        return_tensors="pt",
    )
    inputs = common.move_inputs_to_device(dict(inputs), device)
    with torch.inference_mode():
        try:
            outputs = model(**inputs, multimask_output=args.sam3_candidate_multimask_output)
        except TypeError:
            outputs = model(**inputs)
    masks = candidate_masks_from_output(np, torch, processor, outputs, inputs["original_sizes"], args.sam3_candidate_binarize)
    scores = scores_from_output(np, torch, outputs)
    metadata = {
        "input_point_xy": point_xy,
        "input_points_shape": list(inputs["input_points"].shape) if "input_points" in inputs else None,
        "original_sizes": common.tensor_to_numpy(np, torch, inputs["original_sizes"]).tolist() if "original_sizes" in inputs else None,
        "output_fields": sorted(name for name in dir(outputs) if not name.startswith("_")),
    }
    return masks, scores, metadata


def save_candidates(
    args,
    Image,
    ImageDraw,
    np,
    image,
    masks: list[Any],
    scores: list[float | None],
    point_xy: list[float],
) -> list[dict[str, Any]]:
    detections = []
    candidate_dir = args.output_dir / "sam3_candidates"
    for idx, mask in enumerate(masks[: args.sam3_candidate_max_masks]):
        bbox = common.mask_to_bbox(np, mask)
        mask_path = candidate_dir / f"candidate_mask_{idx:02d}.png"
        overlay_path = candidate_dir / f"candidate_overlay_{idx:02d}.jpg"
        common.save_mask_png(Image, np, mask, mask_path)
        common.save_overlay(Image, ImageDraw, np, image, mask, point_xy, bbox, overlay_path)
        detections.append(
            {
                "index": idx,
                "score": scores[idx] if idx < len(scores) else None,
                "bbox_from_mask": bbox,
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
            }
        )
    return detections
