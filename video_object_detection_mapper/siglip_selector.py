from __future__ import annotations

from typing import Any

from video_object_detection_mapper import common


def load_siglip(args, torch, device: str):
    from transformers import AutoModel, AutoProcessor

    dtype = common.torch_dtype(torch, args.siglip_torch_dtype)
    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    model = AutoModel.from_pretrained(args.siglip_model, **kwargs)
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.siglip_model)
    return model, processor


def siglip_scores(torch, model, processor, images: list[Any], prompts: list[str], device: str) -> list[float]:
    inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
    inputs = common.move_inputs_to_device(dict(inputs), device)
    with torch.inference_mode():
        outputs = model(**inputs)
    logits = getattr(outputs, "logits_per_image", None)
    if logits is not None:
        probabilities = logits.sigmoid()
        return [float(row.mean().detach().cpu()) for row in probabilities]
    image_embeds = getattr(outputs, "image_embeds", None)
    text_embeds = getattr(outputs, "text_embeds", None)
    if image_embeds is None or text_embeds is None:
        fields = sorted(name for name in dir(outputs) if not name.startswith("_"))
        raise RuntimeError(f"SigLIP output has no logits or embeddings. Available fields: {fields}")
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    similarity = (image_embeds @ text_embeds.T + 1.0) / 2.0
    return [float(row.mean().detach().cpu()) for row in similarity]


def rank_candidates(
    args,
    np,
    torch,
    Image,
    image,
    detections: list[dict[str, Any]],
    masks: list[Any],
    scores: list[float | None],
    point_xy: list[float],
    target_object: str,
    siglip_model,
    siglip_processor,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    prompts = [target_object]
    rankable = []
    masked_images = []
    context_images = []
    crop_dir = args.output_dir / "siglip_candidate_crops"
    if args.save_candidate_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)

    for detection in detections:
        idx = int(detection["index"])
        bbox = detection.get("bbox_from_mask")
        area_fraction = common.mask_area_fraction(np, masks[idx])
        detection["mask_area_fraction"] = area_fraction
        detection["point_inside"] = common.point_inside_mask(np, masks[idx], point_xy)
        if bbox is None:
            detection["siglip_skipped"] = True
            detection["siglip_skip_reason"] = "empty_mask"
            continue
        if area_fraction < args.min_mask_area_fraction:
            detection["siglip_skipped"] = True
            detection["siglip_skip_reason"] = "mask_area_too_small"
            continue
        if area_fraction > args.max_mask_area_fraction:
            detection["siglip_skipped"] = True
            detection["siglip_skip_reason"] = "mask_area_too_large"
            continue
        masked_crop, context_crop = common.crop_candidate_views(
            Image,
            np,
            image,
            masks[idx],
            bbox,
            crop_padding=args.crop_padding,
            masked_fill=args.masked_fill,
        )
        if args.save_candidate_crops:
            masked_path = crop_dir / f"candidate_{idx:02d}_masked.jpg"
            context_path = crop_dir / f"candidate_{idx:02d}_context.jpg"
            masked_crop.save(masked_path)
            context_crop.save(context_path)
            detection["masked_crop_path"] = str(masked_path)
            detection["context_crop_path"] = str(context_path)
        rankable.append(detection)
        masked_images.append(masked_crop)
        context_images.append(context_crop)

    if not rankable:
        return detections, None

    masked_scores = siglip_scores(torch, siglip_model, siglip_processor, masked_images, prompts, device)
    context_scores = siglip_scores(torch, siglip_model, siglip_processor, context_images, prompts, device)
    for detection, masked_score, context_score in zip(rankable, masked_scores, context_scores):
        idx = int(detection["index"])
        sam3_score = scores[idx] if idx < len(scores) and scores[idx] is not None else 0.0
        point_bonus = 1.0 if detection.get("point_inside") else 0.0
        final_score = (
            args.siglip_masked_weight * masked_score
            + args.siglip_context_weight * context_score
            + args.siglip_sam3_score_weight * float(sam3_score)
            + args.siglip_point_inside_weight * point_bonus
        )
        detection.update(
            {
                "siglip_masked_score": masked_score,
                "siglip_context_score": context_score,
                "siglip_final_score": final_score,
                "siglip_skipped": False,
                "siglip_skip_reason": None,
            }
        )

    rankings = sorted(detections, key=lambda item: float(item.get("siglip_final_score", float("-inf"))), reverse=True)
    for rank, detection in enumerate(rankings, start=1):
        detection["siglip_rank"] = rank
    selected = rankings[0] if rankings and not rankings[0].get("siglip_skipped") else None
    return rankings, selected
