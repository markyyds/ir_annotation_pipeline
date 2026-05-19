from __future__ import annotations

from pathlib import Path
from typing import Any

from video_object_detection_mapper import common


def load_video_tracker(args, torch, device: str):
    try:
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'transformers' with Sam3TrackerVideoModel/Sam3TrackerVideoProcessor. "
            "Install a SAM3-video-capable Transformers build in the main .venv."
        ) from exc

    model_dir = common.snapshot_download_model(args.sam3_video_model, args.sam3_video_cache_dir)
    dtype = common.torch_dtype(torch, args.sam3_video_torch_dtype)
    model_kwargs: dict[str, Any] = {"local_files_only": True}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = Sam3TrackerVideoModel.from_pretrained(str(model_dir), **model_kwargs)
    if dtype is not None:
        model = model.to(device, dtype=dtype)
    else:
        model = model.to(device)
    model.eval()
    processor = Sam3TrackerVideoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    return model, processor, {
        "model": args.sam3_video_model,
        "model_dir": str(model_dir),
        "torch_dtype": args.sam3_video_torch_dtype,
        "device": device,
        "loader": "transformers.Sam3TrackerVideoModel",
    }


def point_from_bbox_or_point(bbox: list[float] | None, point_xy: list[float]) -> list[float]:
    if bbox is not None:
        return [(float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0]
    return [float(point_xy[0]), float(point_xy[1])]


def first_mask_from_postprocessed(np, masks: Any):
    masks = np.asarray(masks).squeeze()
    if masks.ndim == 2:
        return masks
    if masks.ndim >= 3:
        return masks.reshape((-1,) + masks.shape[-2:])[0]
    return None


def postprocess_output_mask(np, processor, inference_session, output, binarize: bool):
    processed = processor.post_process_masks(
        [output.pred_masks],
        original_sizes=[[inference_session.video_height, inference_session.video_width]],
        binarize=binarize,
    )[0]
    return first_mask_from_postprocessed(np, processed)


def track_video(
    np,
    torch,
    model,
    processor,
    video_path: Path,
    bbox: list[float] | None,
    mask: Any | None,
    point_xy: list[float],
    frame_width: int,
    frame_height: int,
    obj_id: int,
    device: str,
    dtype_name: str,
    binarize: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import transformers

    dtype = common.torch_dtype(torch, dtype_name)
    video_frames, _metadata = transformers.video_utils.load_video(str(video_path))
    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        dtype=dtype,
    )

    prompt_point = point_from_bbox_or_point(bbox, point_xy)
    processor.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=0,
        obj_ids=int(obj_id),
        input_points=[[[[float(prompt_point[0]), float(prompt_point[1])]]]],
        input_labels=[[[1]]],
    )

    frame_records: list[dict[str, Any]] = []
    autocast_context = torch.autocast("cuda", dtype=dtype) if device.startswith("cuda") and dtype is not None else common.null_context()
    seen: set[int] = set()
    with torch.inference_mode(), autocast_context:
        initial_output = model(inference_session=inference_session, frame_idx=0)
        initial_mask = postprocess_output_mask(np, processor, inference_session, initial_output, binarize)
        if initial_mask is not None:
            frame_records.append(
                {
                    "frame_index": 0,
                    "mask": initial_mask,
                    "bbox_xyxy": common.mask_to_bbox(np, initial_mask),
                }
            )
            seen.add(0)

        for fallback_frame_idx, output in enumerate(model.propagate_in_video_iterator(inference_session)):
            frame_idx = int(getattr(output, "frame_idx", fallback_frame_idx))
            if frame_idx in seen:
                continue
            out_mask = postprocess_output_mask(np, processor, inference_session, output, binarize)
            if out_mask is None:
                continue
            frame_records.append(
                {
                    "frame_index": frame_idx,
                    "mask": out_mask,
                    "bbox_xyxy": common.mask_to_bbox(np, out_mask),
                }
            )
            seen.add(frame_idx)

    return sorted(frame_records, key=lambda item: item["frame_index"]), {
        "tracking_prompt_type": "point_from_selected_bbox" if bbox is not None else "molmopoint",
        "tracking_prompt_point_xy": prompt_point,
        "tracking_prompt_obj_id": int(obj_id),
        "tracking_uses_selected_mask_as_prompt": False,
        "tracking_note": "Sam3TrackerVideoProcessor API currently uses point prompts here; selected mask/bbox are used to choose the point.",
    }
