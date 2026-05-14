from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from video_object_detection_mapper import common


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


def load_video_predictor(args, torch, device: str):
    try:
        from sam3.model_builder import build_sam3_video_predictor
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'sam3'. Install it with: python -m pip install git+https://github.com/facebookresearch/sam3.git"
        ) from exc
    model_dir = common.snapshot_download_model(args.sam3_video_model, args.sam3_video_cache_dir)
    checkpoint_path = args.sam3_video_checkpoint or find_checkpoint(model_dir)
    builder = build_sam3_video_predictor
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
        kwargs["version"] = args.sam3_video_version
    if "gpus_to_use" in signature.parameters and device.startswith("cuda"):
        kwargs["gpus_to_use"] = range(torch.cuda.device_count())
    if "compile" in signature.parameters:
        kwargs["compile"] = args.sam3_video_compile
    if "warm_up" in signature.parameters:
        kwargs["warm_up"] = args.sam3_video_warm_up
    if "async_loading_frames" in signature.parameters:
        kwargs["async_loading_frames"] = args.sam3_video_async_loading_frames
    return builder(**kwargs), {"model_dir": str(model_dir), "checkpoint_path": str(checkpoint_path) if checkpoint_path else None, "builder_kwargs": kwargs}


def response_outputs(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        if "outputs" in response and isinstance(response["outputs"], dict):
            return response["outputs"]
        return response
    if hasattr(response, "outputs") and isinstance(response.outputs, dict):
        return response.outputs
    return {}


def mask_from_outputs(np, torch, outputs: dict[str, Any], obj_id: int):
    masks = common.tensor_to_numpy(np, torch, next((outputs[name] for name in ("out_binary_masks", "pred_masks", "masks") if name in outputs), None))
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
    obj_ids = next((outputs[name] for name in ("out_obj_ids", "obj_ids", "object_ids") if name in outputs), None)
    if obj_ids is not None:
        obj_ids = [int(item) for item in np.asarray(common.tensor_to_numpy(np, torch, obj_ids)).reshape(-1).tolist()]
        if obj_id in obj_ids:
            return masks[obj_ids.index(obj_id)]
    return masks[0]


def normalized_point_from_bbox(bbox: list[float], frame_width: int, frame_height: int) -> list[float]:
    cx = (float(bbox[0]) + float(bbox[2])) / 2.0
    cy = (float(bbox[1]) + float(bbox[3])) / 2.0
    return [cx / max(1.0, float(frame_width)), cy / max(1.0, float(frame_height))]


def prompt_variants(torch, bbox: list[float], mask: Any | None, point_xy: list[float], frame_width: int, frame_height: int) -> list[tuple[str, dict[str, Any]]]:
    variants: list[tuple[str, dict[str, Any]]] = []
    if mask is not None:
        variants.append(("mask", {"mask": torch.as_tensor(mask).bool()}))
    if bbox is not None:
        norm_box = [
            float(bbox[0]) / max(1.0, float(frame_width)),
            float(bbox[1]) / max(1.0, float(frame_height)),
            float(bbox[2]) / max(1.0, float(frame_width)),
            float(bbox[3]) / max(1.0, float(frame_height)),
        ]
        variants.append(("bbox", {"box": torch.tensor(norm_box, dtype=torch.float32)}))
        norm_point = normalized_point_from_bbox(bbox, frame_width, frame_height)
    else:
        norm_point = [float(point_xy[0]) / max(1.0, float(frame_width)), float(point_xy[1]) / max(1.0, float(frame_height))]
    variants.append(
        (
            "point",
            {
                "points": torch.tensor([norm_point], dtype=torch.float32),
                "point_labels": torch.tensor([1], dtype=torch.int32),
            },
        )
    )
    return variants


def track_video(
    np,
    torch,
    predictor,
    video_path: Path,
    bbox: list[float],
    mask: Any | None,
    point_xy: list[float],
    frame_width: int,
    frame_height: int,
    obj_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not hasattr(predictor, "handle_request"):
        raise RuntimeError("SAM3 predictor does not expose handle_request; cannot run video tracking.")

    last_error = None
    for prompt_type, prompt_payload in prompt_variants(torch, bbox, mask, point_xy, frame_width, frame_height):
        session_response = predictor.handle_request(request=dict(type="start_session", resource_path=str(video_path)))
        session_id = session_response["session_id"] if isinstance(session_response, dict) else session_response.session_id
        try:
            add_response = predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    obj_id=int(obj_id),
                    **prompt_payload,
                )
            )
            add_outputs = response_outputs(add_response)
            frame_records = []
            add_mask = mask_from_outputs(np, torch, add_outputs, obj_id)
            if add_mask is not None:
                frame_records.append({"frame_index": 0, "mask": add_mask, "bbox_xyxy": common.mask_to_bbox(np, add_mask)})

            stream = predictor.handle_stream_request(request=dict(type="propagate_in_video", session_id=session_id))
            seen = {0} if add_mask is not None else set()
            for fallback_frame_idx, response in enumerate(stream):
                outputs = response_outputs(response)
                frame_idx = int(outputs.get("frame_index") or outputs.get("frame_idx") or fallback_frame_idx)
                if frame_idx in seen:
                    continue
                out_mask = mask_from_outputs(np, torch, outputs, obj_id)
                if out_mask is None:
                    continue
                frame_records.append({"frame_index": frame_idx, "mask": out_mask, "bbox_xyxy": common.mask_to_bbox(np, out_mask)})
                seen.add(frame_idx)
            return sorted(frame_records, key=lambda item: item["frame_index"]), {"tracking_prompt_type": prompt_type}
        except Exception as exc:
            last_error = exc
        finally:
            try:
                predictor.handle_request(request=dict(type="close_session", session_id=session_id))
            except Exception:
                pass
    raise RuntimeError(f"SAM3 tracking failed for mask, bbox, and point prompts. Last error: {last_error}")
