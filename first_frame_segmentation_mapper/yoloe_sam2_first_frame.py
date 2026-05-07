#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


DEFAULT_VIDEO = Path("/Users/markzhao/Desktop/ir_annotation_pipeline/episode_000000.mp4")
DEFAULT_OUTPUT_DIR = Path("/Users/markzhao/Desktop/ir_annotation_pipeline/yoloe_sam2_first_frame")
DEFAULT_DETECTIONS_JSON = DEFAULT_OUTPUT_DIR / "first_frame_detections.json"


def _require(module_name: str, import_stmt: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Missing dependency '{module_name}'. Install it with: {import_stmt}"
        ) from exc


def _load_common_modules():
    cv2 = _require("cv2", "pip install opencv-python")
    np = _require("numpy", "pip install numpy")
    torch = _require("torch", "pip install torch torchvision")
    return cv2, np, torch


def _load_yoloe():
    try:
        from ultralytics import YOLOE
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'ultralytics'. Install it with: pip install ultralytics"
        ) from exc
    return YOLOE


def _validate_yoloe_text_stack():
    try:
        import clip
    except ImportError as exc:
        raise RuntimeError(
            "YOLOE text prompts require the CLIP package expected by Ultralytics. "
            "Install OpenAI CLIP in your environment, for example: "
            "python -m pip install git+https://github.com/openai/CLIP.git"
        ) from exc

    clip_tokenizer = getattr(getattr(clip, "clip", None), "tokenize", None)
    if clip_tokenizer is None:
        clip_file = getattr(clip, "__file__", "<unknown>")
        raise RuntimeError(
            "YOLOE found an incompatible 'clip' package at "
            f"{clip_file}. Ultralytics expects a module that provides "
            "'clip.clip.tokenize'. This usually means the wrong 'clip' package "
            "is installed. Remove that package and install OpenAI CLIP instead, "
            "for example:\n"
            "  python -m pip uninstall clip\n"
            "  python -m pip install git+https://github.com/openai/CLIP.git"
        )


def _load_sam2():
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'sam2'. Install the official repo/package from "
            "https://github.com/facebookresearch/sam2"
        ) from exc
    return build_sam2, SAM2ImagePredictor


def _extract_first_frame(cv2, video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to read the first frame from: {video_path}")
    return frame_bgr


def _auto_device(torch):
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _run_yoloe(YOLOE, frame_bgr, prompts: list[str], model_path: str, conf: float, device: str):
    model = YOLOE(model_path)
    model.set_classes(prompts, model.get_text_pe(prompts))
    results = model.predict(frame_bgr, conf=conf, device=device, verbose=False)
    if not results:
        return []

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.cpu().tolist()
    scores = result.boxes.conf.cpu().tolist()
    class_ids = [int(v) for v in result.boxes.cls.cpu().tolist()]

    detections = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        detections.append(
            {
                "label": prompts[class_id],
                "score": float(score),
                "xyxy": [float(v) for v in box],
            }
        )
    return detections


def _build_sam2_predictor(build_sam2, SAM2ImagePredictor, config_path: str, checkpoint_path: str, device: str):
    model = build_sam2(_normalize_sam2_config(config_path), checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(model)
    return predictor


def _normalize_sam2_config(config_path: str) -> str:
    normalized = config_path.replace("\\", "/")
    marker = "/configs/"
    if normalized.startswith("configs/"):
        return normalized
    if marker in normalized:
        return "configs/" + normalized.split(marker, 1)[1]
    return normalized


def _null_context():
    class _NullContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    return _NullContext()


def _predict_mask_for_box(predictor, np, torch, image_rgb, box_xyxy, device: str):
    predictor.set_image(image_rgb)
    box = np.array(box_xyxy, dtype=np.float32)[None, :]

    autocast_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else _null_context()
    )

    with torch.inference_mode(), autocast_context:
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=False,
        )

    mask = np.asarray(masks[0]).squeeze().astype(bool)
    score = float(scores[0]) if len(scores) else None
    return mask, score


def _save_mask_png(cv2, mask_bool, output_path: Path):
    mask_u8 = (mask_bool.astype("uint8")) * 255
    cv2.imwrite(str(output_path), mask_u8)


def _draw_overlay(cv2, np, frame_bgr, object_detections, gripper_detection, mask_records):
    overlay = frame_bgr.copy()

    for record in mask_records:
        mask = np.asarray(record["mask"]).squeeze().astype(bool)
        if mask.ndim != 2:
            raise ValueError(f"Expected a 2D mask for overlay, got shape {mask.shape}")
        color = np.array(record["color"], dtype=np.uint8)
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype(np.uint8)

    for det in object_detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        score = det.get("score")
        label = det["label"] if score is None else f"{det['label']} {score:.2f}"
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (60, 200, 60), 2)
        cv2.putText(
            overlay,
            label,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (60, 200, 60),
            2,
        )

    if gripper_detection:
        x1, y1, x2, y2 = [int(round(v)) for v in gripper_detection["xyxy"]]
        score = gripper_detection.get("score")
        label = (
            gripper_detection["label"]
            if score is None
            else f"{gripper_detection['label']} {score:.2f}"
        )
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (40, 140, 255), 2)
        cv2.putText(
            overlay,
            label,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 140, 255),
            2,
        )

    return overlay


def _load_detections_json(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    object_detections = []
    for det in payload.get("object_detections", []):
        object_detections.append(
            {
                "label": det["label"],
                "score": det.get("box_score", det.get("score")),
                "xyxy": det.get("box_xyxy", det.get("xyxy")),
            }
        )
    gripper_detection = payload.get("gripper_detection")
    return object_detections, gripper_detection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use YOLOE and/or SAM2 to generate boxes and object masks for the first frame of a video."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help=f"Input video path. Default: {DEFAULT_VIDEO}")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument(
        "--module",
        choices=["yoloe", "sam2", "combined"],
        default="combined",
        help="Which module to run. 'yoloe' creates detections only, 'sam2' creates masks from an existing detections JSON, 'combined' runs both.",
    )
    parser.add_argument("--yoloe-model", default="yoloe-11s-seg.pt", help="YOLOE checkpoint or model name.")
    parser.add_argument("--sam2-config", help="SAM2 config path, e.g. configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--sam2-checkpoint", help="SAM2 checkpoint path, e.g. checkpoints/sam2.1_hiera_small.pt")
    parser.add_argument("--object-prompts", nargs="+", help="Object text prompts for YOLOE, e.g. blanket pillow")
    parser.add_argument("--gripper-prompt", default="robot gripper", help="Text prompt used to detect the gripper.")
    parser.add_argument("--conf", type=float, default=0.01, help="YOLOE confidence threshold.")
    parser.add_argument(
        "--detections-json",
        type=Path,
        default=DEFAULT_DETECTIONS_JSON,
        help=f"Existing detections JSON for --module sam2. Default: {DEFAULT_DETECTIONS_JSON}",
    )
    args = parser.parse_args()

    if args.module in {"yoloe", "combined"} and not args.object_prompts:
        parser.error("--object-prompts is required when --module is 'yoloe' or 'combined'")
    if args.module in {"sam2", "combined"} and (not args.sam2_config or not args.sam2_checkpoint):
        parser.error("--sam2-config and --sam2-checkpoint are required when --module is 'sam2' or 'combined'")
    if args.module == "sam2" and not args.detections_json.exists():
        parser.error(f"--detections-json not found: {args.detections_json}")

    cv2, np, torch = _load_common_modules()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = args.output_dir / "masks"
    if args.module in {"sam2", "combined"}:
        masks_dir.mkdir(parents=True, exist_ok=True)

    frame_bgr = _extract_first_frame(cv2, args.video)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    device = _auto_device(torch)

    object_detections = []
    gripper_detection = None

    if args.module in {"yoloe", "combined"}:
        YOLOE = _load_yoloe()
        _validate_yoloe_text_stack()
        all_prompts = list(args.object_prompts) + [args.gripper_prompt]
        detections = _run_yoloe(
            YOLOE=YOLOE,
            frame_bgr=frame_bgr,
            prompts=all_prompts,
            model_path=args.yoloe_model,
            conf=args.conf,
            device=device,
        )
        object_detections = [d for d in detections if d["label"] != args.gripper_prompt]
        gripper_candidates = [d for d in detections if d["label"] == args.gripper_prompt]
        gripper_detection = max(gripper_candidates, key=lambda d: d["score"]) if gripper_candidates else None
    else:
        object_detections, gripper_detection = _load_detections_json(args.detections_json)

    mask_records = []
    object_outputs = []
    palette = [
        [255, 90, 90],
        [90, 220, 120],
        [90, 180, 255],
        [255, 210, 90],
        [190, 120, 255],
    ]

    if args.module in {"sam2", "combined"}:
        build_sam2, SAM2ImagePredictor = _load_sam2()
        predictor = _build_sam2_predictor(
            build_sam2=build_sam2,
            SAM2ImagePredictor=SAM2ImagePredictor,
            config_path=args.sam2_config,
            checkpoint_path=args.sam2_checkpoint,
            device=device,
        )

        for index, det in enumerate(object_detections):
            mask_bool, mask_score = _predict_mask_for_box(
                predictor=predictor,
                np=np,
                torch=torch,
                image_rgb=frame_rgb,
                box_xyxy=det["xyxy"],
                device=device,
            )
            mask_path = masks_dir / f"object_{index:02d}_{det['label'].replace(' ', '_')}.png"
            _save_mask_png(cv2, mask_bool, mask_path)

            color = palette[index % len(palette)]
            mask_records.append({"mask": mask_bool, "color": color})
            object_outputs.append(
                {
                    "label": det["label"],
                    "box_xyxy": det["xyxy"],
                    "box_score": det.get("score"),
                    "mask_score": mask_score,
                    "mask_path": str(mask_path),
                }
            )
    else:
        object_outputs = [
            {
                "label": det["label"],
                "box_xyxy": det["xyxy"],
                "box_score": det.get("score"),
                "mask_score": None,
                "mask_path": None,
            }
            for det in object_detections
        ]

    first_frame_path = args.output_dir / "first_frame.jpg"
    overlay_path = args.output_dir / "first_frame_overlay.jpg"
    json_path = args.output_dir / "first_frame_detections.json"

    cv2.imwrite(str(first_frame_path), frame_bgr)
    overlay = _draw_overlay(cv2, np, frame_bgr, object_detections, gripper_detection, mask_records)
    cv2.imwrite(str(overlay_path), overlay)

    payload = {
        "video_path": str(args.video),
        "frame_index": 0,
        "module": args.module,
        "device": device,
        "first_frame_path": str(first_frame_path),
        "overlay_path": str(overlay_path),
        "object_prompts": args.object_prompts,
        "gripper_prompt": args.gripper_prompt,
        "object_detections": object_outputs,
        "gripper_detection": gripper_detection,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote first-frame assets to {args.output_dir}")
    print(f"Module: {args.module}")
    print(f"JSON summary: {json_path}")
    print(f"Overlay image: {overlay_path}")


if __name__ == "__main__":
    main()
