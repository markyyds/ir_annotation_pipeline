#!/usr/bin/env python3
"""Create visual artifacts for generated subtask instruction outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_TEST_DATA = PROJECT_ROOT / "test_data"
DEFAULT_INPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "visualizations"


def open_video_reader(video_path: Path):
    try:
        import imageio.v2 as imageio

        return imageio.get_reader(str(video_path))
    except ValueError as exc:
        raise RuntimeError(
            "Could not open video with imageio. Install video support with: "
            "python -m pip install 'imageio[ffmpeg]'"
        ) from exc


def open_video_writer(path: Path, fps: float):
    try:
        import imageio.v2 as imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        return imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8)
    except Exception as exc:
        raise RuntimeError(
            "Could not create video writer. Install video support with: "
            "python -m pip install 'imageio[ffmpeg]'"
        ) from exc


def video_fps(reader) -> float:
    metadata = reader.get_meta_data()
    fps = metadata.get("fps")
    return float(fps) if fps else 10.0


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip().lower())
    return text.strip("_")[:80] or "subtask"


def draw_label(image: Image.Image, lines: list[str]) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width = image.width
    height = 18 + 17 * len(lines)
    draw.rectangle([0, 0, width, height], fill=(0, 0, 0, 160))
    y = 8
    for line in lines:
        draw.text((10, y), line, fill=(255, 255, 255, 255))
        y += 17
    return image


def write_segment_video(reader, segment: dict[str, Any], output_path: Path, fps: float) -> None:
    writer = open_video_writer(output_path, fps=fps)
    try:
        for frame_idx in range(int(segment["start_frame"]), int(segment["end_frame"]) + 1):
            frame = Image.fromarray(reader.get_data(frame_idx)).convert("RGB")
            frame = draw_label(
                frame,
                [
                    f"{segment['segment_id']} | frames {segment['start_frame']}-{segment['end_frame']}",
                    segment["subtask_instruction"],
                ],
            )
            writer.append_data(frame)
    finally:
        writer.close()


def copy_labeled_image(source_path: Path, output_path: Path, lines: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(source_path).convert("RGB")
    draw_label(image, lines).save(output_path, quality=95)


def make_contact_sheet(images: list[tuple[Path, list[str]]], output_path: Path, thumb_width: int = 320) -> None:
    if not images:
        return
    thumbs = []
    for path, lines in images:
        image = Image.open(path).convert("RGB")
        scale = thumb_width / image.width
        thumb = image.resize((thumb_width, max(1, int(image.height * scale))))
        thumb = draw_label(thumb, lines)
        thumbs.append(thumb)

    gap = 12
    sheet_width = thumb_width
    sheet_height = sum(thumb.height for thumb in thumbs) + gap * (len(thumbs) - 1)
    sheet = Image.new("RGB", (sheet_width, sheet_height), (245, 245, 245))
    y = 0
    for thumb in thumbs:
        sheet.paste(thumb, (0, y))
        y += thumb.height + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)


def visualize_episode(input_path: Path, test_data: Path, output_dir: Path) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    episode_id = payload["episode_id"]
    video_path = test_data / f"{episode_id}.mp4"
    episode_dir = output_dir / episode_id
    subtask_video_dir = episode_dir / "subtask_videos"
    subgoal_dir = episode_dir / "subgoal_images"
    goal_dir = episode_dir / "goal"

    reader = open_video_reader(video_path)
    fps = video_fps(reader)
    artifacts = {"episode_id": episode_id, "subtask_videos": [], "subgoal_images": [], "goal_image": None}
    contact_sheet_items = []
    try:
        for segment in payload["subtask_segments"]:
            video_out = (
                subtask_video_dir
                / f"{segment['segment_id']}_{safe_name(segment['subtask_instruction'])}_"
                f"{segment['start_frame']:06d}_{segment['end_frame']:06d}.mp4"
            )
            write_segment_video(reader, segment, video_out, fps)
            artifacts["subtask_videos"].append(str(video_out))

            subgoal_source = Path(segment["subgoal_image_path"])
            subgoal_out = (
                subgoal_dir
                / f"{segment['segment_id']}_subgoal_frame_{segment['subgoal_image_frame']:06d}.jpg"
            )
            lines = [
                f"{segment['segment_id']} subgoal | frame {segment['subgoal_image_frame']}",
                segment["subtask_instruction"],
            ]
            copy_labeled_image(subgoal_source, subgoal_out, lines)
            artifacts["subgoal_images"].append(str(subgoal_out))
            contact_sheet_items.append((subgoal_out, lines))

        goal_source = Path(payload["goal_image_path"])
        goal_out = goal_dir / f"goal_frame_{payload['goal_image_frame']:06d}.jpg"
        goal_lines = [f"goal | frame {payload['goal_image_frame']}", payload.get("task_instruction") or ""]
        copy_labeled_image(goal_source, goal_out, goal_lines)
        artifacts["goal_image"] = str(goal_out)
        contact_sheet_items.append((goal_out, goal_lines))
    finally:
        reader.close()

    contact_sheet = episode_dir / "overview_contact_sheet.jpg"
    make_contact_sheet(contact_sheet_items, contact_sheet)
    artifacts["overview_contact_sheet"] = str(contact_sheet)
    (episode_dir / "visualization_manifest.json").write_text(
        json.dumps(artifacts, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize generated subtask instruction outputs.")
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern", default="episode_*/subtask_instruction_outputs.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = sorted(args.input_dir.glob(args.pattern))
    if not input_paths:
        raise FileNotFoundError(f"No subtask outputs matched {args.input_dir / args.pattern}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    for input_path in input_paths:
        print(f"[{input_path.parent.name}] visualizing subtasks")
        manifests.append(visualize_episode(input_path, args.test_data, args.output_dir))
    summary_path = args.output_dir / "visualization_summary.json"
    summary_path.write_text(json.dumps(manifests, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote visualization summary to {summary_path}")


if __name__ == "__main__":
    main()
