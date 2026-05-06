#!/usr/bin/env python3
"""Split an episode video into clips for each detected gripper phase range."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_PHASE_JSON = SCRIPT_DIR / "episode_000000_gripper_phases.json"
DEFAULT_VIDEO = PROJECT_ROOT / "episode_000000.mp4"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "episode_000000_phase_clips"


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("This script needs ffmpeg and ffprobe on PATH.")


def probe_fps(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    text = result.stdout.strip()
    if "/" in text:
        num, den = text.split("/", 1)
        return float(num) / float(den)
    return float(text)


def phase_display_name(phase: str) -> str:
    aliases = {
        "pre_grasp": "pre-grasp",
        "immobilization": "approach",
        "contact": "contact",
        "detach": "detach",
        "post_grasp": "post-grasp",
        "transition": "transition",
    }
    return aliases.get(phase, phase.replace("_", "-"))


def run_ffmpeg_clip(
    video_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    reencode: bool,
) -> None:
    start_time = start_frame / fps
    duration = (end_frame - start_frame + 1) / fps
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-ss",
        f"{start_time:.6f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.6f}",
    ]
    if reencode:
        cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-an"])
    else:
        cmd.extend(["-c", "copy", "-an"])
    cmd.append(str(output_path))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split an input video into one clip per gripper phase range."
    )
    parser.add_argument("--phase-json", type=Path, default=DEFAULT_PHASE_JSON)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Use ffmpeg stream copy. Faster, but cuts may snap to keyframes.",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Do not create placeholder JSON entries for phases without ranges.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_ffmpeg()
    phase_payload = json.loads(args.phase_json.read_text(encoding="utf-8"))
    phase_ranges = phase_payload["phase_ranges"]
    fps = probe_fps(args.video)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "video": str(args.video),
        "phase_json": str(args.phase_json),
        "fps": fps,
        "clips": [],
    }

    clip_count = 0
    for phase, ranges in phase_ranges.items():
        if not ranges and args.skip_empty:
            continue
        for range_idx, (start_frame, end_frame) in enumerate(ranges):
            display = phase_display_name(phase)
            output_name = (
                f"{clip_count:02d}_{display}_"
                f"frames_{int(start_frame):06d}_{int(end_frame):06d}.mp4"
            )
            output_path = args.output_dir / output_name
            run_ffmpeg_clip(
                video_path=args.video,
                output_path=output_path,
                start_frame=int(start_frame),
                end_frame=int(end_frame),
                fps=fps,
                reencode=not args.copy,
            )
            manifest["clips"].append(
                {
                    "phase": display,
                    "robo2vlm_phase": phase,
                    "range_index": range_idx,
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "duration_seconds": (int(end_frame) - int(start_frame) + 1) / fps,
                    "path": str(output_path),
                }
            )
            clip_count += 1

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {clip_count} phase clips to {args.output_dir}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
