#!/usr/bin/env python3
"""Convert LeRobot episode parquet files into Data-Juicer-style dataset.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INSTRUCTION_KEYS = (
    "language_instruction",
    "annotation.instruction_add",
)


def read_first_row(parquet_path: Path) -> dict[str, Any]:
    """Read one row while allowing whichever parquet backend is installed."""
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path)
        if table.num_rows == 0:
            raise ValueError(f"{parquet_path} is empty")
        return table.slice(0, 1).to_pylist()[0]
    except ImportError:
        pass

    try:
        import pandas as pd

        frame = pd.read_parquet(parquet_path)
        if frame.empty:
            raise ValueError(f"{parquet_path} is empty")
        return frame.iloc[0].to_dict()
    except ImportError:
        pass

    try:
        import polars as pl

        frame = pl.read_parquet(parquet_path, n_rows=1)
        if frame.height == 0:
            raise ValueError(f"{parquet_path} is empty")
        return frame.to_dicts()[0]
    except ImportError:
        pass

    raise RuntimeError("No parquet reader found. Install pyarrow, pandas, or polars.")


def discover_parquets(input_root: Path, pattern: str) -> list[Path]:
    if input_root.is_file():
        return [input_root]

    data_root = input_root / "data"
    search_root = data_root if data_root.exists() else input_root
    return sorted(search_root.glob(pattern))


def format_episode_id(parquet_path: Path, row: dict[str, Any], template: str | None) -> str:
    episode_index = row.get("episode_index")
    values = {
        "parquet_stem": parquet_path.stem,
        "episode_stem": parquet_path.stem,
        "episode_index": episode_index,
        "episode_index_06d": f"{int(episode_index):06d}" if episode_index is not None else "",
    }
    if template:
        return template.format(**values)
    return parquet_path.stem


def choose_instruction(row: dict[str, Any], instruction_key: str | None) -> str:
    candidates = (instruction_key,) if instruction_key else DEFAULT_INSTRUCTION_KEYS
    for key in candidates:
        if not key:
            continue
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    keys = ", ".join(DEFAULT_INSTRUCTION_KEYS)
    raise KeyError(f"Could not find a non-empty instruction. Tried: {keys}")


def format_video_from_template(
    video_template: str,
    parquet_path: Path,
    row: dict[str, Any],
    episode_id: str,
) -> Path:
    episode_index = row.get("episode_index")
    values = {
        "episode_id": episode_id,
        "parquet_stem": parquet_path.stem,
        "episode_stem": parquet_path.stem,
        "episode_index": episode_index,
        "episode_index_06d": f"{int(episode_index):06d}" if episode_index is not None else "",
        "chunk": parquet_path.parent.name,
        "camera_view": row.get("camera_view", ""),
    }
    return Path(video_template.format(**values))


def find_video_path(
    parquet_path: Path,
    input_root: Path,
    row: dict[str, Any],
    episode_id: str,
    video_root: Path | None,
    video_template: str | None,
) -> Path:
    if video_template:
        return format_video_from_template(video_template, parquet_path, row, episode_id)

    if input_root.is_dir():
        videos_root = video_root if video_root else input_root / "videos"
        default_lerobot_video = (
            videos_root
            / parquet_path.parent.name
            / "observation.images.primary"
            / f"{parquet_path.stem}.mp4"
        )
        return default_lerobot_video

    candidate_roots = []
    if video_root:
        candidate_roots.append(video_root)

    same_dir_video = parquet_path.with_suffix(".mp4")
    if same_dir_video.exists():
        return same_dir_video

    for root in candidate_roots:
        if root.exists():
            matches = sorted(root.rglob(f"{parquet_path.stem}.mp4"))
            if matches:
                return matches[0]

    if input_root.is_dir() and "data" in parquet_path.parts:
        parts = list(parquet_path.parts)
        parts[parts.index("data")] = "videos"
        return Path(*parts).with_suffix(".mp4")

    return parquet_path.with_suffix(".mp4")


def render_path(path: Path, base_dir: Path, absolute: bool) -> str:
    if absolute:
        return str(path.expanduser().resolve())

    try:
        rel = path.expanduser().resolve().relative_to(base_dir.expanduser().resolve())
    except ValueError:
        rel = path

    text = rel.as_posix()
    return text if text.startswith(".") or text.startswith("/") else f"./{text}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert lerobot_droid_anno/data/chunk-*/episode_*.parquet files "
            "to dataset.jsonl records with videos/text/language_instruction/episode_id."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("lerobot_droid_anno"),
        help="Dataset root or a single parquet file. Default: lerobot_droid_anno",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset.jsonl"),
        help="Output JSONL path. Default: dataset.jsonl",
    )
    parser.add_argument(
        "-n",
        "--max-episodes",
        type=int,
        default=None,
        help="Convert only the first N parquet files after sorting.",
    )
    parser.add_argument(
        "--pattern",
        default="chunk-*/episode_*.parquet",
        help="Glob under input-root/data, or input-root if no data dir exists.",
    )
    parser.add_argument(
        "--instruction-key",
        default=None,
        help="Exact parquet column to use for language_instruction.",
    )
    parser.add_argument(
        "--episode-id-template",
        default=None,
        help="Optional format string, e.g. droid_example_{episode_index_06d}.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help=(
            "Videos root. Default: input-root/videos. The default layout is "
            "VIDEO_ROOT/chunk-000/observation.images.primary/episode_*.mp4."
        ),
    )
    parser.add_argument(
        "--video-template",
        default=None,
        help=(
            "Optional video path format string. Available fields include "
            "{episode_id}, {episode_stem}, {episode_index_06d}, {chunk}, {camera_view}."
        ),
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory for relative paths in the videos field. Default: current directory.",
    )
    parser.add_argument(
        "--absolute-video-paths",
        action="store_true",
        help="Write absolute video paths instead of ./relative paths.",
    )
    parser.add_argument(
        "--allow-missing-videos",
        action="store_true",
        help="Do not fail if the resolved video file does not exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquets = discover_parquets(args.input_root, args.pattern)
    if not parquets:
        raise FileNotFoundError(f"No parquet files found under {args.input_root}")

    if args.max_episodes is not None:
        if args.max_episodes < 0:
            raise ValueError("--max-episodes must be non-negative")
        parquets = parquets[: args.max_episodes]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    missing_videos = []

    with args.output.open("w", encoding="utf-8") as out_file:
        for parquet_path in parquets:
            row = read_first_row(parquet_path)
            instruction = choose_instruction(row, args.instruction_key)
            episode_id = format_episode_id(parquet_path, row, args.episode_id_template)
            video_path = find_video_path(
                parquet_path=parquet_path,
                input_root=args.input_root,
                row=row,
                episode_id=episode_id,
                video_root=args.video_root,
                video_template=args.video_template,
            )

            if not video_path.exists():
                missing_videos.append(video_path)
                if not args.allow_missing_videos:
                    raise FileNotFoundError(
                        f"Video for {parquet_path} was resolved to {video_path}, "
                        "but that file does not exist. Use --video-root, "
                        "--video-template, or --allow-missing-videos."
                    )

            video_text = render_path(video_path, args.base_dir, args.absolute_video_paths)
            sample = {
                "videos": [video_text],
                "text": f"<__dj__video> {instruction}",
                "language_instruction": instruction,
                "episode_id": episode_id,
            }
            out_file.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1

    print(f"Wrote {written} records to {args.output}")
    if missing_videos:
        print(f"Warning: {len(missing_videos)} video paths did not exist.")


if __name__ == "__main__":
    main()
