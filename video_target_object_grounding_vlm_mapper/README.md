# video_target_object_grounding_vlm_mapper

Standalone first-frame target-object grounding, rewritten out of the
Data-Juicer operator path.

## Inputs

- `test_data/episode_*.mp4`
- matching `test_data/episode_*.parquet`
- task instruction from `other_information.language_instruction_2`

The standalone script does not use `annotation.*` or `Q_annotation.*` fields.

## What It Does

For each episode:

1. Reads the parquet task instruction.
2. Extracts the first video frame.
3. Sends the first frame plus instruction to a local vLLM OpenAI-compatible VLM.
4. Parses JSON with `target_object`, `bbox`, `center`, and `confidence`.
5. Writes one JSON result per episode plus a summary CSV/JSON.
6. Saves a first-frame overlay with the predicted bbox and center.

## Usage

Smoke test without VLM:

```bash
video_target_object_grounding_vlm_mapper/run_generate.sh \
  --pattern episode_000000.parquet
```

VLM inference:

```bash
video_target_object_grounding_vlm_mapper/run_generate.sh \
  --use-vlm \
  --vlm-base-url http://localhost:8000/v1 \
  --vlm-model auto \
  --pattern episode_000000.parquet
```

Outputs are written to:

```text
video_target_object_grounding_vlm_mapper/outputs/
```

Each episode gets:

- `first_frame.jpg`
- `target_object_grounding_overlay.jpg`
- `target_object_grounding.json`

The folder also gets:

- `target_object_grounding_summary.json`
- `target_object_grounding_summary.csv`
