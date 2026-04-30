# video_subtask_instruction_mapper

Generate subtask-level instructions and visual artifacts from only:

- episode `.mp4` video
- parquet `other_information.language_instruction_2`
- non-annotation robot pose fields for subgoal/goal poses

This mapper must not use `annotation.*` or `Q_annotation.*` fields.

## Approach

The intended path is open-source VLM planning with Qwen3.5-35B-A3B served by
vLLM:

1. Uniformly sample frames from the full video.
2. Provide the sampled frames plus `other_information.language_instruction_2`.
3. Ask the VLM to return JSON subtask segments with actual frame indices and
   concise subtask instructions.
4. Use each segment end frame as the subgoal image frame.
5. Use the final episode frame as the goal image frame.
6. Read gripper/TCP pose from non-annotation parquet fields.

If `--use-vlm` is not passed, the script creates a single fallback segment for
the entire episode using the task instruction. This is only a smoke-test mode;
multi-step segmentation requires the VLM.

## Outputs

For each episode:

- `subtask_segments`: frame range and subtask instruction for each segment
- `subgoal_image_frame`: the final frame of each segment
- `subgoal_image_path`: exported subgoal image
- `subgoal_image_gripper_pose`: gripper pose at the subgoal image frame
- `goal_image_frame`: final frame of the episode
- `goal_image_path`: exported goal image
- `goal_image_gripper_pose`: gripper pose at the goal image frame

## Usage

Fallback/smoke test:

```bash
video_subtask_instruction_mapper/run_generate.sh
video_subtask_instruction_mapper/run_visualize.sh
```

Serve Qwen3.5-35B-A3B locally with vLLM on 8 GPUs:

```bash
video_subtask_instruction_mapper/serve_qwen35_vllm.sh
```

Equivalent command:

```bash
vllm serve Qwen/Qwen3.5-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --trust-remote-code
```

VLM segmentation and instruction generation through the local OpenAI-compatible
endpoint:

```bash
video_subtask_instruction_mapper/run_generate.sh \
  --use-vlm \
  --vlm-backend openai \
  --vlm-base-url http://localhost:8000/v1 \
  --vlm-model Qwen/Qwen3.5-35B-A3B \
  --vlm-frame-samples 12
```

Video reading/writing uses `imageio`, so install:

```bash
python -m pip install "imageio[ffmpeg]"
```

Client-side dependencies:

```bash
python -m pip install "imageio[ffmpeg]"
```

The vLLM server environment needs `vllm` with Qwen3.5 support and the model
weights available from Hugging Face.
