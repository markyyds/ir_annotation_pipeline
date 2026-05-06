# video_phase_segmentation_mapper

Robo2VLM-style gripper phase segmentation utilities.

## Files

- `generate_gripper_phase_annotations.py`: generates per-frame phase labels, phase ranges, first contact frame, and contact gripper pose.
- `split_video_by_gripper_phase.py`: splits the episode video into one clip per phase range.
- `evaluate_phase_segmentation.py`: evaluates first-contact prediction and predicted contact TCP pose against annotated test parquet files.
- `run_eval.sh`: one-command evaluation wrapper for `../test_data`.
- `episode_000000_gripper_phases.json`: generated phase annotations for the sample episode.
- `episode_000000_phase_clips/`: generated phase clips for visual inspection.
- `eval_outputs/`: generated evaluation reports.

## Usage

```bash
python video_phase_segmentation_mapper/generate_gripper_phase_annotations.py
python video_phase_segmentation_mapper/split_video_by_gripper_phase.py
video_phase_segmentation_mapper/run_eval.sh
```
