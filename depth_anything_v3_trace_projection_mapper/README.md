# depth_anything_v3_trace_projection_mapper

Depth Anything V3 camera estimation and 3D-to-2D trace projection pipeline.

## Scripts

- `run_da3_camera_pipeline.py`: runs Depth Anything V3 on videos and saves `intrinsics`/`extrinsics` per episode.
- `project_trace_and_visualize.py`: projects parquet 3D future gripper traces to 2D and overlays predicted traces with `annotation.trace`.
- `run_da3_cameras.sh`: wrapper for DA3 camera estimation on `../test_data`.
- `run_trace_projection.sh`: wrapper for trace projection and visualization.
- `run_full_pipeline.sh`: runs both stages.

## Usage

```bash
depth_anything_v3_trace_projection_mapper/run_da3_cameras.sh
depth_anything_v3_trace_projection_mapper/run_trace_projection.sh
```

The overlay convention is:

- cyan: projected/predicted 2D trace
- magenta: ground-truth `annotation.trace`

If no robot-to-DA3-world calibration is provided, `project_trace_and_visualize.py`
still computes the raw DA3 projection, but `--projection-mode auto` falls back to a
weak-perspective fit from available GT traces when most raw projected points are
outside the frame. For physical DA3 projection, provide:

```bash
depth_anything_v3_trace_projection_mapper/run_trace_projection.sh \
  --robot-to-da3-json path/to/robot_to_da3.json \
  --projection-mode da3
```
