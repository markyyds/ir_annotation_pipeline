python3 yoloe_sam2_first_frame.py \
  --module combined \
  --object-prompts fleece \
  --gripper-prompt "robot gripper" \
  --yoloe-model yoloe-11l-seg.pt \
  --sam2-config /Users/markzhao/Desktop/ir_annotation_pipeline/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2-checkpoint /Users/markzhao/Desktop/ir_annotation_pipeline/sam2/checkpoints/sam2.1_hiera_large.pt

# YOLOE only:
# python3 yoloe_sam2_first_frame.py \
#   --module yoloe \
#   --object-prompts fleece\
#   --gripper-prompt "robot gripper" \
#   --yoloe-model yoloe-11l-seg.pt

# SAM2 only, reusing a detections JSON from a prior YOLOE/combined run:
# python3 yoloe_sam2_first_frame.py \
#   --module sam2 \
#   --sam2-config /Users/markzhao/Desktop/ir_annotation_pipeline/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
#   --sam2-checkpoint /absolute/path/to/sam2/checkpoints/sam2.1_hiera_large.pt \
#   --detections-json /Users/markzhao/Desktop/ir_annotation_pipeline/yoloe_sam2_first_frame/first_frame_detections.json
