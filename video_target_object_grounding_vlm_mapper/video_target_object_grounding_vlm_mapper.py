import ast
import json
import os
import re
import tempfile
from typing import Dict, Optional

import numpy as np
from loguru import logger

from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE
from data_juicer.utils.constant import Fields
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import (
    get_model,
    prepare_model,
    prepare_qwen_vl_inputs_for_vllm,
    torch,
    update_sampling_params,
)
from data_juicer.utils.ray_utils import is_ray_mode

from ..base_op import OPERATORS, TAGGING_OPS, Mapper
from ..op_fusion import LOADED_VIDEOS

cv2 = LazyLoader("cv2", "opencv-contrib-python")
vllm = LazyLoader("vllm")

OP_NAME = "video_target_object_grounding_vlm_mapper"


DEFAULT_SYSTEM_PROMPT = """
You are a precise robotic manipulation perception assistant.
Return only valid JSON.
"""

DEFAULT_INPUT_TEMPLATE = """
Given the robot language instruction below and the first frame of the video,
identify the target object being manipulated and localize it in the image.

Instruction: {instruction}

Return exactly this JSON schema:
{{
  "target_object": "short object name",
  "bbox": [x1, y1, x2, y2],
  "center": [cx, cy],
  "confidence": 0.0
}}

Use absolute pixel coordinates in the image coordinate system, with the origin
at the top-left corner. If the target object is not visible, use null for
"bbox" and "center".
"""

DEFAULT_TAG_FIELD = "video_target_object_grounding"


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoTargetObjectGroundingVLMMapper(Mapper):
    """Ground the manipulation target object from a video's first frame.

    This operator reads the first frame of each input video, sends it together with the
    language instruction to a VLM such as Qwen-VL, and stores the target object name,
    2D bbox, and bbox center in the sample metadata.
    """

    _accelerator = "cuda"

    def __init__(
        self,
        api_or_hf_model: str = "Qwen/Qwen3-VL-8B-Instruct",
        enable_vllm: bool = True,
        instruction_key: Optional[str] = None,
        tag_field_name: str = DEFAULT_TAG_FIELD,
        system_prompt: Optional[str] = None,
        input_template: Optional[str] = None,
        frame_dir: Optional[str] = None,
        keep_frame_path: bool = False,
        model_params: Optional[Dict] = None,
        sampling_params: Optional[Dict] = None,
        try_num: int = 3,
        *args,
        **kwargs,
    ):
        """
        :param api_or_hf_model: Hugging Face model id or local path for the VLM.
        :param enable_vllm: Whether to use vLLM. If False, use Hugging Face
            transformers generation.
        :param instruction_key: Sample key containing the robotic language
            instruction. If None, use the operator's text key.
        :param tag_field_name: Metadata field used to store grounding results.
        :param system_prompt: Optional system prompt.
        :param input_template: Prompt template. It must accept `{instruction}`.
        :param frame_dir: Directory for temporary first-frame JPEGs.
        :param keep_frame_path: Whether to keep the first-frame JPEG path in
            the output metadata. If False, temporary images are deleted.
        :param model_params: Parameters for initializing the model.
        :param sampling_params: Extra generation parameters.
        :param try_num: Retry attempts for generation/parsing.
        """
        kwargs["memory"] = "70GB" if kwargs.get("memory", 0) == 0 else kwargs["memory"]
        super().__init__(*args, **kwargs)
        self.enable_vllm = enable_vllm
        self.instruction_key = instruction_key or self.text_key
        self.tag_field_name = tag_field_name
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.input_template = input_template or DEFAULT_INPUT_TEMPLATE
        self.frame_dir = frame_dir or DATA_JUICER_ASSETS_CACHE
        self.keep_frame_path = keep_frame_path
        self.try_num = try_num

        model_params = model_params or {}
        sampling_params = sampling_params or {}
        if not sampling_params:
            sampling_params = {
                "temperature": 0,
                "max_tokens": 512,
                "top_p": 1.0,
            }
        sampling_params = update_sampling_params(sampling_params, api_or_hf_model, self.enable_vllm)

        if self.enable_vllm:
            if not is_ray_mode():
                self.num_proc = 1
            self.model_key = prepare_model(
                model_type="vllm",
                pretrained_model_name_or_path=api_or_hf_model,
                return_processor=True,
                **model_params,
            )
            self.sampling_params = vllm.SamplingParams(**sampling_params)
        else:
            self.model_key = prepare_model(
                model_type="huggingface",
                pretrained_model_name_or_path=api_or_hf_model,
                return_pipe=False,
                trust_remote_code=True,
                **model_params,
            )
            self.sampling_params = sampling_params

    def _extract_first_frame(self, video_path):
        os.makedirs(self.frame_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        cap.release()
        if not success:
            return None, None, None

        height, width = frame.shape[:2]
        fd, frame_path = tempfile.mkstemp(
            suffix=".jpg",
            prefix="dj_first_frame_",
            dir=self.frame_dir,
        )
        os.close(fd)
        cv2.imwrite(frame_path, frame)
        return frame_path, width, height

    def _build_messages(self, instruction, frame_path, for_vllm):
        prompt = self.input_template.format(instruction=instruction)
        if for_vllm:
            image_content = {"type": "image", "image": frame_path}
        else:
            image_content = {"type": "image", "image": frame_path}

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_content,
                ],
            }
        )
        return messages

    @staticmethod
    def _strip_json_markers(raw_output):
        text = raw_output.strip()
        for marker in ["```json", "```", "JSON:", "Response:"]:
            text = text.replace(marker, "")
        return text.strip()

    @classmethod
    def parse_output(cls, raw_output):
        json_str = cls._strip_json_markers(raw_output)
        try:
            result = json.loads(json_str, strict=False)
        except Exception:
            try:
                result = ast.literal_eval(json_str)
            except Exception:
                match = re.search(r"\{.*\}", json_str, flags=re.S)
                if not match:
                    logger.warning(f"Failed to parse VLM grounding output: {raw_output}")
                    return {}
                try:
                    result = json.loads(match.group(0), strict=False)
                except Exception:
                    logger.warning(f"Failed to parse VLM grounding output: {raw_output}")
                    return {}

        if isinstance(result, list) and result:
            result = result[0]
        if not isinstance(result, dict):
            return {}
        return result

    @staticmethod
    def _normalize_bbox(bbox, width, height):
        if bbox is None:
            return None
        if isinstance(bbox, str):
            nums = re.findall(r"-?\d+(?:\.\d+)?", bbox)
            bbox = [float(num) for num in nums[:4]]
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None

        x1, y1, x2, y2 = [float(v) for v in bbox]
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        x1 = max(0.0, min(x1, float(width)))
        x2 = max(0.0, min(x2, float(width)))
        y1 = max(0.0, min(y1, float(height)))
        y2 = max(0.0, min(y2, float(height)))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    @staticmethod
    def _normalize_center(center, width, height):
        if center is None:
            return None
        if isinstance(center, str):
            nums = re.findall(r"-?\d+(?:\.\d+)?", center)
            center = [float(num) for num in nums[:2]]
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            return None
        cx, cy = [float(v) for v in center]
        cx = max(0.0, min(cx, float(width)))
        cy = max(0.0, min(cy, float(height)))
        return [cx, cy]

    def _postprocess_result(self, parsed, width, height, frame_path):
        target_object = (
            parsed.get("target_object")
            or parsed.get("object")
            or parsed.get("target")
            or parsed.get("label")
        )
        bbox = self._normalize_bbox(
            parsed.get("bbox") or parsed.get("bbox_2d") or parsed.get("box"),
            width,
            height,
        )
        if bbox is None:
            center = self._normalize_center(parsed.get("center") or parsed.get("center_xy"), width, height)
        else:
            center = [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]

        result = {
            "target_object": str(target_object).strip() if target_object else "",
            "bbox_xyxy": bbox,
            "center_xy": center,
            "frame_size": [width, height],
            "raw_response": parsed,
        }
        if "confidence" in parsed:
            result["confidence"] = parsed["confidence"]
        if self.keep_frame_path:
            result["first_frame_path"] = frame_path
        return result

    def _generate(self, messages, frame_path, rank):
        if self.enable_vllm:
            model, processor = get_model(self.model_key, rank, self.use_cuda())
            inputs = [prepare_qwen_vl_inputs_for_vllm(messages, processor)]
            outputs = model.generate(inputs, sampling_params=self.sampling_params)
            return outputs[0].outputs[0].text

        model, processor = get_model(self.model_key, rank, self.use_cuda())
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512, **self.sampling_params)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def process_single(self, sample=None, rank=None):
        if self.tag_field_name in sample[Fields.meta]:
            return sample

        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.meta][self.tag_field_name] = []
            return sample

        instruction = sample.get(self.instruction_key, "")
        if isinstance(instruction, (list, tuple)):
            instruction = " ".join([str(item) for item in instruction])
        instruction = str(instruction)

        results = []
        for video_path in sample[self.video_key]:
            frame_path, width, height = self._extract_first_frame(video_path)
            if frame_path is None:
                results.append(
                    {
                        "target_object": "",
                        "bbox_xyxy": None,
                        "center_xy": None,
                        "frame_size": None,
                        "raw_response": {},
                        "error": "failed_to_read_first_frame",
                    }
                )
                continue

            messages = self._build_messages(instruction, frame_path, self.enable_vllm)
            parsed = {}
            for _ in range(self.try_num):
                try:
                    raw_output = self._generate(messages, frame_path, rank)
                    parsed = self.parse_output(raw_output)
                    if parsed:
                        break
                except Exception as e:
                    logger.warning(f"VLM grounding failed for {video_path}: {e}")

            results.append(self._postprocess_result(parsed, width, height, frame_path))
            if not self.keep_frame_path:
                try:
                    os.remove(frame_path)
                except OSError:
                    pass

        sample[Fields.meta][self.tag_field_name] = np.array(results, dtype=object)
        return sample
