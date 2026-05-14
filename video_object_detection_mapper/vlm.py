from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video_object_detection_mapper import common


SYSTEM_PROMPT = "You are a precise robotic manipulation perception assistant. Return only valid JSON."

TARGET_REFERRING_PROMPT = """
Given the robot instruction, the first frame, and the final frame of the video,
identify the object that the robot should directly manipulate.

Instruction: {instruction}
First-frame image size: {width} x {height} pixels

Return exactly this JSON schema:
{{
  "target_object": "short object name for text-image matching",
  "referring_expression": "specific phrase that can be used to point to the target object in the first frame",
  "confidence": 0.0
}}

Use the final frame only as temporal context. The referring_expression must
describe the same target object but may include useful visual or spatial
attributes from the first frame. Do not return points or boxes. Return only
valid JSON.
""".strip()


@dataclass
class ChatGenerationConfig:
    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 0.95
    top_k: int | None = 20
    min_p: float | None = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0


class VLLMChatClient:
    def __init__(self, model: str, base_url: str, api_key: str, generation: ChatGenerationConfig, timeout: int, print_raw_response: bool = False):
        self.model = model
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.api_key = api_key
        self.generation = generation
        self.timeout = timeout
        self.print_raw_response = print_raw_response

    def chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.generation.max_tokens,
            "temperature": self.generation.temperature,
            "top_p": self.generation.top_p,
            "presence_penalty": self.generation.presence_penalty,
        }
        if self.generation.top_k is not None:
            payload["top_k"] = self.generation.top_k
        if self.generation.min_p is not None:
            payload["min_p"] = self.generation.min_p
        if self.generation.repetition_penalty != 1.0:
            payload["repetition_penalty"] = self.generation.repetition_penalty
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        if self.print_raw_response:
            print(json.dumps(response_payload, ensure_ascii=False, indent=2))
        return response_payload["choices"][0]["message"]


class ChatJSONClient:
    def __init__(self, chat_client: VLLMChatClient):
        self.chat_client = chat_client

    def chat_json(self, prompt: str, image_paths: list[Path]) -> tuple[dict[str, Any], str, str]:
        content = [common.text_part(prompt)]
        for image_path in image_paths:
            content.append(common.image_part(image_path))
        message = self.chat_client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
        )
        raw_content = common.assistant_content(message)
        reasoning = common.assistant_reasoning(message)
        return common.parse_vlm_json(raw_content), raw_content, reasoning


def build_vllm_json_client(args) -> ChatJSONClient:
    generation = ChatGenerationConfig(
        max_tokens=args.vlm_max_new_tokens,
        temperature=args.vlm_temperature,
        top_p=args.vlm_top_p,
        top_k=args.vlm_top_k,
        min_p=args.vlm_min_p,
        presence_penalty=args.vlm_presence_penalty,
        repetition_penalty=args.vlm_repetition_penalty,
    )
    return ChatJSONClient(
        VLLMChatClient(
            model=args.vlm_model,
            base_url=args.vllm_base_url,
            api_key=args.vllm_api_key,
            generation=generation,
            timeout=args.vlm_timeout,
            print_raw_response=args.print_raw_response,
        )
    )


def extract_target_and_referring_expression(
    client: ChatJSONClient,
    instruction: str,
    first_frame_path: Path,
    last_frame_path: Path,
    width: int,
    height: int,
) -> dict[str, Any]:
    prompt = TARGET_REFERRING_PROMPT.format(instruction=instruction, width=width, height=height)
    parsed, raw_content, reasoning = client.chat_json(prompt, [first_frame_path, last_frame_path])
    target_object = str(parsed.get("target_object") or parsed.get("object") or parsed.get("target") or "").strip()
    referring_expression = str(parsed.get("referring_expression") or parsed.get("referring") or target_object).strip()
    if not target_object:
        raise RuntimeError(f"VLM did not return target_object. Raw response: {raw_content}")
    if not referring_expression:
        referring_expression = target_object
    return {
        "model": client.chat_client.model,
        "prompt": prompt,
        "target_object": target_object,
        "referring_expression": referring_expression,
        "confidence": parsed.get("confidence"),
        "raw_response": parsed,
        "raw_content": raw_content,
        "reasoning": reasoning,
    }
