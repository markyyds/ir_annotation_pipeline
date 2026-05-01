#!/usr/bin/env python3
"""Check local OpenAI-compatible vLLM server and run a tiny inference test."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from typing import Any


def get_json(url: str, api_key: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, api_key: str, payload: dict[str, Any], timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Request failed with HTTP {exc.code}: {body}") from exc


def first_model_id(models_payload: dict[str, Any]) -> str:
    models = models_payload.get("data", [])
    if not models:
        raise RuntimeError("The server responded, but /models returned no served models.")
    model_id = models[0].get("id")
    if not model_id:
        raise RuntimeError(f"Could not find a model id in /models response: {models_payload}")
    return str(model_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check vLLM OpenAI-compatible server.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default=None, help="Served model name. Defaults to the first /models entry.")
    parser.add_argument("--prompt", default="Reply with exactly: vllm-ok")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--skip-inference", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    models_payload = get_json(f"{base_url}/models", args.api_key)
    print("Models:")
    print(json.dumps(models_payload, indent=2, ensure_ascii=False))

    if args.skip_inference:
        return

    model_id = args.model or first_model_id(models_payload)
    chat_payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    response = post_json(f"{base_url}/chat/completions", args.api_key, chat_payload, args.timeout)
    content = response["choices"][0]["message"].get("content", "")
    print("\nInference smoke test:")
    print(f"model: {model_id}")
    print(f"prompt: {args.prompt}")
    print(f"response: {content}")


if __name__ == "__main__":
    main()
