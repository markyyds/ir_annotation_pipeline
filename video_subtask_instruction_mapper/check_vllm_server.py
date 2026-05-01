#!/usr/bin/env python3
"""Check local OpenAI-compatible vLLM server and served model names."""

from __future__ import annotations

import argparse
import json
import urllib.request


def get_json(url: str, api_key: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check vLLM OpenAI-compatible server.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = get_json(f"{args.base_url.rstrip('/')}/models", args.api_key)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
