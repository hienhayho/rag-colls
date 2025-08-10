# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example Python client for `vllm.entrypoints.api_server`
Start the demo server:
    python -m vllm.entrypoints.api_server --model <model_name>

NOTE: The API server is used only for demonstration and simple performance
benchmarks. It is not intended for production use.
For production use, we recommend `vllm serve` and the OpenAI client API.
"""

import argparse
import base64
import json
from argparse import Namespace
from collections.abc import Iterable

import requests


def clear_line(n: int = 1) -> None:
    LINE_UP = "\033[1A"
    LINE_CLEAR = "\x1b[2K"
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def encode_image_base64(image_path: str) -> str:
    """Encode local image to base64 format."""

    with open(image_path, "rb") as f:
        image_data = f.read()
        result = base64.b64encode(image_data).decode("utf-8")

    return result


def post_http_request(
    prompt: str, image_path: str, api_url: str, stream: bool = False
) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": prompt,
        "image_base64": encode_image_base64(image_path),
    }
    response = requests.post(api_url, headers=headers, json=pload, stream=stream)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[list[str]]:
    for chunk in response.iter_lines(
        chunk_size=8192, decode_unicode=False, delimiter=b"\n"
    ):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            yield output


def get_response(response: requests.Response) -> list[str]:
    data = json.loads(response.content)
    output = data["text"]
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--prompt", type=str, default="Parse the reading order of this document."
    )
    parser.add_argument(
        "--image_path", type=str, default="./demo/page_imgs/page_1.jpeg"
    )
    parser.add_argument("--stream", action="store_true")
    return parser.parse_args()


def main(args: Namespace):
    prompt = args.prompt
    image_path = args.image_path
    api_url = f"http://{args.host}:{args.port}/generate"
    stream = args.stream

    print(f"Prompt: {prompt!r}\n", flush=True)
    response = post_http_request(prompt, image_path, api_url, stream)

    if stream:
        num_printed_lines = 0
        for h in get_streaming_response(response):
            clear_line(num_printed_lines)
            num_printed_lines = 0
            for i, line in enumerate(h):
                num_printed_lines += 1
                print(f"Response {i}: {line!r}", flush=True)
    else:
        output = get_response(response)
        print(f"Response: {output!r}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
