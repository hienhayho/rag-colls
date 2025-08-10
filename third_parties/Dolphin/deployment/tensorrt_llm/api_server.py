# copied from: https://github.com/NVIDIA/TensorRT-LLM/blob/v0.18.1/examples/apps/fastapi_server.py

#!/usr/bin/env python
import asyncio
import base64
import io
import logging
import signal
from http import HTTPStatus
from PIL import Image
from typing import Optional

import click
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from tensorrt_llm.executor import CppExecutorError, RequestError
from dolphin_runner import DolphinRunner, InferenceConfig

TIMEOUT_KEEP_ALIVE = 5  # seconds.


async def decode_image(image_base64: str) -> Image.Image:
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data))
    return image


class LlmServer:
    def __init__(self, runner: DolphinRunner):
        self.runner = runner
        self.app = FastAPI()
        self.register_routes()

    def register_routes(self):
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/generate", self.generate, methods=["POST"])

    async def health(self) -> Response:
        return Response(status_code=200)

    async def generate(self, request: Request) -> Response:
        """Generate completion for the request.

        The request should be a JSON object with the following fields:
        - prompt: the prompt to use for the generation.
        - image_base64: the image to use for the generation.
        """
        request_dict = await request.json()

        prompt = request_dict.pop("prompt", "")
        logging.info(f"request prompt: {prompt}")
        image_base64 = request_dict.pop("image_base64", "")
        image = await decode_image(image_base64)

        try:
            output_texts = self.runner.run([prompt], [image], 4024)
            output_texts = [texts[0] for texts in output_texts]
            return JSONResponse({"text": output_texts[0]})
        except RequestError as e:
            return JSONResponse(content=str(e), status_code=HTTPStatus.BAD_REQUEST)
        except CppExecutorError:
            # If internal executor error is raised, shutdown the server
            signal.raise_signal(signal.SIGINT)

    async def __call__(self, host, port):
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info",
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        )
        await uvicorn.Server(config).serve()


@click.command()
@click.option("--hf_model_dir", type=str, required=True)
@click.option("--visual_engine_dir", type=str, required=True)
@click.option("--llm_engine_dir", type=str, required=True)
@click.option("--max_batch_size", type=int, default=16)
@click.option("--max_new_tokens", type=int, default=4024)
@click.option("--host", type=str, default=None)
@click.option("--port", type=int, default=8000)
def entrypoint(
    hf_model_dir: str,
    visual_engine_dir: str,
    llm_engine_dir: str,
    max_batch_size: int,
    max_new_tokens: int,
    host: Optional[str] = None,
    port: int = 8000,
):
    host = host or "0.0.0.0"
    port = port or 8000
    logging.info(f"Starting server at {host}:{port}")

    config = InferenceConfig(
        max_new_tokens=max_new_tokens,
        batch_size=max_batch_size,
        log_level="info",
        hf_model_dir=hf_model_dir,
        visual_engine_dir=visual_engine_dir,
        llm_engine_dir=llm_engine_dir,
    )

    dolphin_runner = DolphinRunner(config)
    server = LlmServer(runner=dolphin_runner)

    asyncio.run(server(host, port))


if __name__ == "__main__":
    entrypoint()
