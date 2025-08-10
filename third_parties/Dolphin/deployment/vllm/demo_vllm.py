"""
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import argparse
from argparse import Namespace
from PIL import Image

from vllm import LLM, SamplingParams
from vllm.inputs import ExplicitEncoderDecoderPrompt, TextPrompt, TokensPrompt

import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def offline_inference(
    model_id: str, prompt: str, image_path: str, max_tokens: int = 2048
):
    dtype = "float16" if torch.cuda.is_available() else "float32"
    # Create an encoder/decoder model instance
    llm = LLM(
        model=model_id,
        dtype=dtype,
        enforce_eager=True,
        trust_remote_code=True,
        max_num_seqs=8,
        hf_overrides={"architectures": ["DolphinForConditionalGeneration"]},
    )

    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0.0,
        logprobs=0,
        max_tokens=max_tokens,
        prompt_logprobs=None,
        skip_special_tokens=False,
    )

    # process prompt
    tokenizer = llm.llm_engine.get_tokenizer_group().tokenizer

    # The Dolphin model does not require an Encoder Prompt. To ensure vllm correctly allocates KV Cache,
    # it is necessary to simulate an Encoder Prompt.
    encoder_prompt = "0" * 783
    decoder_prompt = f"<s>{prompt.strip()} <Answer/>"

    image = Image.open(image_path)
    enc_dec_prompt = ExplicitEncoderDecoderPrompt(
        encoder_prompt=TextPrompt(
            prompt=encoder_prompt, multi_modal_data={"image": image}
        ),
        decoder_prompt=TokensPrompt(
            prompt_token_ids=tokenizer(decoder_prompt, add_special_tokens=False)[
                "input_ids"
            ]
        ),
    )

    # Generate output tokens from the prompts. The output is a list of
    # RequestOutput objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(enc_dec_prompt, sampling_params)

    print("------" * 8)
    # Print the outputs.
    for output in outputs:
        decoder_prompt_tokens = tokenizer.batch_decode(
            output.prompt_token_ids, skip_special_tokens=True
        )
        decoder_prompt = "".join(decoder_prompt_tokens)
        generated_text = output.outputs[0].text.strip()
        print(
            f"Decoder prompt: {decoder_prompt!r}, \nGenerated text: {generated_text!r}"
        )

        print("------" * 8)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ByteDance/Dolphin")
    parser.add_argument(
        "--image_path", type=str, default="./demo/page_imgs/page_1.jpeg"
    )
    parser.add_argument(
        "--prompt", type=str, default="Parse the reading order of this document."
    )
    return parser.parse_args()


def main(args: Namespace):
    model = args.model
    prompt = args.prompt
    image_path = args.image_path

    offline_inference(model, prompt, image_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
