"""
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import json
import os
from typing import Optional

import tensorrt_llm.profiler as profiler
import torch
from PIL import Image
from pydantic import BaseModel, Field
from tensorrt_llm import logger
from tensorrt_llm import mpi_rank
from tensorrt_llm.runtime import MultimodalModelRunner
from transformers import AutoTokenizer, DonutProcessor


class InferenceConfig(BaseModel):
    max_new_tokens: int = Field(128, description="Maximum new tokens to generate")
    batch_size: int = Field(1, description="Batch size for inference")
    log_level: str = Field("info", description="Logging level")
    visual_engine_dir: Optional[str] = Field(
        None, description="Directory for visual engine files"
    )
    visual_engine_name: str = Field(
        "model.engine", description="Visual engine filename"
    )
    llm_engine_dir: Optional[str] = Field(
        None, description="Directory for LLM engine files"
    )
    hf_model_dir: Optional[str] = Field(
        None, description="Hugging Face model directory"
    )
    input_text: Optional[str] = Field(None, description="Input text for inference")
    num_beams: int = Field(1, description="Number of beams for beam search")
    top_k: int = Field(1, description="Top-k sampling value")
    top_p: float = Field(0.0, description="Top-p (nucleus) sampling value")
    temperature: float = Field(1.0, description="Sampling temperature")
    repetition_penalty: float = Field(1.0, description="Repetition penalty factor")
    run_profiling: bool = Field(False, description="Enable profiling mode")
    profiling_iterations: int = Field(20, description="Number of profiling iterations")
    check_accuracy: bool = Field(False, description="Enable accuracy checking")
    video_path: Optional[str] = Field(None, description="Path to input video file")
    video_num_frames: Optional[int] = Field(
        None, description="Number of video frames to process"
    )
    image_path: Optional[str] = Field(None, description="Path to input image file")
    path_sep: str = Field(",", description="Path separator character")
    prompt_sep: str = Field(",", description="Prompt separator character")
    enable_context_fmha_fp32_acc: Optional[bool] = Field(
        None, description="Enable FP32 accumulation for context FMHA"
    )
    enable_chunked_context: bool = Field(
        False, description="Enable chunked context processing"
    )
    use_py_session: bool = Field(False, description="Use Python session instead of C++")
    kv_cache_free_gpu_memory_fraction: float = Field(
        0.9, description="Fraction of GPU memory free for KV cache", ge=0.0, le=1.0
    )
    cross_kv_cache_fraction: float = Field(
        0.5, description="Fraction of cross-attention KV cache", ge=0.0, le=1.0
    )
    multi_block_mode: bool = Field(
        True, description="Enable multi-block processing mode"
    )


class DolphinRunner(MultimodalModelRunner):
    def __init__(self, args):
        self.args = args

        self.runtime_rank = mpi_rank()
        device_id = self.runtime_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        self.device = "cuda:%d" % (device_id)

        self.stream = torch.cuda.Stream(torch.cuda.current_device())
        torch.cuda.set_stream(self.stream)

        # parse model type from visual engine config
        with open(os.path.join(self.args.visual_engine_dir, "config.json"), "r") as f:
            config = json.load(f)
        self.model_type = config["builder_config"]["model_type"]
        self.vision_precision = config["builder_config"]["precision"]
        self.decoder_llm = not (
            "t5" in self.model_type or self.model_type in ["nougat", "pix2struct"]
        )  # BLIP2-T5, pix2struct and Nougat are using encoder-decoder models as LLMs

        if self.model_type == "mllama":
            self.vision_input_names = [
                "pixel_values",
                "aspect_ratio_ids",
                "aspect_ratio_mask",
            ]
            self.vision_output_names = [
                "output",
            ]
        else:
            self.vision_input_names = ["input"]
            self.vision_output_names = ["output"]

        self.use_py_session = True

        self.init_image_encoder()
        self.init_tokenizer()
        self.init_processor()
        self.init_llm()

    def init_tokenizer(self):
        assert self.model_type == "nougat"
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_model_dir)
        self.tokenizer.padding_side = "right"

    def init_processor(self):
        assert self.model_type == "nougat"
        self.processor = DonutProcessor.from_pretrained(
            self.args.hf_model_dir, use_fast=True
        )

    def run(self, input_texts, input_images, max_new_tokens):
        prompts = [f"<s>{text.strip()} <Answer/>" for text in input_texts]
        images = self.processor(input_images, return_tensors="pt")["pixel_values"].to(
            "cuda"
        )
        prompt_ids = self.tokenizer(
            prompts, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to("cuda")

        # 🚨🚨🚨 Important! If the type of prompt_ids is not int32, the output will be wrong. 🚨🚨🚨
        prompt_ids = prompt_ids.to(torch.int32)

        logger.info("---------------------------------------------------------")
        logger.info(f"images size: {images.size()}")
        logger.info(
            f"prompt_ids: {prompt_ids}, size: {prompt_ids.size()}, dtype: {prompt_ids.dtype}"
        )
        logger.info("---------------------------------------------------------")

        output_texts = self.generate(
            input_texts,
            [None] * len(input_texts),
            images,
            prompt_ids,
            max_new_tokens,
            warmup=False,
        )

        return output_texts

    def generate(
        self,
        pre_prompt,
        post_prompt,
        image,
        decoder_input_ids,
        max_new_tokens,
        warmup=False,
        other_vision_inputs={},
        other_decoder_inputs={},
    ):
        if not warmup:
            profiler.start("Generate")
        input_ids, input_lengths, ptuning_args, visual_features = self.preprocess(
            warmup, pre_prompt, post_prompt, image, other_vision_inputs
        )

        if warmup:
            return None

        # use prompt tuning to pass multimodal features
        # model.generate() expects the following params (see layers/embedding.py):
        # args[0]: prompt embedding table, [batch_size, multimodal_len, hidden_size], later flattened to [batch_size * multimodal_len, hidden_size]
        # args[1]: prompt task ids, [batch_size]. in multimodal case, arange(batch_size), i.e. in VILA batching mode 2, each image is treated separately in the batch instead of concated together (although the prompt embedding table has to be concated)
        # args[2]: prompt task vocab size, [1]. assuming all table has the same length, which in multimodal case equals to multimodal_len
        profiler.start("LLM")
        if self.model_type in ["nougat", "pix2struct"]:
            # Trim encoder input_ids to match visual features shape
            ids_shape = (
                min(self.args.batch_size, len(pre_prompt)),
                visual_features.shape[1],
            )
            if self.model_type == "nougat":
                input_ids = torch.zeros(ids_shape, dtype=torch.int32)
            elif self.model_type == "pix2struct":
                input_ids = torch.ones(ids_shape, dtype=torch.int32)

        output_ids = self.model.generate(
            input_ids,
            decoder_input_ids,
            max_new_tokens,
            num_beams=self.args.num_beams,
            bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            debug_mode=False,
            prompt_embedding_table=ptuning_args[0],
            prompt_tasks=ptuning_args[1],
            prompt_vocab_size=ptuning_args[2],
        )
        profiler.stop("LLM")

        if mpi_rank() == 0:
            # Extract a list of tensors of shape beam_width x output_ids.
            output_beams_list = [
                self.tokenizer.batch_decode(
                    output_ids[batch_idx, :, decoder_input_ids.shape[1] :],
                    skip_special_tokens=False,
                )
                for batch_idx in range(
                    min(self.args.batch_size, decoder_input_ids.shape[0])
                )
            ]

            stripped_text = [
                [
                    output_beams_list[batch_idx][beam_idx]
                    .replace("</s>", "")
                    .replace("<pad>", "")
                    .strip()
                    for beam_idx in range(self.args.num_beams)
                ]
                for batch_idx in range(
                    min(self.args.batch_size, decoder_input_ids.shape[0])
                )
            ]
            profiler.stop("Generate")
            return stripped_text
        else:
            profiler.stop("Generate")
            return None


if __name__ == "__main__":
    config = InferenceConfig(
        max_new_tokens=4024,
        batch_size=16,
        log_level="info",
        hf_model_dir="./tmp/hf_models/Dolphin",
        visual_engine_dir="./tmp/trt_engines/Dolphin/vision_encoder",
        llm_engine_dir="./tmp/trt_engines/Dolphin/1-gpu/bfloat16",
    )

    model = DolphinRunner(config)

    image_path = "../../demo/page_imgs/page_1.jpeg"
    prompt = "Parse the reading order of this document."
    image = Image.open(image_path).convert("RGB")
    output_texts = model.run([prompt], [image], 4024)
    output_texts = [texts[0] for texts in output_texts]
    print(output_texts)
