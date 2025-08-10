"""
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import numpy as np
import torch
from PIL import ImageOps
from torchvision import transforms
from torchvision.transforms.functional import resize

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class DolphinProcessor:
    def __init__(
        self,
        dp_config,
        tokenizer,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        transform_args = kwargs.get("transform_args", {})
        self.max_length = transform_args.get("max_length", 2048)
        self.input_size = transform_args.get("input_size", [896, 896])  # height, width
        if isinstance(self.input_size, int):
            self.input_size = [self.input_size, self.input_size]

        try:
            self.answer_start_token = self.tokenizer._prompt_end_token
        except AttributeError:
            print('No answer_start_token found, use "" instead')
            self.answer_start_token = ""

        self.prefix_answer_space_flag = dp_config.get("prefix_answer_space_flag", True)
        self.suffix_prompt_space_flag = dp_config.get("suffix_prompt_space_flag", True)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
                ),
            ]
        )

    def process_prompt_for_inference(self, prompt):
        prompt = prompt.replace("<image>\n", "")
        if not prompt.startswith("<s>"):
            prompt = "<s>" + prompt
        message_ids = [self.tokenizer.encode(prompt, add_special_tokens=False)]
        ids = torch.from_numpy(np.hstack(message_ids, dtype=np.int32))
        return ids.unsqueeze(0)

    def process_image_for_inference(self, image, return_img_size=False):
        image = resize(image, min(self.input_size))

        image.thumbnail((self.input_size[1], self.input_size[0]))
        origin_w, origin_h = image.size

        delta_width = self.input_size[1] - image.width
        delta_height = self.input_size[0] - image.height
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )
        image = ImageOps.expand(image, padding)
        if return_img_size:
            return self.transform(image).unsqueeze(0), (origin_w, origin_h)
        return self.transform(image).unsqueeze(0)
