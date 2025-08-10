import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from loguru import logger

try:
    import vllm_dolphin  # noqa: F401
    from vllm import LLM, SamplingParams
    from vllm.inputs import (
        ExplicitEncoderDecoderPrompt,
        TextPrompt,
        TokensPrompt,
    )
except ImportError as e:
    raise ImportError(
        "DolphinReader cannot be initialized since not all dependencies are available. "
        "Please install it with 'pip install rag-colls[dolphin]'."
    ) from e

from rag_colls.types.core.document import Document
from rag_colls.core.base.readers.base import BaseReader

from .utils import (
    ImageDimensions,
    convert_pdf_to_images,
    is_pdf_file,
    prepare_image,
    get_output_str,
    parse_layout_string,
    process_coordinates,
    save_figure_to_local,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DolphinReader(BaseReader):
    read_layout_prompt = "Parse the reading order of this document."
    read_text_prompt = "Read text in the image."
    parse_table_prompt = "Parse the table in the image."

    def __init__(
        self,
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_num_seqs: int = 8,
        max_tokens: int = 2048,
        download_dir: str = "./model_cache",
        show_progress: bool = True,
        batch_size: int = 2,
        image_dir: str = "output_figures",
    ):
        """
        Initialize the DolphinReader with vLLM Dolphin model.

        Args:
            dtype (str): Data type for the model. Default is `auto`.
            tensor_parallel_size (int): Number of GPUs to use for tensor parallelism.
            gpu_memory_utilization (float): vllm's GPU memory utilization.
            max_num_seqs (int): Maximum number of sequences to process in parallel.
            max_tokens (int): Maximum number of tokens to generate.
            download_dir (str): Directory to cache the model.
            show_progress (bool): Whether to show progress bars.
            batch_size (int): Batch size for processing elements.
            image_dir (str): Directory to save output figures.
        """
        self.llm = LLM(
            model="ByteDance/Dolphin",
            dtype=dtype,
            enforce_eager=True,
            trust_remote_code=True,
            max_num_seqs=max_num_seqs,
            hf_overrides={"architectures": ["DolphinForConditionalGeneration"]},
            download_dir=download_dir,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        self.sampling_params = SamplingParams(
            temperature=0.0,
            logprobs=0,
            max_tokens=max_tokens,
            prompt_logprobs=None,
            skip_special_tokens=False,
        )

        self.tokenizer = self.llm.llm_engine.get_tokenizer_group().tokenizer

        # The Dolphin model does not require an Encoder Prompt. To ensure vllm correctly allocates KV Cache,
        # it is necessary to simulate an Encoder Prompt.
        self.encoder_prompt = "0" * 783
        self.decoder_prompt = "<s>{prompt} <Answer/>"
        self.show_progress = show_progress
        self.batch_size = batch_size
        self.image_dir = image_dir

        logger.info("DolphinReader initialized !")

    def process_element_batch(self, elements: list[dict], prompt: str):
        """Process elements of the same type in batches"""
        results = []

        # Determine batch size

        # Process in batches
        for i in tqdm(
            range(0, len(elements), self.batch_size),
            total=len(elements) // self.batch_size,
            desc="Processing elements ...",
            disable=not self.show_progress,
        ):
            batch_elements = elements[i : i + self.batch_size]
            crops_list = [elem["crop"] for elem in batch_elements]

            enc_dec_prompts = [
                ExplicitEncoderDecoderPrompt(
                    encoder_prompt=TextPrompt(
                        prompt=self.encoder_prompt, multi_modal_data={"image": im}
                    ),
                    decoder_prompt=TokensPrompt(
                        prompt_token_ids=self.tokenizer(
                            self.decoder_prompt.format(prompt=prompt.strip()),
                            add_special_tokens=False,
                        )["input_ids"]
                    ),
                )
                for im in crops_list
            ]
            outputs = self.llm.generate(
                enc_dec_prompts, self.sampling_params, use_tqdm=False
            )

            # Add results
            for j, output in enumerate(outputs):
                elem = batch_elements[j]
                results.append(
                    {
                        "label": elem["label"],
                        "bbox": elem["bbox"],
                        "text": output.outputs[0].text.strip(),
                        "reading_order": elem["reading_order"],
                    }
                )

        return results

    def process_elements(
        self,
        layout_results: str,
        padded_image: np.ndarray,
        dims: ImageDimensions,
        save_dir: str | None = None,
        image_name: str = "image",
    ):
        """Parse all document elements with parallel decoding"""
        parsed_layout_results = parse_layout_string(layout_results)

        # Store text and table elements separately
        text_elements = []  # Text elements
        table_elements = []  # Table elements
        figure_results = []  # Image elements (no processing needed)
        previous_box = None
        reading_order = 0

        # Collect elements to process and group by type
        for bbox, label in parsed_layout_results:
            try:
                # Adjust coordinates
                x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, previous_box = (
                    process_coordinates(bbox, padded_image, dims, previous_box)
                )

                # Crop and parse element
                cropped = padded_image[y1:y2, x1:x2]
                if cropped.size > 0 and cropped.shape[0] > 3 and cropped.shape[1] > 3:
                    if label == "fig":
                        pil_crop = Image.fromarray(
                            cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                        )

                        figure_filename = save_figure_to_local(
                            pil_crop, save_dir, image_name, reading_order
                        )

                        # For figure regions, store relative path instead of base64
                        figure_results.append(
                            {
                                "label": label,
                                "text": f"![Figure](figures/{figure_filename})",
                                "figure_path": f"figures/{figure_filename}",
                                "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                                "reading_order": reading_order,
                            }
                        )
                    else:
                        # Prepare element for parsing
                        pil_crop = Image.fromarray(
                            cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                        )
                        element_info = {
                            "crop": pil_crop,
                            "label": label,
                            "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                            "reading_order": reading_order,
                        }

                        # Group by type
                        if label == "tab":
                            table_elements.append(element_info)
                        else:  # Text elements
                            text_elements.append(element_info)

                reading_order += 1

            except Exception as e:
                print(f"Error processing bbox with label {label}: {str(e)}")
                continue

        # Initialize results list
        recognition_results = figure_results.copy()

        # Process text elements (in batches)
        if text_elements:
            text_results = self.process_element_batch(
                elements=text_elements, prompt=self.read_text_prompt
            )
            recognition_results.extend(text_results)

        # Process table elements (in batches)
        if table_elements:
            table_results = self.process_element_batch(
                elements=table_elements, prompt=self.parse_table_prompt
            )
            recognition_results.extend(table_results)

        # Sort elements by reading order
        recognition_results.sort(key=lambda x: x.get("reading_order", 0))

        return recognition_results

    def _load_data(
        self,
        file_path: str | Path,
        should_split: bool = True,
        extra_info: dict | None = None,
    ) -> list[Document]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        file_name = file_path.name

        if is_pdf_file(file_path):
            images = convert_pdf_to_images(file_path)
        else:
            images = [Image.open(file_path)]

        if not extra_info:
            extra_info = {}

        extra_info["file_path"] = str(file_path)
        extra_info["should_split"] = should_split

        documents = []
        for i, image in tqdm(
            enumerate(images),
            total=len(images),
            desc="Processing images ...",
            disable=not self.show_progress,
        ):
            enc_dec_prompt = ExplicitEncoderDecoderPrompt(
                encoder_prompt=TextPrompt(
                    prompt=self.encoder_prompt, multi_modal_data={"image": image}
                ),
                decoder_prompt=TokensPrompt(
                    prompt_token_ids=self.tokenizer(
                        self.decoder_prompt.format(
                            prompt=self.read_layout_prompt.strip()
                        ),
                        add_special_tokens=False,
                    )["input_ids"]
                ),
            )
            outputs = self.llm.generate(
                enc_dec_prompt, self.sampling_params, use_tqdm=False
            )
            for output in outputs:
                layout_output = output.outputs[0].text.strip()

                padded_image, dims = prepare_image(image)
                recognition_results = self.process_elements(
                    layout_output,
                    padded_image,
                    dims,
                    save_dir=self.image_dir,
                    image_name=file_path.stem,
                )
                # TODO: Handle the recognition results
                # Currently only return the text
                text_str = get_output_str(recognition_results)
                document = Document(
                    document=text_str,
                    metadata={
                        "source": f"{file_name}: Page {i + 1}",
                        **extra_info,
                    },
                )
                documents.append(document)

        logger.info(f"Loaded {len(documents)} documents from {file_path}")

        return documents
