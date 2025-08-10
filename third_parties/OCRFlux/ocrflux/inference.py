import json
import copy
from PIL import Image
from pypdf import PdfReader
from vllm import LLM, SamplingParams
from ocrflux.image_utils import get_page_image
from ocrflux.table_format import table_matrix2html
from ocrflux.prompts import (
    PageResponse,
    build_page_to_markdown_prompt,
    build_element_merge_detect_prompt,
    build_html_table_merge_prompt,
)


def build_qwen2_5_vl_prompt(question):
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{question}<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_page_to_markdown_query(
    file_path: str,
    page_number: int,
    target_longest_image_dim: int = 1024,
    image_rotation: int = 0,
) -> dict:
    assert image_rotation in [
        0,
        90,
        180,
        270,
    ], "Invalid image rotation provided in build_page_query"
    image = get_page_image(
        file_path,
        page_number,
        target_longest_image_dim=target_longest_image_dim,
        image_rotation=image_rotation,
    )
    question = build_page_to_markdown_prompt()
    prompt = build_qwen2_5_vl_prompt(question)
    query = {
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    }
    return query


def build_element_merge_detect_query(text_list_1, text_list_2) -> dict:
    image = Image.new("RGB", (28, 28), color="black")
    question = build_element_merge_detect_prompt(text_list_1, text_list_2)
    prompt = build_qwen2_5_vl_prompt(question)
    query = {
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    }
    return query


def build_html_table_merge_query(text_1, text_2) -> dict:
    image = Image.new("RGB", (28, 28), color="black")
    question = build_html_table_merge_prompt(text_1, text_2)
    prompt = build_qwen2_5_vl_prompt(question)
    query = {
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    }
    return query


def build_document_text(
    page_to_markdown_result,
    element_merge_detect_result,
    html_table_merge_result,
):
    page_to_markdown_keys = list(page_to_markdown_result.keys())
    element_merge_detect_keys = list(element_merge_detect_result.keys())
    html_table_merge_keys = list(html_table_merge_result.keys())

    for page_1, page_2, elem_idx_1, elem_idx_2 in sorted(
        html_table_merge_keys, key=lambda x: -x[0]
    ):
        page_to_markdown_result[page_1][elem_idx_1] = html_table_merge_result[
            (page_1, page_2, elem_idx_1, elem_idx_2)
        ]
        page_to_markdown_result[page_2][elem_idx_2] = ""

    for page_1, page_2 in sorted(
        element_merge_detect_keys, key=lambda x: -x[0]
    ):
        for elem_idx_1, elem_idx_2 in element_merge_detect_result[
            (page_1, page_2)
        ]:
            if (
                len(page_to_markdown_result[page_1][elem_idx_1]) == 0
                or page_to_markdown_result[page_1][elem_idx_1][-1] == "-"
                or (
                    "\u4e00"
                    <= page_to_markdown_result[page_1][elem_idx_1][-1]
                    <= "\u9fff"
                )
            ):
                page_to_markdown_result[page_1][elem_idx_1] = (
                    page_to_markdown_result[page_1][elem_idx_1]
                    + ""
                    + page_to_markdown_result[page_2][elem_idx_2]
                )
            else:
                page_to_markdown_result[page_1][elem_idx_1] = (
                    page_to_markdown_result[page_1][elem_idx_1]
                    + " "
                    + page_to_markdown_result[page_2][elem_idx_2]
                )
            page_to_markdown_result[page_2][elem_idx_2] = ""

    document_text_list = []
    for page in page_to_markdown_keys:
        page_text_list = [s for s in page_to_markdown_result[page] if s]
        document_text_list += page_text_list
    return "\n\n".join(document_text_list)


def parse(llm, file_path, skip_cross_page_merge=False, max_page_retries=0):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=8192)
    if file_path.lower().endswith(".pdf"):
        try:
            reader = PdfReader(file_path)
            num_pages = reader.get_num_pages()
        except Exception as e:
            print(
                f"Error reading PDF file: {file_path} {e}. Please check the file path and format."
            )
            return None
    else:
        num_pages = 1

    try:
        # Stage 1: Page to Markdown
        page_to_markdown_query_list = [
            build_page_to_markdown_query(file_path, page_num)
            for page_num in range(1, num_pages + 1)
        ]
        responses = llm.generate(
            page_to_markdown_query_list, sampling_params=sampling_params
        )
        results = [response.outputs[0].text for response in responses]
        page_to_markdown_result = {}
        retry_list = []
        for i, result in enumerate(results):
            try:
                json_data = json.loads(result)
                page_response = PageResponse(**json_data)
                natural_text = page_response.natural_text
                markdown_element_list = []
                for text in natural_text.split("\n\n"):
                    if text.startswith("<Image>") and text.endswith(
                        "</Image>"
                    ):
                        pass
                    elif text.startswith("<table>") and text.endswith(
                        "</table>"
                    ):
                        try:
                            new_text = table_matrix2html(text)
                        except:
                            new_text = (
                                text.replace("<t>", "")
                                .replace("<l>", "")
                                .replace("<lt>", "")
                            )
                        markdown_element_list.append(new_text)
                    else:
                        markdown_element_list.append(text)
                page_to_markdown_result[i + 1] = markdown_element_list
            except:
                retry_list.append(i)

        attempt = 0
        while len(retry_list) > 0 and attempt < max_page_retries:
            retry_page_to_markdown_query_list = [
                build_page_to_markdown_query(file_path, i + 1)
                for i in retry_list
            ]
            retry_sampling_params = SamplingParams(
                temperature=0.1 * attempt, max_tokens=8192
            )
            responses = llm.generate(
                retry_page_to_markdown_query_list,
                sampling_params=retry_sampling_params,
            )
            results = [response.outputs[0].text for response in responses]
            next_retry_list = []
            for i, result in zip(retry_list, results):
                try:
                    json_data = json.loads(result)
                    page_response = PageResponse(**json_data)
                    natural_text = page_response.natural_text
                    markdown_element_list = []
                    for text in natural_text.split("\n\n"):
                        if text.startswith("<Image>") and text.endswith(
                            "</Image>"
                        ):
                            pass
                        elif text.startswith("<table>") and text.endswith(
                            "</table>"
                        ):
                            try:
                                new_text = table_matrix2html(text)
                            except:
                                new_text = (
                                    text.replace("<t>", "")
                                    .replace("<l>", "")
                                    .replace("<lt>", "")
                                )
                            markdown_element_list.append(new_text)
                        else:
                            markdown_element_list.append(text)
                    page_to_markdown_result[i + 1] = markdown_element_list
                except:
                    next_retry_list.append(i)
            retry_list = next_retry_list
            attempt += 1

        page_texts = {}
        fallback_pages = []
        for page_number in range(1, num_pages + 1):
            if page_number not in page_to_markdown_result.keys():
                fallback_pages.append(page_number - 1)
            else:
                page_texts[str(page_number - 1)] = "\n\n".join(
                    page_to_markdown_result[page_number]
                )

        if skip_cross_page_merge:
            document_text_list = []
            for i in range(num_pages):
                if i not in fallback_pages:
                    document_text_list.append(page_texts[str(i)])
            document_text = "\n\n".join(document_text_list)
            return {
                "orig_path": file_path,
                "num_pages": num_pages,
                "document_text": document_text,
                "page_texts": page_texts,
                "fallback_pages": fallback_pages,
            }

        # Stage 2: Element Merge Detect
        element_merge_detect_keys = []
        element_merge_detect_query_list = []
        for page_num in range(1, num_pages):
            if (
                page_num in page_to_markdown_result.keys()
                and page_num + 1 in page_to_markdown_result.keys()
            ):
                element_merge_detect_query_list.append(
                    build_element_merge_detect_query(
                        page_to_markdown_result[page_num],
                        page_to_markdown_result[page_num + 1],
                    )
                )
                element_merge_detect_keys.append((page_num, page_num + 1))
        responses = llm.generate(
            element_merge_detect_query_list, sampling_params=sampling_params
        )
        results = [response.outputs[0].text for response in responses]
        element_merge_detect_result = {}
        for key, result in zip(element_merge_detect_keys, results):
            try:
                element_merge_detect_result[key] = eval(result)
            except:
                pass

        # Stage 3: HTML Table Merge
        html_table_merge_keys = []
        for key, result in element_merge_detect_result.items():
            page_1, page_2 = key
            for elem_idx_1, elem_idx_2 in result:
                text_1 = page_to_markdown_result[page_1][elem_idx_1]
                text_2 = page_to_markdown_result[page_2][elem_idx_2]
                if (
                    text_1.startswith("<table>")
                    and text_1.endswith("</table>")
                    and text_2.startswith("<table>")
                    and text_2.endswith("</table>")
                ):
                    html_table_merge_keys.append(
                        (page_1, page_2, elem_idx_1, elem_idx_2)
                    )

        html_table_merge_keys = sorted(
            html_table_merge_keys, key=lambda x: -x[0]
        )

        html_table_merge_result = {}
        page_to_markdown_result_tmp = copy.deepcopy(page_to_markdown_result)
        i = 0
        while i < len(html_table_merge_keys):
            tmp = set()
            keys = []
            while i < len(html_table_merge_keys):
                page_1, page_2, elem_idx_1, elem_idx_2 = html_table_merge_keys[
                    i
                ]
                if (page_2, elem_idx_2) in tmp:
                    break
                tmp.add((page_1, elem_idx_1))
                keys.append((page_1, page_2, elem_idx_1, elem_idx_2))
                i += 1

            html_table_merge_query_list = [
                build_html_table_merge_query(
                    page_to_markdown_result_tmp[page_1][elem_idx_1],
                    page_to_markdown_result_tmp[page_2][elem_idx_2],
                )
                for page_1, page_2, elem_idx_1, elem_idx_2 in keys
            ]
            responses = llm.generate(
                html_table_merge_query_list, sampling_params=sampling_params
            )
            results = [response.outputs[0].text for response in responses]
            for key, result in zip(keys, results):
                if result.startswith("<table>") and result.endswith(
                    "</table>"
                ):
                    html_table_merge_result[key] = result
                    page_to_markdown_result_tmp[page_1][elem_idx_1] = result

        document_text = build_document_text(
            page_to_markdown_result,
            element_merge_detect_result,
            html_table_merge_result,
        )
        return {
            "orig_path": file_path,
            "num_pages": num_pages,
            "document_text": document_text,
            "page_texts": page_texts,
            "fallback_pages": fallback_pages,
        }
    except Exception as e:
        print(f"Error during parsing: {e}")
        return None


if __name__ == "__main__":
    file_path = "test.pdf"
    llm = LLM(
        model="ChatDOC/OCRFlux-3B",
        gpu_memory_utilization=0.8,
        max_model_len=8192,
    )
    result = parse(llm, file_path, max_page_retries=4)
    if result != None:
        document_markdown = result["document_text"]
        print(document_markdown)
        with open("test.md", "w") as f:
            f.write(document_markdown)
    else:
        print("Parse failed")
