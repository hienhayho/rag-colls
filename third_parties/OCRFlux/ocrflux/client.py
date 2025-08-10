import asyncio
import base64
import json
import copy
import traceback
from io import BytesIO
from argparse import Namespace
from urllib.parse import urlparse

from PIL import Image
from pypdf import PdfReader

from ocrflux.image_utils import get_page_image
from ocrflux.table_format import table_matrix2html
from ocrflux.prompts import PageResponse, build_page_to_markdown_prompt, build_element_merge_detect_prompt, build_html_table_merge_prompt

def build_page_to_markdown_query(args, file_path: str, page_number: int, target_longest_image_dim: int = 1024, image_rotation: int = 0) -> dict:
    assert image_rotation in [0, 90, 180, 270], "Invalid image rotation provided in build_page_query"

    image = get_page_image(file_path, page_number, target_longest_image_dim=target_longest_image_dim, image_rotation=image_rotation)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_page_to_markdown_prompt()},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ],
        "temperature": 0.0,
    }

def build_element_merge_detect_query(args,text_list_1,text_list_2) -> dict:
    image = Image.new('RGB', (28, 28), color='black')

    buffered = BytesIO()
    image.save(buffered, format="PNG")

    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_element_merge_detect_prompt(text_list_1,text_list_2)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ],
        "temperature": 0.0,
    }

def build_html_table_merge_query(args,text_1,text_2) -> dict:
    image = Image.new('RGB', (28, 28), color='black')

    buffered = BytesIO()
    image.save(buffered, format="PNG")

    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_html_table_merge_prompt(text_1,text_2)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ],
        "temperature": 0.0,
    }

# Manual simple implementation of HTTP Post
# It feels strange perhaps, but httpx and aiohttp are very complex beasts
# Ex. the sessionpool in httpcore has 4 different locks in it, and I've noticed
# that at the scale of 100M+ requests, that they deadlock in different strange ways
async def apost(url, json_data):
    parsed_url = urlparse(url)
    host = parsed_url.hostname
    port = parsed_url.port or 80
    path = parsed_url.path or "/"

    writer = None
    try:
        reader, writer = await asyncio.open_connection(host, port)

        json_payload = json.dumps(json_data)
        request = (
            f"POST {path} HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(json_payload)}\r\n"
            f"Connection: close\r\n\r\n"
            f"{json_payload}"
        )
        writer.write(request.encode())
        await writer.drain()

        # Read status line
        status_line = await reader.readline()
        if not status_line:
            raise ConnectionError("No response from server")
        status_parts = status_line.decode().strip().split(" ", 2)
        if len(status_parts) < 2:
            raise ValueError(f"Malformed status line: {status_line.decode().strip()}")
        status_code = int(status_parts[1])

        # Read headers
        headers = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            key, _, value = line.decode().partition(":")
            headers[key.strip().lower()] = value.strip()

        # Read response body
        if "content-length" in headers:
            body_length = int(headers["content-length"])
            response_body = await reader.readexactly(body_length)
        else:
            raise ConnectionError("Anything other than fixed content length responses are not implemented yet")

        return status_code, response_body
    except Exception as e:
        # Pass through errors
        raise e
    finally:
        # But just make sure to close the socket on your way out
        if writer is not None:
            try:
                writer.close()
                await writer.wait_closed()
            except:
                pass

async def process_task(args, task_name, task_args):
    COMPLETION_URL = f"{args.url}:{args.port}/v1/chat/completions"
    MAX_RETRIES = args.max_page_retries
    attempt = 0
    while attempt < MAX_RETRIES:
        if task_name == 'page_to_markdown':
            query = build_page_to_markdown_query(args, *task_args)
        elif task_name == 'element_merge_detect':
            query = build_element_merge_detect_query(args, *task_args)
        elif task_name == 'html_table_merge':
            query = build_html_table_merge_query(args, *task_args)

        query["temperature"] = 0.1 * attempt

        try:
            status_code, response_body = await apost(COMPLETION_URL, json_data=query)

            if status_code != 200:
                raise ValueError(f"Error http status {status_code}")

            base_response_data = json.loads(response_body)
            response_content = base_response_data["choices"][0]["message"]["content"]

            if task_name == 'page_to_markdown':
                model_response_json = json.loads(response_content)
                page_response = PageResponse(**model_response_json)
                natural_text = page_response.natural_text
                markdown_element_list = []
                for text in natural_text.split('\n\n'):
                    if text.startswith("<Image>") and text.endswith("</Image>"):
                        pass
                    elif text.startswith("<table>") and text.endswith("</table>"):
                        try:
                            new_text = table_matrix2html(text)
                        except:
                            new_text = text.replace("<t>","").replace("<l>","").replace("<lt>","")
                        markdown_element_list.append(new_text)
                    else:
                        markdown_element_list.append(text)
                return_data = markdown_element_list

            elif task_name == 'element_merge_detect':
                return_data = eval(response_content)

            elif task_name == 'html_table_merge':
                if not (response_content.startswith("<table>") and response_content.endswith("</table>")):
                    raise ValueError("Response is not a table")
                return_data = response_content

            return return_data

        except Exception as e:
            traceback.print_exc()
            attempt += 1
    return None

def build_document_text(page_to_markdown_result, element_merge_detect_result, html_table_merge_result):
    page_to_markdown_keys = list(page_to_markdown_result.keys())
    element_merge_detect_keys = list(element_merge_detect_result.keys())
    html_table_merge_keys = list(html_table_merge_result.keys())

    for page_1,page_2,elem_idx_1,elem_idx_2 in sorted(html_table_merge_keys,key=lambda x: -x[0]):
        page_to_markdown_result[page_1][elem_idx_1] = html_table_merge_result[(page_1,page_2,elem_idx_1,elem_idx_2)]
        page_to_markdown_result[page_2][elem_idx_2] = ''

    for page_1,page_2 in sorted(element_merge_detect_keys,key=lambda x: -x[0]):
        for elem_idx_1,elem_idx_2 in element_merge_detect_result[(page_1,page_2)]:
            if len(page_to_markdown_result[page_1][elem_idx_1]) == 0 or page_to_markdown_result[page_1][elem_idx_1][-1] == '-' or ('\u4e00' <= page_to_markdown_result[page_1][elem_idx_1][-1] <= '\u9fff'):
                page_to_markdown_result[page_1][elem_idx_1] = page_to_markdown_result[page_1][elem_idx_1] + '' + page_to_markdown_result[page_2][elem_idx_2]
            else:
                page_to_markdown_result[page_1][elem_idx_1] = page_to_markdown_result[page_1][elem_idx_1] + ' ' + page_to_markdown_result[page_2][elem_idx_2]
            page_to_markdown_result[page_2][elem_idx_2] = ''

    document_text_list = []
    for page in page_to_markdown_keys:
        page_text_list = [s for s in page_to_markdown_result[page] if s]
        document_text_list += page_text_list
    return "\n\n".join(document_text_list)

async def request(args, file_path: str):
    if file_path.lower().endswith(".pdf"):
        try:
            reader = PdfReader(file_path)
            num_pages = reader.get_num_pages()
        except:
            return None
    else:
        num_pages = 1

    try:
        # Stage 1: Page to Markdown
        page_to_markdown_tasks = []
        results = []
        async with asyncio.TaskGroup() as tg:
            for page_num in range(1, num_pages + 1):
                task = tg.create_task(process_task(args, task_name='page_to_markdown', task_args=(file_path,page_num)))
                page_to_markdown_tasks.append(task)

        results = [task.result() for task in page_to_markdown_tasks]

        page_to_markdown_result = {}
        for i,result in enumerate(results):
            if result != None:
                page_to_markdown_result[i+1] = result

        page_texts = {}
        fallback_pages = []
        for page_number in range(1, num_pages+1):
            if page_number not in page_to_markdown_result.keys():
                fallback_pages.append(page_number-1)
            else:
                page_texts[str(page_number-1)] = "\n\n".join(page_to_markdown_result[page_number])

        if args.skip_cross_page_merge:
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
        element_merge_detect_tasks = []
        async with asyncio.TaskGroup() as tg:
            for page_num in range(1,num_pages):
                if page_num in page_to_markdown_result.keys() and page_num+1 in page_to_markdown_result.keys():
                    element_merge_detect_keys.append((page_num,page_num+1))
                    task = tg.create_task(process_task(args, task_name='element_merge_detect', task_args=(page_to_markdown_result[page_num],page_to_markdown_result[page_num+1])))
                    element_merge_detect_tasks.append(task)

        results = [task.result() for task in element_merge_detect_tasks]

        element_merge_detect_result = {}
        for key,result in zip(element_merge_detect_keys,results):
            if result != None:
                element_merge_detect_result[key] = result

        # Stage 3: HTML Table Merge
        html_table_merge_keys = []
        for key,result in element_merge_detect_result.items():
            page_1,page_2 = key
            for elem_idx_1,elem_idx_2 in result:
                text_1 = page_to_markdown_result[page_1][elem_idx_1]
                text_2 = page_to_markdown_result[page_2][elem_idx_2]
                if text_1.startswith("<table>") and text_1.endswith("</table>") and text_2.startswith("<table>") and text_2.endswith("</table>"):
                    html_table_merge_keys.append((page_1,page_2,elem_idx_1,elem_idx_2))

        html_table_merge_keys = sorted(html_table_merge_keys,key=lambda x: -x[0])

        html_table_merge_result = {}
        page_to_markdown_result_tmp = copy.deepcopy(page_to_markdown_result)
        i = 0
        while i < len(html_table_merge_keys):
            tmp = set()
            keys = []
            while i < len(html_table_merge_keys):
                page_1,page_2,elem_idx_1,elem_idx_2 = html_table_merge_keys[i]
                if (page_2,elem_idx_2) in tmp:
                    break
                tmp.add((page_1,elem_idx_1))
                keys.append((page_1,page_2,elem_idx_1,elem_idx_2))
                i += 1
            html_table_merge_tasks = []
            async with asyncio.TaskGroup() as tg:
                for page_1,page_2,elem_idx_1,elem_idx_2 in keys:
                    task = tg.create_task(process_task(args, task_name='html_table_merge', task_args=(page_to_markdown_result_tmp[page_1][elem_idx_1],page_to_markdown_result_tmp[page_2][elem_idx_2])))
                    html_table_merge_tasks.append(task)
            results = [task.result() for task in html_table_merge_tasks]
            for key,result in zip(keys,results):
                if result != None:
                    html_table_merge_result[key] = result
                    page_to_markdown_result_tmp[page_1][elem_idx_1] = result

        document_text = build_document_text(page_to_markdown_result, element_merge_detect_result, html_table_merge_result)

        return {
            "orig_path": file_path,
            "num_pages": num_pages,
            "document_text": document_text,
            "page_texts": page_texts,
            "fallback_pages": fallback_pages,
        }
    except Exception as e:
        traceback.print_exc()
        return None

if __name__ == "__main__":
    args = Namespace(
        model="ChatDOC/OCRFlux-3B",
        skip_cross_page_merge=False,
        max_page_retries=1,
        url="http://localhost",
        port=30024,
    )
    file_path = 'test.pdf'
    result = asyncio.run(request(args,file_path))
    print(result)
