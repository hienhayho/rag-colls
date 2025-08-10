import argparse
import asyncio
import atexit
import base64
import json
import logging
import shutil
import os
import copy
import random
import re
import sys
import time
from concurrent.futures.process import BrokenProcessPool
from io import BytesIO
from urllib.parse import urlparse

import httpx
from huggingface_hub import snapshot_download
from PIL import Image
from pypdf import PdfReader
from tqdm import tqdm

from ocrflux.check import (
    check_poppler_version,
    check_vllm_version,
    check_torch_gpu_available,
)
from ocrflux.image_utils import get_page_image, is_image
from ocrflux.table_format import table_matrix2html
from ocrflux.metrics import MetricsKeeper, WorkerTracker
from ocrflux.prompts import PageResponse, build_page_to_markdown_prompt, build_element_merge_detect_prompt, build_html_table_merge_prompt
from ocrflux.work_queue import LocalWorkQueue, WorkQueue

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

vllm_logger = logging.getLogger("vllm")
vllm_logger.propagate = False

file_handler = logging.FileHandler("OCRFlux-debug.log", mode="a")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
vllm_logger.addHandler(file_handler)

# Quiet logs from pypdf
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Global variables for token statistics
metrics = MetricsKeeper(window=60 * 5)
tracker = WorkerTracker()

def build_page_to_markdown_query(args, pdf_path: str, page_number: int, target_longest_image_dim: int, image_rotation: int = 0) -> dict:
    assert image_rotation in [0, 90, 180, 270], "Invalid image rotation provided in build_page_query"

    image = get_page_image(pdf_path, page_number, target_longest_image_dim=target_longest_image_dim, image_rotation=image_rotation)
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

async def process_task(args, worker_id, task_name, task_args):
    COMPLETION_URL = f"http://localhost:{args.port}/v1/chat/completions"
    MAX_RETRIES = args.max_page_retries
    TEMPERATURE_BY_ATTEMPT = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    exponential_backoffs = 0
    local_image_rotation = 0
    attempt = 0
    await tracker.track_work(worker_id, f"{worker_id}", "started")
    while attempt < MAX_RETRIES:
        if task_name == 'page_to_markdown':
            pdf_path,page_number = task_args
            query = build_page_to_markdown_query(args, pdf_path, page_number, args.target_longest_image_dim, image_rotation=local_image_rotation)
        elif task_name == 'element_merge_detect':
            text_list_1,text_list_2 = task_args
            query = build_element_merge_detect_query(args, text_list_1, text_list_2)
        elif task_name == 'html_table_merge':
            table_1,table_2 = task_args
            query = build_html_table_merge_query(args, table_1, table_2)
        query["temperature"] = TEMPERATURE_BY_ATTEMPT[
            min(attempt, len(TEMPERATURE_BY_ATTEMPT) - 1)
        ]  # Change temperature as number of attempts increases to overcome repetition issues at expense of quality

        try:
            status_code, response_body = await apost(COMPLETION_URL, json_data=query)

            if status_code == 400:
                raise ValueError(f"Got BadRequestError from server: {response_body}, skipping this response")
            elif status_code == 500:
                raise ValueError(f"Got InternalServerError from server: {response_body}, skipping this response")
            elif status_code != 200:
                raise ValueError(f"Error http status {status_code}")

            base_response_data = json.loads(response_body)

            metrics.add_metrics(
                vllm_input_tokens=base_response_data["usage"].get("prompt_tokens", 0),
                vllm_output_tokens=base_response_data["usage"].get("completion_tokens", 0),
            )

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
                return_data = "\n\n".join(markdown_element_list)

            elif task_name == 'element_merge_detect':
                return_data = eval(response_content)
            elif task_name == 'html_table_merge':
                if not (response_content.startswith("<table>") and response_content.endswith("</table>")):
                    raise ValueError("Response is not a table")
                return_data = response_content
            else:
                raise ValueError(f"Unknown task_name {task_name}")

            await tracker.track_work(worker_id, f"{worker_id}", "finished")
            return return_data

        except (ConnectionError, OSError, asyncio.TimeoutError) as e:
            logger.warning(f"Client error on attempt {attempt} for {worker_id}: {type(e)} {e}")

            # Now we want to do exponential backoff, and not count this as an actual page retry
            # Page retrys are supposed to be for fixing bad results from the model, but actual requests to vllm
            # are supposed to work. Probably this means that the server is just restarting
            sleep_delay = 10 * (2**exponential_backoffs)
            exponential_backoffs += 1
            logger.info(f"Sleeping for {sleep_delay} seconds on {worker_id} to allow server restart")
            await asyncio.sleep(sleep_delay)
        except asyncio.CancelledError:
            logger.info(f"Process {worker_id} cancelled")
            await tracker.track_work(worker_id, f"{worker_id}", "cancelled")
            raise
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error on attempt {attempt} for {worker_id}: {e}")
            attempt += 1
        except ValueError as e:
            logger.warning(f"ValueError on attempt {attempt} for {worker_id}: {type(e)} - {e}")
            attempt += 1
        except Exception as e:
            logger.exception(f"Unexpected error on attempt {attempt} for {worker_id}: {type(e)} - {e}")
            attempt += 1

    logger.error(f"Failed to process {worker_id} after {MAX_RETRIES} attempts.")
    await tracker.track_work(worker_id, f"{worker_id}", "errored")

    return None

def postprocess_markdown_text(args, response_text, pdf_path, page_number):
    text_list = response_text.split("\n\n")
    new_text_list = []
    for text in text_list:
        if text.startswith("<Image>") and text.endswith("</Image>"):
            pass
        else:
            new_text_list.append(text)
    return "\n\n".join(new_text_list)

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

async def process_pdf(args, worker_id: int, pdf_path: str):
    logger.info(f"Start process_pdf for {pdf_path}")
    if pdf_path.lower().endswith(".pdf"):
        try:
            reader = PdfReader(pdf_path)
            num_pages = reader.get_num_pages()
        except:
            logger.exception(f"Could not count number of pages for {pdf_path}, aborting document")
            return None
    else:
        num_pages = 1

    logger.info(f"Got {num_pages} pages to do for {pdf_path} in worker {worker_id}")

    try:
        tasks = []
        results = []
        async with asyncio.TaskGroup() as tg:
            for page_num in range(1, num_pages + 1):
                task = tg.create_task(process_task(args, worker_id, task_name='page_to_markdown', task_args=(pdf_path,page_num)))
                tasks.append(task)

        results = [task.result() for task in tasks]

        fallback_pages = []
        page_to_markdown_result = {}
        page_pairs = []
        for i,result in enumerate(results):
            if result != None:
                page_number = i+1
                page_to_markdown_result[i+1] = postprocess_markdown_text(args,result,pdf_path,page_number).split("\n\n")
                if page_number-1 in page_to_markdown_result.keys():
                    page_pairs.append((page_number-1,page_number))
            else:
                fallback_pages.append(i)

        num_fallback_pages = len(fallback_pages)

        if num_fallback_pages / num_pages > args.max_page_error_rate:
            logger.error(
                f"Document {pdf_path} has {num_fallback_pages} fallback pages out of {num_pages} exceeding max_page_error_rate of {args.max_page_error_rate}, discarding document."
            )
            return None
        elif num_fallback_pages > 0:
            logger.warning(
                f"Document {pdf_path} processed with {num_fallback_pages} fallback pages out of {num_pages}."
            )

        if args.skip_cross_page_merge:
            page_texts = {}
            document_text_list = []
            sorted_page_keys = sorted(list(page_to_markdown_result.keys()))
            for page_number in sorted_page_keys:
                page_texts[str(page_number-1)] = "\n\n".join(page_to_markdown_result[page_number])
                document_text_list.append(page_texts[str(page_number-1)])
            document_text = "\n\n".join(document_text_list)
            return {
                "orig_path": pdf_path,
                "num_pages": num_pages,
                "document_text": document_text,
                "page_texts": page_texts,
                "fallback_pages": fallback_pages,
            }

        tasks = []
        results = []
        async with asyncio.TaskGroup() as tg:
            for page_1,page_2 in page_pairs:
                task = tg.create_task(process_task(args, worker_id, task_name='element_merge_detect', task_args=(page_to_markdown_result[page_1], page_to_markdown_result[page_2])))
                tasks.append(task)
        results = [task.result() for task in tasks]

        element_merge_detect_result = {}
        table_pairs = []
        for page_pair,result in zip(page_pairs,results):
            if result != None:
                page_1,page_2 = page_pair
                element_merge_detect_result[(page_1,page_2)] = result
                for elem_idx_1,elem_idx_2 in result:
                    text_1 = page_to_markdown_result[page_1][elem_idx_1]
                    text_2 = page_to_markdown_result[page_2][elem_idx_2]
                    if text_1.startswith("<table>") and text_1.endswith("</table>") and text_2.startswith("<table>") and text_2.endswith("</table>"):
                        table_pairs.append((page_1,page_2,elem_idx_1,elem_idx_2))

        tmp_page_to_markdown_result = copy.deepcopy(page_to_markdown_result)
        table_pairs = sorted(table_pairs,key=lambda x: -x[0])
        html_table_merge_result = {}
        i = 0
        while i < len(table_pairs):
            async with asyncio.TaskGroup() as tg:
                tasks = []
                ids_1 = []
                ids_2 = []
                page_1,page_2,elem_idx_1,elem_idx_2 = table_pairs[i]
                task = tg.create_task(process_task(args, worker_id, task_name='html_table_merge', task_args=(tmp_page_to_markdown_result[page_1][elem_idx_1], tmp_page_to_markdown_result[page_2][elem_idx_2])))
                tasks.append(task)
                ids_1.append((page_1,elem_idx_1))
                ids_2.append((page_2,elem_idx_2))
                j = i + 1
                while j < len(table_pairs):
                    page_1,page_2,elem_idx_1,elem_idx_2 = table_pairs[j]
                    if (page_2, elem_idx_2) not in ids_1:
                        task = tg.create_task(process_task(args, worker_id, task_name='html_table_merge', task_args=(tmp_page_to_markdown_result[page_1][elem_idx_1], tmp_page_to_markdown_result[page_2][elem_idx_2])))
                        tasks.append(task)
                        ids_1.append((page_1,elem_idx_1))
                        ids_2.append((page_2,elem_idx_2))
                        j = j + 1
                    else:
                        break

            results = [task.result() for task in tasks]

            for k,result in enumerate(results):
                page_1,elem_idx_1 = ids_1[k]
                page_2,elem_idx_2 = ids_2[k]
                if result != None:
                    html_table_merge_result[(page_1,page_2,elem_idx_1,elem_idx_2)] = result
                    tmp_page_to_markdown_result[page_1][elem_idx_1] = html_table_merge_result[(page_1,page_2,elem_idx_1,elem_idx_2)]
            i = j

        page_texts = {}
        for page_number in page_to_markdown_result.keys():
            page_texts[str(page_number-1)] = "\n\n".join(page_to_markdown_result[page_number])

        document_text = build_document_text(page_to_markdown_result, element_merge_detect_result, html_table_merge_result)

        return {
            "orig_path": pdf_path,
            "num_pages": num_pages,
            "document_text": document_text,
            "page_texts": page_texts,
            "fallback_pages": fallback_pages,
        }
    except Exception as e:
        # Check for ExceptionGroup with BrokenProcessPool
        if isinstance(e, ExceptionGroup):
            broken_pool, other = e.split(BrokenProcessPool)
            if broken_pool is not None:  # Found at least one BrokenProcessPool
                logger.critical("Encountered BrokenProcessPool, exiting process.")
                sys.exit(1)

        logger.exception(f"Exception in process_pdf for {pdf_path}: {e}")
        return None

async def process_json(args, worker_id: int, json_path: str):
    try:
        json_data = json.load(open(json_path,'r'))
    except:
        logger.exception(f"Could not load {json_path}, aborting document")
    try:
        if args.task == 'merge_pages':
            page_1 = json_data['page_1'].split("\n\n")
            page_2 = json_data['page_2'].split("\n\n")
            async with asyncio.TaskGroup() as tg:
                task = tg.create_task(process_task(args, worker_id, task_name='element_merge_detect', task_args=(page_1, page_2)))
            result = task.result()
            return {
                "orig_path": json_path,
                "merge_pairs": result
            }
        elif args.task == 'merge_tables':
            table_1 = json_data['table_1']
            table_2 = json_data['table_2']
            async with asyncio.TaskGroup() as tg:
                task = tg.create_task(process_task(args, worker_id, task_name='html_table_merge', task_args=(table_1, table_2)))
            result = task.result()
            return {
                "orig_path": json_path,
                "merged_tables": result
            }
        else:
            raise ValueError(f"Unknown task {args.task}")

    except Exception as e:
        # Check for ExceptionGroup with BrokenProcessPool
        if isinstance(e, ExceptionGroup):
            broken_pool, other = e.split(BrokenProcessPool)
            if broken_pool is not None:  # Found at least one BrokenProcessPool
                logger.critical("Encountered BrokenProcessPool, exiting process.")
                sys.exit(1)

        logger.exception(f"Exception in process_json for {json_path}: {e}")
        return None

async def worker(args, work_queue: WorkQueue, semaphore, worker_id):
    while True:
        # Wait until allowed to proceed
        await semaphore.acquire()

        work_item = await work_queue.get_work()

        if work_item is None:
            logger.info(f"Worker {worker_id} exiting due to empty queue")
            semaphore.release()
            break

        logger.info(f"Worker {worker_id} processing work item {work_item.hash}")
        await tracker.clear_work(worker_id)

        try:
            async with asyncio.TaskGroup() as tg:
                if args.task == 'pdf2markdown':
                    tasks = [tg.create_task(process_pdf(args, worker_id, pdf_path)) for pdf_path in work_item.work_paths]
                elif args.task == 'merge_pages' or args.task == 'merge_tables':
                    tasks = [tg.create_task(process_json(args, worker_id, json_path)) for json_path in work_item.work_paths]
                else:
                    raise ValueError(f"Unknown task {args.task}")

                logger.info(f"Created all tasks for {work_item.hash}")

            logger.info(f"Finished TaskGroup for worker on {work_item.hash}")

            results = []
            for task in tasks:
                try:
                    result = task.result()
                except:
                    pass

                if result is not None:
                    results.append(result)

            logger.info(f"Got {len(results)} docs for {work_item.hash}")

            output_final_path = os.path.join(args.workspace, "results", f"output_{work_item.hash}.jsonl")
            with open(output_final_path, "w") as f:
                for result in results:
                    f.write(json.dumps(result))
                    f.write("\n")

            await work_queue.mark_done(work_item)
        except Exception as e:
            logger.exception(f"Exception occurred while processing work_hash {work_item.hash}: {e}")
        finally:
            semaphore.release()

async def vllm_server_task(args, semaphore):
    model_name_or_path = args.model

    cmd = [
        "vllm",
        "serve",
         model_name_or_path,
        "--port",
        str(args.port),
        "--max-model-len",
        str(args.model_max_context),
        "--gpu_memory_utilization",
        str(args.gpu_memory_utilization),
        "--tensor_parallel_size",
        str(args.tensor_parallel_size),
        "--dtype",
        str(args.dtype)
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # Ensure the subprocess is terminated on exit
    def _kill_proc():
        proc.terminate()

    atexit.register(_kill_proc)

    # Shared variables between tasks
    last_running_req, last_queue_req = 0, 0
    server_printed_ready_message = False
    last_semaphore_release = time.time()

    async def process_line(line):
        nonlocal last_running_req, last_queue_req, last_semaphore_release, server_printed_ready_message
        vllm_logger.info(line)

        # if the server hasn't initialized yet, log all the lines to the main logger also, so that the user
        # can see any warnings/errors more easily
        if not server_printed_ready_message:
            logger.info(line)

        if "Detected errors during sampling" in line:
            logger.error("Cannot continue, sampling errors detected, model is probably corrupt")
            sys.exit(1)

        # TODO, need to trace down this issue in vllm itself, but it will otherwise cause the server to lock up
        if "IndexError: list index out of range" in line:
            logger.error("IndexError in model, restarting server")
            proc.terminate()

        if not server_printed_ready_message and "The server is fired up and ready to roll!" in line:
            server_printed_ready_message = True
            last_semaphore_release = time.time()

        match = re.search(r"Running: (\d+)", line)
        if match:
            last_running_req = int(match.group(1))

        match = re.search(r"(?:Waiting|Pending):\s*(\d+)", line)
        if match:
            last_queue_req = int(match.group(1))
            logger.info(f"vllm running req: {last_running_req} queue req: {last_queue_req}")

    async def read_stream(stream):
        while True:
            line = await stream.readline()
            if not line:
                break
            try:
                line = line.decode("utf-8").rstrip()
                await process_line(line)
            except Exception as ex:
                logger.warning(f"Got {ex} when reading log line from inference server, skipping")

    async def timeout_task():
        nonlocal last_running_req, last_queue_req, last_semaphore_release
        try:
            while True:
                await asyncio.sleep(1)
                if server_printed_ready_message and last_queue_req == 0 and time.time() - last_semaphore_release > 30 and semaphore.locked():
                    semaphore.release()
                    last_semaphore_release = time.time()
                    logger.info("Semaphore released, allowing a worker to proceed.")
        except asyncio.CancelledError:
            pass  # Clean up if the task is cancelled

    # Start tasks to read stdout, stderr, and handle timeout logic
    stdout_task = asyncio.create_task(read_stream(proc.stdout))
    stderr_task = asyncio.create_task(read_stream(proc.stderr))
    timeout_task = asyncio.create_task(timeout_task())

    try:
        await proc.wait()
    except asyncio.CancelledError:
        logger.info("Got cancellation request for VLLM server")
        proc.terminate()
        raise

    timeout_task.cancel()
    await asyncio.gather(stdout_task, stderr_task, timeout_task, return_exceptions=True)

async def vllm_server_host(args, semaphore):
    MAX_RETRIES = 5
    retry = 0

    while retry < MAX_RETRIES:
        await vllm_server_task(args, semaphore)
        logger.warning("VLLM server task ended")
        retry += 1

    if retry >= MAX_RETRIES:
        logger.error(f"Ended up starting the vllm server more than {retry} times, cancelling pipeline")
        logger.error("")
        logger.error("Please make sure vllm is installed according to the latest instructions here: https://docs.vllm.ai/start/install.html")
        sys.exit(1)

async def vllm_server_ready(args):
    max_attempts = 300
    delay_sec = 1
    url = f"http://localhost:{args.port}/v1/models"

    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient() as session:
                response = await session.get(url)

                if response.status_code == 200:
                    logger.info("vllm server is ready.")
                    return
                else:
                    logger.info(f"Attempt {attempt}: Unexpected status code {response.status_code}")
        except Exception:
            logger.warning(f"Attempt {attempt}: Please wait for vllm server to become ready...")

        await asyncio.sleep(delay_sec)

    raise Exception("vllm server did not become ready after waiting.")

async def download_model(model_name_or_path: str):
    if os.path.isabs(model_name_or_path) and os.path.isdir(model_name_or_path):
        logger.info(f"Using local model path at '{model_name_or_path}'")
    else:
        logger.info(f"Downloading model with hugging face '{model_name_or_path}'")
        snapshot_download(repo_id=model_name_or_path)

async def metrics_reporter(work_queue):
    while True:
        # Leading newlines preserve table formatting in logs
        logger.info(f"Queue remaining: {work_queue.size}")
        logger.info("\n" + str(metrics))
        logger.info("\n" + str(await tracker.get_status_table()))
        await asyncio.sleep(10)

async def main():
    parser = argparse.ArgumentParser(description="Manager for running millions of PDFs through a batch inference pipeline")
    parser.add_argument(
        "workspace",
        help="The filesystem path where work will be stored, can be a local folder",
    )

    parser.add_argument("--task", type=str, choices=['pdf2markdown','merge_pages','merge_tables'], default='pdf2markdown', help="task names, could be 'pdf2markdown', 'merge_pages' or 'merge_tables'")

    parser.add_argument(
        "--data",
        nargs="*",
        help="List of paths to files to process",
        default=None,
    )

    parser.add_argument("--pages_per_group", type=int, default=500, help="Aiming for this many pdf pages per work item group")
    parser.add_argument("--max_page_retries", type=int, default=8, help="Max number of times we will retry rendering a page")
    parser.add_argument("--max_page_error_rate", type=float, default=0.004, help="Rate of allowable failed pages in a document, 1/250 by default")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="Fraction of GPU memory to use, default is 0.8")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of tensor parallel replicas")
    parser.add_argument("--dtype", type=str, choices=['auto','half','float16', 'float', 'bfloat16', 'float32'], default="auto", help="Data type for model weights and activations.")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers to run at a time")

    # Model parameters
    parser.add_argument(
        "--model",
        help="The path to the model",
        default="ChatDOC/OCRFlux-3B",
    )
    parser.add_argument("--model_max_context", type=int, default=16384, help="Maximum context length that the model was fine tuned under")
    parser.add_argument("--model_chat_template", type=str, default="qwen2-vl", help="Chat template to pass to vllm server")
    parser.add_argument("--target_longest_image_dim", type=int, help="Dimension on longest side to use for rendering the pdf pages", default=1024)

    parser.add_argument("--skip_cross_page_merge", action="store_true", help="Whether to skip cross-page merging")

    parser.add_argument("--port", type=int, default=40078, help="Port to use for the VLLM server")
    args = parser.parse_args()

    if os.path.exists(args.workspace):
        shutil.rmtree(args.workspace)

    # We need poppler to load the initial pdfs, even if we are not processing them here
    check_poppler_version()

    work_queue = LocalWorkQueue(args.workspace)

    if args.task == 'pdf2markdown':
        pdf_work_paths = set()

        for pdf_path in args.data:
            if os.path.exists(pdf_path):
                if pdf_path.lower().endswith(".pdf") and open(pdf_path, "rb").read(4) == b"%PDF":
                    logger.info(f"Loading file at {pdf_path} as PDF document")
                    pdf_work_paths.add(pdf_path)
                elif is_image(pdf_path):
                    logger.info(f"Loading file at {pdf_path} as image document")
                    pdf_work_paths.add(pdf_path)
                else:
                    raise ValueError(f"Unsupported file extension for {pdf_path}")
            else:
                raise ValueError(f"{pdf_path} does not exist")

        logger.info(f"Found {len(pdf_work_paths):,} total pdf paths to add")

        # Estimate average pages per pdf
        sample_size = min(100, len(pdf_work_paths))
        sampled_pdfs = random.sample(list(pdf_work_paths), sample_size)
        page_counts = []

        for pdf_path in tqdm(sampled_pdfs, desc="Sampling PDFs to calculate optimal length"):
            try:
                if pdf_path.lower().endswith(".pdf"):
                    reader = PdfReader(pdf_path)
                    page_counts.append(len(reader.pages))
                else:
                    page_counts.append(1)
            except Exception as e:
                logger.warning(f"Failed to read {pdf_path}: {e}")

        if page_counts:
            avg_pages_per_pdf = sum(page_counts) / len(page_counts)
        else:
            logger.warning("Could not read any PDFs to estimate average page count.")
            avg_pages_per_pdf = 10  # Default to 10 pages per PDF if sampling fails

        items_per_group = max(1, int(args.pages_per_group / avg_pages_per_pdf))
        logger.info(f"Calculated items_per_group: {items_per_group} based on average pages per PDF: {avg_pages_per_pdf:.2f}")

        # Now call populate_queue
        await work_queue.populate_queue(pdf_work_paths, items_per_group)
    elif args.task == 'merge_pages' or args.task == 'merge_tables':
        json_work_paths = set()
        for json_path in args.data:
            if os.path.exists(json_path):
                if json_path.lower().endswith(".json"):
                    json_work_paths.add(json_path)
                elif json_path.lower().endswith(".txt"):
                    logger.info(f"Loading file at {json_path} as list of paths")
                    with open(json_path, "r") as f:
                        json_work_paths |= set(filter(None, (line.strip() for line in f)))
                else:
                    raise ValueError(f"Unsupported file extension for {json_path}")
            else:
                raise ValueError(f"{json_path} does not exist")

        # Now call populate_queue
        await work_queue.populate_queue(json_work_paths, args.pages_per_group)


    # If you get this far, then you are doing inference and need a GPU
    check_vllm_version()
    check_torch_gpu_available()

    logger.info(f"Starting pipeline with PID {os.getpid()}")

    # Download the model before you do anything else
    await download_model(args.model)

    # Initialize the work queue
    qsize = await work_queue.initialize_queue()

    if qsize == 0:
        logger.info("No work to do, exiting")
        return
    # Create a semaphore to control worker access
    # We only allow one worker to move forward with requests, until the server has no more requests in its queue
    # This lets us get full utilization by having many workers, but also to be outputting dolma docs as soon as possible
    # As soon as one worker is no longer saturating the gpu, the next one can start sending requests
    semaphore = asyncio.Semaphore(1)

    vllm_server = asyncio.create_task(vllm_server_host(args, semaphore))

    await vllm_server_ready(args)

    metrics_task = asyncio.create_task(metrics_reporter(work_queue))

    # Create worker tasks to process the queue concurrently.
    worker_tasks = []
    for i in range(args.workers):
        task = asyncio.create_task(worker(args, work_queue, semaphore, worker_id=i))
        worker_tasks.append(task)

    # Wait for all worker tasks to finish
    await asyncio.gather(*worker_tasks)

    vllm_server.cancel()
    metrics_task.cancel()
    logger.info("Work done")


if __name__ == "__main__":
    asyncio.run(main())
