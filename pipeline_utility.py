import asyncio
import atexit
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from functools import cache
from urllib.parse import urlparse
import httpx
import datetime
import hashlib

from olmocr.filter.filter import Language, PdfFilter
from olmocr.prompts import PageResponse
from olmocr.version import VERSION


# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

sglang_logger = logging.getLogger("sglang")
sglang_logger.propagate = False

file_handler = logging.FileHandler("olmocr-pipeline-debug.log", mode="a")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
sglang_logger.addHandler(file_handler)

# Filter object, cached so it will only get loaded when/if you need it
get_pdf_filter = cache(lambda: PdfFilter(languages_to_keep={Language.ENGLISH, None}, apply_download_spam_check=True, apply_form_check=True))

# Specify a default port, but it can be overridden by args
SGLANG_SERVER_PORT = 30024


@dataclass(frozen=True)
class PageResult:
    s3_path: str
    page_num: int
    response: PageResponse

    input_tokens: int
    output_tokens: int
    is_fallback: bool


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


async def sglang_server_task(args, semaphore):
    model_name_or_path = args.model
    import torch

    # if "://" in model_name_or_path:
    #     # TODO, Fix this code so that we support the multiple s3/weka paths, or else remove it
    #     model_cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'olmocr', 'model')
    #     download_directory(model_name_or_path, model_cache_dir)

    #     # Check the rope config and make sure it's got the proper key
    #     with open(os.path.join(model_cache_dir, "config.json"), "r") as cfin:
    #         config_data = json.load(cfin)

    #     if "rope_type" in config_data["rope_scaling"]:
    #         del config_data["rope_scaling"]["rope_type"]
    #         config_data["rope_scaling"]["type"] = "mrope"

    #         with open(os.path.join(model_cache_dir, "config.json"), "w") as cfout:
    #             json.dump(config_data, cfout)

    # Check GPU memory, lower mem devices need a bit less KV cache space because the VLM takes additional memory
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
    mem_fraction_arg = ["--mem-fraction-static", "0.80"] if gpu_memory < 60 else []

    cmd = [
        "python3",#"/root/miniconda3/envs/olmocr/bin/python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_name_or_path,
        "--chat-template",
        args.model_chat_template,
        # "--context-length", str(args.model_max_context),  # Commented out due to crashes
        "--port",
        str(SGLANG_SERVER_PORT),
        "--log-level-http",
        "warning",
    ]
    cmd.extend(mem_fraction_arg)

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
        sglang_logger.info(line)

        # if the server hasn't initialized yet, log all the lines to the main logger also, so that the user
        # can see any warnings/errors more easily
        if not server_printed_ready_message:
            logger.info(line)

        if "Detected errors during sampling" in line:
            logger.error("Cannot continue, sampling errors detected, model is probably corrupt")
            sys.exit(1)

        # TODO, need to trace down this issue in sglang itself, but it will otherwise cause the server to lock up
        if "IndexError: list index out of range" in line:
            logger.error("IndexError in model, restarting server")
            proc.terminate()

        if not server_printed_ready_message and "The server is fired up and ready to roll!" in line:
            server_printed_ready_message = True
            last_semaphore_release = time.time()

        match = re.search(r"#running-req: (\d+)", line)
        if match:
            last_running_req = int(match.group(1))

        match = re.search(r"#queue-req: (\d+)", line)
        if match:
            last_queue_req = int(match.group(1))
            logger.info(f"sglang running req: {last_running_req} queue req: {last_queue_req}")

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
        logger.info("Got cancellation request for SGLang server")
        proc.terminate()
        raise

    timeout_task.cancel()
    await asyncio.gather(stdout_task, stderr_task, timeout_task, return_exceptions=True)


async def sglang_server_host(args, semaphore):
    MAX_RETRIES = 5
    retry = 0

    while retry < MAX_RETRIES:
        await sglang_server_task(args, semaphore)
        logger.warning("SGLang server task ended")
        retry += 1

    if retry >= MAX_RETRIES:
        logger.error(f"Ended up starting the sglang server more than {retry} times, cancelling pipeline")
        logger.error("")
        logger.error("Please make sure sglang is installed according to the latest instructions here: https://docs.sglang.ai/start/install.html")
        sys.exit(1)


async def sglang_server_ready():
    max_attempts = 300
    delay_sec = 1
    url = f"http://localhost:{SGLANG_SERVER_PORT}/v1/models"

    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient() as session:
                response = await session.get(url)

                if response.status_code == 200:
                    logger.info("sglang server is ready.")
                    return
                else:
                    logger.info(f"Attempt {attempt}: Unexpected status code {response.status_code}")
        except Exception:
            logger.warning(f"Attempt {attempt}: Please wait for sglang server to become ready...")

        await asyncio.sleep(delay_sec)

    raise Exception("sglang server did not become ready after waiting.")


async def download_model(model_name_or_path: str):
    logger.info(f"Downloading model '{model_name_or_path}'")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=model_name_or_path)
    logger.info(f"Model download complete '{model_name_or_path}'")


def build_dolma_document(pdf_orig_path, page_results):
    # Build the document text and page spans
    document_text = ""
    pdf_page_spans = []
    current_char_pos = 0

    for index, page_result in enumerate(page_results):
        if page_result.response.natural_text is not None:
            content = page_result.response.natural_text + ("\n" if index < len(page_results) - 1 else "")
        else:
            content = ""

        start_pos = current_char_pos
        document_text += content
        current_char_pos = len(document_text)
        pdf_page_spans.append([start_pos, current_char_pos, page_result.page_num])

    if not document_text:
        logger.info(f"No document text for {pdf_orig_path}")
        return None  # Return None if the document text is empty

    # Build the Dolma document
    metadata = {
        "Source-File": pdf_orig_path,
        "olmocr-version": VERSION,
        "pdf-total-pages": len(page_results),
        "total-input-tokens": sum(page.input_tokens for page in page_results),
        "total-output-tokens": sum(page.output_tokens for page in page_results),
        "total-fallback-pages": sum(page.is_fallback for page in page_results),
    }

    id_ = hashlib.sha1(document_text.encode()).hexdigest()

    dolma_doc = {
        "id": id_,
        "text": document_text,
        "source": "olmocr",
        "added": datetime.datetime.now().strftime("%Y-%m-%d"),
        "created": datetime.datetime.now().strftime("%Y-%m-%d"),
        "metadata": metadata,
        "attributes": {"pdf_page_numbers": pdf_page_spans},
    }
    return dolma_doc
