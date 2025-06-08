import logging, sys, os, json, base64, asyncio, tempfile, boto3, multiprocessing, runpod # httpx,
from io import BytesIO
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from PIL import Image
import torch
import time
from dotenv import load_dotenv
from botocore.config import Config

from olmocr.check import check_sglang_version, check_torch_gpu_available, check_poppler_version
from olmocr.prompts import PageResponse, build_finetuning_prompt
from olmocr.prompts.anchor import PageReport, _linearize_pdf_report
from olmocr.s3_utils import parse_s3_path
from olmocr.work_queue import S3WorkQueue
from pipeline_utility import PageResult, download_model, sglang_server_ready, sglang_server_host, apost, build_dolma_document, metrics_reporter, worker, process_pool


load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

S3_BUCKET = os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION")

S3_CONFIG = Config(max_pool_connections=30)
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION'),
    config=S3_CONFIG
)
N_WORKERS = 8

SGLANG_PORT = 30024
SGLANG_URL = f"http://127.0.0.1:{SGLANG_PORT}/v1/chat/completions"

MODEL = "allenai/olmOCR-7B-0225-preview"
MODEL_CHAT_TEMPLATE = "qwen2-vl"
NUM_PAGES_PER_GROUP = 500

# Flag to ensure initialize is run once within handler's event loop
initialized = False

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

async def initialize():

    # checks
    check_poppler_version()
    check_sglang_version()
    logger.info("CUDA_VISIBLE_DEVICES = %r", os.environ.get("CUDA_VISIBLE_DEVICES"))
    logger.info("torch.version.cuda    = %r", torch.version.cuda)
    logger.info("torch.cuda.is_available() = %r", torch.cuda.is_available())
    logger.info("torch.cuda.device_count() = %d", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info("  device %d: %s, %.2f GiB", i, props.name, props.total_memory / (1024**3))
    check_torch_gpu_available()

    # download model
    await download_model(MODEL)

    # limit sglang server to 1 concurrent request
    global semaphore
    semaphore = asyncio.Semaphore(1)
    
    # start sglang server
    global sglang_server
    args = SimpleNamespace(
        model=MODEL,
        model_chat_template=MODEL_CHAT_TEMPLATE
    )
    sglang_server = asyncio.create_task(sglang_server_host(args, semaphore))

    await sglang_server_ready()

# ---------------------------------------------------------------------------
# Webhook with retries
# ---------------------------------------------------------------------------

"""
async def post_webhook(url: str, payload: dict, max_attempts: int = 5):
    async with httpx.AsyncClient() as client:
        for attempt in range(max_attempts):
            try:
                await client.post(url, json=payload, timeout=10)
                return
            except Exception:
                await asyncio.sleep((2 ** attempt) + 0.1 * attempt)  # exp backoff + jitter
        print(f"WebhookFailed url={url} attempts={max_attempts}")
"""

# ---------------------------------------------------------------------------
# RunPod handler entry-point
# ---------------------------------------------------------------------------

async def handler(job):

    job_input = job["input"]
    job_id = job_input.get("id")  # job id
    manifest = job_input.get("manifest")

    s3_workspace = f"s3://{S3_BUCKET}/jobs/{job_id}"

     # Initialize the SGLang server and model once
    global initialized
    if not initialized:
        await initialize()
        initialized = True

    logger.info(f"Processing job {job_id}")

    page_counts = []
    for doc in manifest["documents"]:
        page_counts.append(doc["pages"])
    avg_pages_per_pdf = sum(page_counts) / len(page_counts)
    items_per_group = max(1, int(NUM_PAGES_PER_GROUP / avg_pages_per_pdf))

    logger.info(f"Found {len(manifest['documents']):,} documents with an average of {avg_pages_per_pdf:.2f} pages per document")

    work_queue = S3WorkQueue(s3, s3_workspace)
    await work_queue.populate_queue([doc['path'] for doc in manifest["documents"]], items_per_group)

    logger.info(f"Work queue populated")

    # Initialize the work queue
    qsize = await work_queue.initialize_queue()

    if qsize == 0:
        logger.info("No work to do, exiting")
        return
    
    metrics_task = asyncio.create_task(metrics_reporter(work_queue))

    # Create worker tasks to process the queue concurrently.
    worker_tasks = []
    for i in range(N_WORKERS):
        task = asyncio.create_task(worker(s3_workspace, work_queue, semaphore, worker_id=i))
        worker_tasks.append(task)

    # Wait for all worker tasks to finish
    await asyncio.gather(*worker_tasks)

    # Wait for server to stop
    process_pool.shutdown(wait=False)

    sglang_server.cancel()
    metrics_task.cancel()
    logger.info("Work done")

    logger.info(f"Finished job {job_id}")
    return {"job_id": job_id, "status": "completed"}


runpod.serverless.start({
    "handler": handler,
    # "return_aggregate_stream": True
    })
