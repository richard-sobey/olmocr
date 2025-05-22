import runpod, asyncio, os, json, httpx, boto3, logging
from typing import Tuple, List
from types import SimpleNamespace
from dotenv import load_dotenv

from olmocr.pipeline import check_sglang_version, check_torch_gpu_available, download_model, sglang_server_ready, sglang_server_host

load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

S3_BUCKET = os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION")

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

SGLANG_PORT = 30024
SGLANG_URL = f"http://127.0.0.1:{SGLANG_PORT}/v1/chat/completions"

MODEL = "allenai/olmOCR-7B-0225-preview"
MAX_CONTEXT = 8192
MODEL_CHAT_TEMPLATE = "qwen2-vl"
TARGET_LONGEST_IMAGE_DIM = 1024
TARGET_ANCHOR_TEXT_LEN = 6000

async def initialize():

    # checks
    check_sglang_version()
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


# asyncio.run(initialize())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def ocr_page(client: httpx.AsyncClient, img_key: str, anchor_key: str, page_idx: int) -> Tuple[int, str]:
    """Return (page_number, text)."""
    img_bytes = s3.get_object(Bucket=S3_BUCKET, Key=img_key)["Body"].read()
    try:
        anchor = s3.get_object(Bucket=S3_BUCKET, Key=anchor_key)["Body"].read().decode()
    except s3.exceptions.NoSuchKey:
        anchor = ""

    payload = (img_bytes, anchor)
    resp = await client.post(SGLANG_URL, json=payload, timeout=None)
    text = resp.json()["choices"][0]["message"]["content"].strip()
    return page_idx, text

# ---------------------------------------------------------------------------
# Per-document processing
# ---------------------------------------------------------------------------

async def process_document(client: httpx.AsyncClient, job_id: str, doc: dict):
    """Process all pages of a single document concurrently."""
    prefix_inside_bucket = doc["png_prefix"].split("/", 3)[-1]
    tasks = []
    for i in range(doc["pages"]):
        img_key = f"{job_id}/{prefix_inside_bucket}{i:04}.png"
        anchor_key = f"{job_id}/anchor/{doc['doc_id']}/page_{i:04}.txt"
        tasks.append(ocr_page(client, img_key, anchor_key, i + 1))

    results: List[Tuple[int, str]] = await asyncio.gather(*tasks)
    results.sort(key=lambda t: t[0])

    # write JSONL to S3
    key_out = f"{job_id}/results/{doc['doc_id']}.jsonl"
    lines = [json.dumps({"page": p, "text": t}, ensure_ascii=False) for p, t in results]
    s3.put_object(Bucket=S3_BUCKET, Key=key_out, Body="\n".join(lines).encode())

    return {
        "doc_id": doc["doc_id"],
        "pages": doc["pages"],
        "result_url": f"s3://{S3_BUCKET}/{key_out}",
        "status": "completed",
    }

# ---------------------------------------------------------------------------
# Webhook with retries
# ---------------------------------------------------------------------------

async def post_webhook(url: str, payload: dict, max_attempts: int = 5):
    async with httpx.AsyncClient() as client:
        for attempt in range(max_attempts):
            try:
                await client.post(url, json=payload, timeout=10)
                return
            except Exception:
                await asyncio.sleep((2 ** attempt) + 0.1 * attempt)  # exp backoff + jitter
        print(f"WebhookFailed url={url} attempts={max_attempts}")


# ---------------------------------------------------------------------------
# RunPod handler entry-point
# ---------------------------------------------------------------------------

async def handler(event):
    job_id = event["id"]  # RunPod job id
    manifest = event["input"]

    logger.info(f"Processing job {job_id}")
    asyncio.sleep(10)
    logger.info(f"Finished job {job_id}")
    return {"job_id": job_id, "status": "completed"}

    """
    async with httpx.AsyncClient(timeout=None) as client:
        doc_results = await asyncio.gather(*(process_document(client, job_id, d) for d in manifest["documents"]))

    response_payload = {"job_id": manifest["job_id"], "documents": doc_results}

    # optional webhook
    if manifest.get("webhook_url"):
        await post_webhook(manifest["webhook_url"], response_payload)

    return response_payload
    """


runpod.serverless.start({
    "handler": handler,
    # "return_aggregate_stream": True
    })
