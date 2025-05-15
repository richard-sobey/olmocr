import runpod, asyncio, os, json, httpx, boto3
from typing import Tuple, List
from prompt_utils import build_prompt

S3_BUCKET = os.getenv("OCR_S3_BUCKET", "ocr-pipeline-prod")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
SGLANG_URL = "http://127.0.0.1:30024/v1/chat/completions"

s3 = boto3.client("s3", region_name=AWS_REGION)

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

    payload = build_prompt(img_bytes, anchor)
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
# RunPod handler entry-point
# ---------------------------------------------------------------------------

async def handler(event):
    job_id = event["id"]  # RunPod job id
    manifest = event["input"]

    async with httpx.AsyncClient(timeout=None) as client:
        doc_results = await asyncio.gather(*(process_document(client, job_id, d) for d in manifest["documents"]))

    response_payload = {"job_id": manifest["job_id"], "documents": doc_results}

    # optional webhook
    if manifest.get("webhook_url"):
        await post_webhook(manifest["webhook_url"], response_payload)

    return response_payload

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

runpod.serverless.start({"handler": handler})
