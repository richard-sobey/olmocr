import logging
from pathlib import Path
import uuid, os, asyncio, tempfile, json
from pypdf import PdfReader
import uvicorn
import requests
from dataclasses import asdict
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
import runpod
import boto3

from auth import validate_api_key
from olmocr.pipeline import build_page_query
from olmocr.prompts.anchor import _pdf_report

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()

RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT_ID")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")
MAX_CONCURRENT_TASKS = 50

TARGET_LONGEST_IMAGE_DIM = 1024
TARGET_ANCHOR_TEXT_LEN = 6000

runpod.api_key = RUNPOD_API_KEY
endpoint = runpod.Endpoint(RUNPOD_ENDPOINT)

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

app = FastAPI(title="OCR Pipeline API")


@app.post("/ocr", status_code=202, dependencies=[Depends(validate_api_key)])
async def ocr(pdfs: list[UploadFile], webhook_url: str | None = None):
    
    manifest_id = uuid.uuid4().hex
    
    logger.info(f"Received OCR request for {len(pdfs)} documents")

    manifest = {
        "manifest_id": manifest_id,
        "documents": [],
        "output_format": "jsonl",
        "webhook_url": webhook_url,
    }

    # Create a semaphore to limit concurrency to 20 tasks
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    # temporary workspace for image conversion
    with tempfile.TemporaryDirectory() as td:
        for pdf_file in pdfs:
            data = await pdf_file.read()
            doc_id = Path(pdf_file.filename).stem or uuid.uuid4().hex
            
            # Save PDF to temp dir
            pdf_path = Path(td) / f"{doc_id}.pdf"
            with open(pdf_path, "wb") as f:
                f.write(data)
            
            # Get page count
            try:
                reader = PdfReader(pdf_path)
                page_count = reader.get_num_pages()
            except Exception as e:
                logger.error(f"Error counting pages for {pdf_path}: {e}")
                return JSONResponse({"error": f"Could not count number of pages for {pdf_path}, aborting document"}, status_code=500)
            
            # Process each page with build_page_query
            page_queries = []
            page_tasks = []
            
            async with asyncio.TaskGroup() as tg:
                for page_num in range(page_count):
                    # Create a task for processing each page in parallel
                    task = tg.create_task(
                        process_page(pdf_path, page_num + 1, manifest_id, doc_id, S3_BUCKET, semaphore)
                    )
                    page_tasks.append(task)
            
            # Collect results from all tasks
            page_queries = [task.result() for task in page_tasks]

            logger.info(f"Collected {len(page_queries)} page queries for {doc_id}")
            
            manifest["documents"].append({
                "doc_id": doc_id,
                "pages": page_count,
                "page_queries": [f"s3://{S3_BUCKET}/{key}" for key in page_queries],
            })

    # enqueue RunPod serverless job (non-blocking)
    logger.info(f"Enqueuing RunPod job for manifest: {manifest_id}")
    run_request = endpoint.run(manifest)
    logger.info(f"RunPod job enqueued with ID: {run_request.job_id}")
    status = run_request.status()
    return JSONResponse({"job_id": run_request.job_id, "manifest_id": manifest_id, "status": status})


@app.get("/status/{job_id}")
async def status(job_id: str):
    if not len(job_id):
        return JSONResponse({"error": "Job ID is required"}, status_code=400)
    headers = {
        'Authorization': f'Bearer {os.getenv("RUNPOD_API_KEY")}',
        'Content-Type': 'application/json'
    }
    url = f"https://api.runpod.ai/v2/{os.getenv('RUNPOD_ENDPOINT_ID')}/status/{job_id}"
    resp = requests.get(url, headers=headers)
    return JSONResponse(resp.json())


# Updated process_page function to use a semaphore
async def process_page(pdf_path, page_num, job_id, doc_id, bucket, semaphore):
    async with semaphore:  # Acquire semaphore before processing, release after
        
        # Build page query
        query = await build_page_query(
            str(pdf_path), 
            page_num, 
            TARGET_LONGEST_IMAGE_DIM, 
            TARGET_ANCHOR_TEXT_LEN
        )

        # separately get raw anchor text
        page_report = await asyncio.to_thread(_pdf_report, str(pdf_path), page_num)
        page_report_dict = asdict(page_report)

        # prepare payload
        payload = {
            "query": query,
            "page_report": page_report_dict
        }
        
        # Store the complete query result in S3
        query_key = f"jobs/{job_id}/queries/{doc_id}/page_{page_num:04}.json"
        s3.put_object(
            Bucket=bucket, 
            Key=query_key, 
            Body=json.dumps(payload).encode(),
            ContentType="application/json"
        )
        
        logger.info(f"Processed {doc_id} page {page_num}")
        return query_key
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
