import logging
from pathlib import Path
import uuid, os, asyncio, tempfile, json
from pypdf import PdfReader
import uvicorn
import requests
from dataclasses import asdict
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, StreamingResponse
import runpod
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from auth import validate_api_key
from olmocr.pipeline import build_page_query
from olmocr.prompts.anchor import _pdf_report
from olmocr.s3_utils import parse_s3_path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()

RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT_ID")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")
MAX_CONCURRENT_TASKS = 25

TARGET_LONGEST_IMAGE_DIM = 1024
TARGET_ANCHOR_TEXT_LEN = 6000

runpod.api_key = RUNPOD_API_KEY
endpoint = runpod.Endpoint(RUNPOD_ENDPOINT)

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION'),
    config=Config(max_pool_connections=50)
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

    # temporary workspace for image conversion
    with tempfile.TemporaryDirectory() as td:
        
        async def process_pdf(pdf_file: UploadFile) -> dict:
            """Process a single PDF file asynchronously"""
            data = await pdf_file.read()
            doc_id = Path(pdf_file.filename).stem or uuid.uuid4().hex
            
            """
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
                raise Exception(f"Could not count number of pages for {pdf_path}")
            
            """
            # Store the complete query result in S3 (run in thread pool to avoid blocking)
            query_key = f"jobs/{manifest_id}/{doc_id}.pdf"
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: s3.put_object(
                    Bucket=S3_BUCKET, 
                    Key=query_key, 
                    Body=data,
                    ContentType="application/pdf"
                )
            )
            
            # Clean up the temporary PDF file
            # pdf_path.unlink()
            
            return {
                "doc_id": doc_id,
                #"pages": page_count,
                "path": query_key,
            }
        
        # Process all PDFs in parallel
        try:
            results = await asyncio.gather(*[process_pdf(pdf_file) for pdf_file in pdfs])
            manifest["documents"].extend(results)
        except Exception as e:
            logger.error(f"Error processing PDFs: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    # enqueue RunPod serverless job (non-blocking)
    logger.info(f"Enqueuing RunPod job for manifest: {manifest_id}")

    return JSONResponse({
        "input": {
            "id": manifest_id,
            "manifest": manifest
        }
    })
    
    run_request = endpoint.run({
        "input": {
            "id": manifest_id,
            "manifest": manifest
        }
    })
    logger.info(f"RunPod job enqueued with ID: {run_request.job_id}")
    status = run_request.status()
    return JSONResponse({"job_id": run_request.job_id, "manifest_id": manifest_id, "status": status})


@app.get("/status/{job_id}", dependencies=[Depends(validate_api_key)])
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


@app.get("/results/{job_id}", dependencies=[Depends(validate_api_key)])
async def results(job_id: str):
    if not len(job_id):
        return JSONResponse({"error": "Job ID is required"}, status_code=400)
    results_path = f"s3://{S3_BUCKET}/jobs/{job_id}/results/output_{job_id}.jsonl"
    try:
        bucket, key = parse_s3_path(results_path)
        obj = s3.get_object(Bucket=bucket, Key=key)
        lines = obj['Body'].read().decode().splitlines()
        records = [json.loads(line) for line in lines]
        return JSONResponse(records)
        # # Stream the JSONL file directly from S3
        # body = obj["Body"]
        # return StreamingResponse(
        #     (line + "\n" for line in body.iter_lines(decode_unicode=True)),
        #     media_type="application/x-ndjson"
        # )
    except ClientError as e:
        # Handle missing key: job may still be processing or failed
        if e.response.get("Error", {}).get("Code") == "NoSuchKey":
            return JSONResponse(
                {"error": "Results not available. Job may still be processing or failed. Please check job status."},
                status_code=404
            )
        else:
            logger.error(f"Error getting results for {job_id}: {e}")
            return JSONResponse({"error": f"Could not get results for {job_id}"}, status_code=500)
    except Exception as e:
        logger.error(f"Error getting results for {job_id}: {e}")
        return JSONResponse({"error": f"Could not get results for {job_id}"}, status_code=500)
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
