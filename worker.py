import logging, sys, os, json, base64, asyncio, tempfile, boto3, multiprocessing, runpod # httpx,
from io import BytesIO
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from PIL import Image
from dotenv import load_dotenv

from olmocr.check import check_sglang_version, check_torch_gpu_available
from olmocr.prompts import PageResponse, build_finetuning_prompt
from olmocr.prompts.anchor import PageReport, _linearize_pdf_report
from olmocr.s3_utils import parse_s3_path
from pipeline_utility import PageResult, download_model, sglang_server_ready, sglang_server_host, apost, build_dolma_document

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
MAX_RETRIES = 8
TEMPERATURE_BY_ATTEMPT = [0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
TARGET_LONGEST_IMAGE_DIM = 1024
TARGET_ANCHOR_TEXT_LEN = 6000
MAX_PAGE_ERROR_RATE = 0.004

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

async def load_page_data(s3_path: str) -> dict:
    assert s3_path.startswith("s3://")
    bucket, key = parse_s3_path(s3_path)
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode()


def rotate_image(img_url: str, rotation: int) -> bytes:
    assert rotation in [0, 90, 180, 270], "Invalid image rotation provided in rotate_image"
    image_base64 = img_url.split("data:image/png;base64,")[1]
    image_bytes = base64.b64decode(image_base64)
    with Image.open(BytesIO(image_bytes)) as img:
        rotated_img = img.rotate(-rotation, expand=True)
        # Save the rotated image to a bytes buffer
        buffered = BytesIO()
        rotated_img.save(buffered, format="PNG")
    # Encode the rotated image back to base64
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"


def report_to_natural_text(report: PageReport) -> str:
    out = ""
    for el in report.text_elements:
        out += el['text']
    return out

# ---------------------------------------------------------------------------
# Page processing
# ---------------------------------------------------------------------------

async def process_page(page_query_path: str) -> PageResult:
    exponential_backoffs = 0
    local_anchor_text_len = TARGET_ANCHOR_TEXT_LEN
    local_image_rotation = 0
    attempt = 0

    page_num = int(page_query_path.split("_")[-1].replace('.json', ''))
    query_data = await load_page_data(page_query_path)
    query = query_data["query"]
    page_report = PageReport(**query_data["page_report"])

    logger.info(f"Processing page query: {page_query_path}")

    while attempt < MAX_RETRIES:
        
        query["temperature"] = TEMPERATURE_BY_ATTEMPT[
            min(attempt, len(TEMPERATURE_BY_ATTEMPT) - 1)
        ]  # Change temperature as number of attempts increases to overcome repetition issues at expense of quality
        if local_image_rotation != 0:
            query["messages"][0]["content"][1]["image_url"]["url"] = rotate_image(query["messages"][0]["content"][1]["image_url"]["url"], local_image_rotation)

        logger.info(f"Retrieved page query for {page_query_path}")

        try:
            status_code, response_body = await apost(SGLANG_URL, json_data=query)

            if status_code == 400:
                raise ValueError(f"Got BadRequestError from server: {response_body}, skipping this response")
            elif status_code == 500:
                raise ValueError(f"Got InternalServerError from server: {response_body}, skipping this response")
            elif status_code != 200:
                raise ValueError(f"Error http status {status_code}")

            base_response_data = json.loads(response_body)

            if base_response_data["usage"]["total_tokens"] > MAX_CONTEXT:
                local_anchor_text_len = max(1, local_anchor_text_len // 2)
                logger.info(f"Reducing anchor text len to {local_anchor_text_len} for {page_query_path}")
                with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as process_pool:
                    loop = asyncio.get_running_loop()
                    anchor_text = await loop.run_in_executor(
                        process_pool, _linearize_pdf_report, page_report, local_anchor_text_len
                    )
                query["messages"][0]["content"][0]["text"] = build_finetuning_prompt(anchor_text)
                raise ValueError("Response exceeded model_max_context, cannot use this response")

            logger.info(
                f"sglang_input_tokens={base_response_data['usage'].get('prompt_tokens', 0)}, sglang_output_tokens={base_response_data['usage'].get('completion_tokens', 0)}"
            )

            model_response_json = json.loads(base_response_data["choices"][0]["message"]["content"])
            page_response = PageResponse(**model_response_json)

            if not page_response.is_rotation_valid and attempt < MAX_RETRIES - 1:
                logger.info(
                    f"Got invalid_page rotation for {page_query_path} attempt {attempt}, retrying with {page_response.rotation_correction} rotation"
                )
                local_image_rotation = page_response.rotation_correction
                raise ValueError(f"invalid_page rotation for {page_query_path}")
            
            return PageResult(
                page_query_path,
                page_num,
                page_response,
                input_tokens=base_response_data["usage"].get("prompt_tokens", 0),
                output_tokens=base_response_data["usage"].get("completion_tokens", 0),
                is_fallback=False,
            )
        except (ConnectionError, OSError, asyncio.TimeoutError) as e:
            logger.warning(f"Client error on attempt {attempt} for {page_query_path}: {type(e)} {e}")

            # Now we want to do exponential backoff, and not count this as an actual page retry
            # Page retrys are supposed to be for fixing bad results from the model, but actual requests to sglang
            # are supposed to work. Probably this means that the server is just restarting
            sleep_delay = 10 * (2**exponential_backoffs)
            exponential_backoffs += 1
            logger.info(f"Sleeping for {sleep_delay} seconds on {page_query_path} to allow server restart")
            await asyncio.sleep(sleep_delay)
        except asyncio.CancelledError:
            logger.info(f"Process page {page_query_path} cancelled")
            raise
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error on attempt {attempt} for {page_query_path}: {e}")
            attempt += 1
        except ValueError as e:
            logger.warning(f"ValueError on attempt {attempt} for {page_query_path}: {type(e)} - {e}")
            attempt += 1
        except Exception as e:
            logger.exception(f"Unexpected error on attempt {attempt} for {page_query_path}: {type(e)} - {e}")
            attempt += 1

    logger.error(f"Failed to process {page_query_path} after {MAX_RETRIES} attempts.")

    return PageResult(
        page_query_path,
        page_num,
        PageResponse(
            natural_text=report_to_natural_text(page_report),
            primary_language=None,
            is_rotation_valid=True,
            rotation_correction=0,
            is_table=False,
            is_diagram=False,
        ),
        input_tokens=0,
        output_tokens=0,
        is_fallback=True,
    )


async def process_doc(job_id: str, doc: dict):

    doc_id = doc["doc_id"]
    num_pages = doc["pages"]
    page_queries = doc["page_queries"]

    # List to hold the tasks for processing each page
    page_tasks = []
    page_results = []

    try:
        async with asyncio.TaskGroup() as tg:
            for page_num in range(1, num_pages + 1):
                task = tg.create_task(process_page(page_queries[page_num - 1]))
                page_tasks.append(task)

        # Collect the results from the entire task group, assuming no exceptions
        page_results = [task.result() for task in page_tasks]

        num_fallback_pages = sum(page_result.is_fallback for page_result in page_results)

        if num_fallback_pages / num_pages > MAX_PAGE_ERROR_RATE:
            logger.error(
                f"Document {doc_id} has {num_fallback_pages} fallback pages out of {num_pages} exceeding max_page_error_rate of {MAX_PAGE_ERROR_RATE}, discarding document."
            )
            return None
        elif num_fallback_pages > 0:
            logger.warning(
                f"Document {doc_id} processed with {num_fallback_pages} fallback pages out of {num_pages}, proceeding to build Dolma document."
            )

        return build_dolma_document(doc_id, page_results)
    
    except Exception as e:
        # Check for ExceptionGroup with BrokenProcessPool
        if isinstance(e, ExceptionGroup):
            broken_pool, other = e.split(BrokenProcessPool)
            if broken_pool is not None:  # Found at least one BrokenProcessPool
                logger.critical("Encountered BrokenProcessPool, exiting process.")
                sys.exit(1)

        logger.exception(f"Exception in process_doc for {doc_id}: {e}")
        # You can't build a dolma doc with even 1 failed page, so just get out of here
        # However, you don't want to propagate an exception higher up and cancel the entire work_group
        return None


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

    logger.info(f"Processing job {job_id}")

    semaphore = asyncio.Semaphore(1)
    
    try:
        async with asyncio.TaskGroup() as tg:
            doc_tasks = [tg.create_task(process_doc(job_id, doc)) for doc in manifest["documents"]]
            logger.info(f"Created all tasks for {job_id}")

        doc_results = []
        for task in doc_tasks:
            try:
                result = task.result()
            except:
                # some dolma doc creations may have failed
                pass

            if result is not None:
                doc_results.append(result)

        logger.info(f"Got {len(doc_results)} docs for {job_id}")
        
        # Write the Dolma documents to a local temporary file in JSONL format
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tf:
            for doc in doc_results:
                tf.write(json.dumps(doc))
                tf.write("\n")
            tf.flush()

            # Define the output S3 path using the work_hash
            output_final_path = os.path.join(S3_BUCKET, "jobs", job_id, "results", f"output_{job_id}.jsonl")
            s3.upload_file(tf.name, S3_BUCKET, output_final_path)

    except Exception as e:
        logger.exception(f"Exception occurred while processing job {job_id}: {e}")
    finally:
        semaphore.release()


    logger.info(f"Finished job {job_id}")
    return {"job_id": job_id, "status": "completed"}


asyncio.run(initialize())
runpod.serverless.start({
    "handler": handler,
    # "return_aggregate_stream": True
    })
