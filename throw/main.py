import os
import logging
import subprocess
import tempfile
import json
import glob
import base64
import binascii

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_base64(s: str) -> bool:

    stripped = ''.join(s.split())

    # Base64 strings must have a length that’s a multiple of 4
    if len(stripped) % 4:
        return False

    try:
        # Try to decode; validate by round-trip re-encoding
        decoded = base64.b64decode(stripped, validate=True)
        reencoded = base64.b64encode(decoded).decode('ascii')
        return reencoded == stripped
    except (ValueError, binascii.Error):
        return False


def process_pdf(inp_data, out_data=None) -> str:
    """
    Process PDF data using olmOCR and return the JSON results.
    
    Args:
        inp_data: Base64-encoded PDF string OR S3 path
        out_data: None OR S3 path
        
    Returns:
        str: The extracted text as JSON string
    """
    logger.info("Processing PDF data")

    # Determine input/output types
    is_s3_input = inp_data.startswith("s3://")
    is_base64_input = is_base64(inp_data)
    is_s3_output = out_data is not None

    # Validate parameters
    if not (is_s3_input or is_base64_input):
        raise ValueError("Input data must be a base64-encoded PDF string or an S3 path")
    if is_s3_output and not out_data.startswith("s3://"):
        raise ValueError("Output data must be an S3 path")

    # Asynchronous processing for S3 output
    if is_s3_output:
        workspace = out_data
        # Prepare PDF path (local file for base64, S3 path otherwise)
        if is_base64_input:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            try:
                tmp.write(base64.b64decode(inp_data))
                tmp.flush()
                pdf_path = tmp.name
            finally:
                tmp.close()
        else:
            pdf_path = inp_data

        cmd = ["python3", "-m", "olmocr.pipeline", workspace, "--pdfs", pdf_path]
        subprocess.run(cmd, check=True)
        return json.dumps({"status": "success", "s3_output": workspace})

    # Synchronous processing for local output
    with tempfile.TemporaryDirectory() as workspace:
        # Prepare PDF path (base64 decode or S3 path)
        if is_base64_input:
            try:
                pdf_bytes = base64.b64decode(inp_data)
            except Exception as e:
                raise ValueError(f"Failed to decode base64 PDF data: {e}")
            pdf_path = os.path.join(workspace, "input.pdf")
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
            logger.info("PDF written to file")
        elif is_s3_input:
            pdf_path = inp_data
        else:
            raise ValueError("Input data must be a base64-encoded PDF string or an S3 path")

        logger.info("Running OCR pipeline")
        cmd = ["python3", "-m", "olmocr.pipeline", workspace, "--pdfs", pdf_path]
        subprocess.run(cmd, check=True)

        # Collect and return results
        results = []
        for result_file in glob.glob(os.path.join(workspace, "results", "output_*.jsonl")):
            with open(result_file, "r") as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))

        logger.info("Results read")
        return json.dumps(results)
