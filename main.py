import os
import logging
import subprocess
import tempfile
import json
import glob
import base64
from typing import Union, List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_pdf(pdf_data) -> str:
    """
    Process PDF data using olmOCR and return the JSON results.
    
    Args:
        pdf_data: Base64-encoded PDF string
        
    Returns:
        str: The extracted text as JSON string
    """
    logger.info(f"Processing PDF data")

    # Create temporary workspace
    with tempfile.TemporaryDirectory() as workspace:
        # Decode base64 PDF
        try:
            pdf_bytes = base64.b64decode(pdf_data)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 PDF data: {str(e)}")
        
        logger.info(f"PDF bytes decoded")
        
        # Write PDF to file
        pdf_path = os.path.join(workspace, "input.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        logger.info(f"PDF written to file")
        
        # Build the command
        cmd = ["python", "-m", "olmocr.pipeline", workspace, "--pdfs", pdf_path]
        
        # Run the command
        subprocess.run(cmd, check=True)
        
        # Find and read the results
        results = []
        for result_file in glob.glob(os.path.join(workspace, "results", "output_*.jsonl")):
            with open(result_file, "r") as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))

        logger.info(f"Results read")
        
        # Return results as JSON string
        return json.dumps(results)
