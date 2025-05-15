import runpod
import json
from main import process_pdf


def handler(job):
    # Extract input and optional output from the RunPod job
    job_input = job.get("input")
    if isinstance(job_input, dict):
        if "inp_data" in job_input:
            inp_data = job_input["inp_data"]
        elif "input" in job_input:
            inp_data = job_input["input"]
        else:
            return {"error": "Missing 'inp_data' or 'input' in job input"}
        out_data = job_input.get("out_data")
    else:
        inp_data = job_input
        out_data = None

    try:
        result_str = process_pdf(inp_data, out_data)
    except Exception as e:
        return {"error": str(e)}

    # Try to parse JSON string into Python object
    try:
        return json.loads(result_str)
    except Exception:
        return result_str


# Start the RunPod serverless worker with this handler
if __name__ == '__main__':
    runpod.serverless.start({"handler": handler})