FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    poppler-utils \
    ttf-mscorefonts-installer \
    msttcorefonts \
    fonts-crosextra-caladea \
    fonts-crosextra-carlito \
    gsfonts \
    lcdf-typetools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies including runpod and GPU extras
COPY requirements.txt .
RUN pip3 install --upgrade pip \
    && pip3 install runpod \
    && pip3 install -e .[gpu] --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/ \
    && pip3 install runpod

# Copy the rest of the application code
COPY . .

# Start the RunPod serverless worker
CMD ["python3", "-u", "main_runpod.py"]