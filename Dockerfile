FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    poppler-utils \
    ttf-mscorefonts-installer \
    msttcorefonts \
    fonts-crosextra-caladea \
    fonts-crosextra-carlito \
    gsfonts \
    lcdf-typetools

WORKDIR /app

# Install Python dependencies including runpod and GPU extras
COPY requirements.txt pyproject.toml ./

RUN mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm ~/miniconda3/miniconda.sh

RUN pip install runpod

# Copy the rest of the application code
COPY . .

# Start the RunPod serverless worker
CMD ["python", "main_runpod.py"]