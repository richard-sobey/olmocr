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
    wget

WORKDIR /app

# Copy application files
COPY . .

# Install Miniconda
RUN mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm ~/miniconda3/miniconda.sh

# Create conda environment
RUN ~/miniconda3/bin/conda create -n olmocr python=3.11 -y

# Install Python dependencies in the conda environment
RUN ~/miniconda3/envs/olmocr/bin/pip install .[gpu] --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/ && \
    ~/miniconda3/envs/olmocr/bin/pip install runpod

# Start the RunPod serverless worker
CMD ["/root/miniconda3/envs/olmocr/bin/python", "main_runpod.py"]