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

# Install Python dependencies including runpod and GPU extras
COPY . .

# Install Miniconda
RUN mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm ~/miniconda3/miniconda.sh

# Add conda to path and initialize for bash
RUN ~/miniconda3/bin/conda init bash && \
    . ~/.bashrc && \
    ~/miniconda3/bin/conda create -n olmocr python=3.11 -y

# Set up shell to use conda environment for subsequent commands
SHELL ["/bin/bash", "--login", "-c"]

# Use conda environment for all subsequent RUN commands
RUN echo "conda activate olmocr" >> ~/.bashrc

# Install Python dependencies in the conda environment
RUN pip install .[gpu] --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/ && \
    pip install runpod

# Start the RunPod serverless worker
CMD ["bash", "--login", "-c", "~/miniconda3/bin/conda activate olmocr && python main_runpod.py"]