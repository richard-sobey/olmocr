FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
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
COPY requirements.txt pyproject.toml ./

ENV CONDA_DIR=/opt/conda

RUN mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda3/miniconda.sh \
    $CONDA_DIR/bin/conda clean -afy

ENV PATH=$CONDA_DIR/bin:$PATH

SHELL ["/bin/bash", "-c"]

RUN pip install runpod \
    && pip install -e .[gpu] --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/

# Copy the rest of the application code
COPY . .

# Start the RunPod serverless worker
CMD ["python", "main_runpod.py"]