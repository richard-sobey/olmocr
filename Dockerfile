FROM --platform=linux/amd64 nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

RUN apt-get update -y && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get -y update

# Install requirements specific to pdfs
RUN apt-get update && apt-get -y install python3-apt
RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections
RUN apt-get update -y && apt-get install -y poppler-utils ttf-mscorefonts-installer msttcorefonts fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    ca-certificates \
    build-essential \
    curl \
    unzip

RUN rm -rf /var/lib/apt/lists/* \
    && unlink /usr/bin/python3 \
    && ln -s /usr/bin/python3.11 /usr/bin/python3 \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python \
    && pip3 install -U pip    

RUN apt-get update && apt-get -y install python3.11-venv 
ADD --chmod=755 https://astral.sh/uv/install.sh /install.sh
RUN /install.sh && rm /install.sh

ENV PYTHONUNBUFFERED=1
WORKDIR /root
COPY pyproject.toml pyproject.toml
COPY olmocr/version.py olmocr/version.py

RUN /root/.local/bin/uv pip install --system --no-cache -e .

RUN /root/.local/bin/uv pip install --system --no-cache sgl-kernel==0.0.3.post1 --force-reinstall --no-deps
RUN /root/.local/bin/uv pip install --system --no-cache "sglang[all]==0.4.2" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/
RUN /root/.local/bin/uv pip install --system --no-cache runpod

COPY olmocr olmocr
COPY worker.py worker.py
COPY pipeline_utility.py pipeline_utility.py

WORKDIR /root

RUN python3 -m sglang.launch_server --help
RUN python3 -m olmocr.pipeline --help

CMD ["python3", "worker.py"]