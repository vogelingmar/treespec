FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ADD . /workspace/treespec
WORKDIR /workspace

RUN python3 -m venv /workspace/venv
RUN /workspace/venv/bin/pip install --upgrade pip wheel setuptools
RUN /workspace/venv/bin/pip install torch torchvision
WORKDIR /workspace/treespec
RUN /workspace/venv/bin/pip install .

RUN echo ". /workspace/venv/bin/activate" >> ~/.bashrc

ENTRYPOINT ["/bin/bash", "-l", "-c"]