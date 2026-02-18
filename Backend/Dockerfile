FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OPEN3D_CPU_RENDERING=true
ENV PIP_CACHE_DIR=/pip-cache


RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app


RUN pip install --upgrade pip


RUN pip install --no-cache-dir torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121


COPY requirements-docker.txt requirements-docker.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-docker.txt


COPY . .

EXPOSE 8443

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8443"]