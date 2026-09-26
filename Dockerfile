FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serverless.txt .

RUN pip install --no-cache-dir -r requirements-serverless.txt

RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/ByteDance-Seed/Depth-Anything-3.git

COPY handler.py .

CMD ["python", "-u", "handler.py"]
