FROM ghcr.io/mindee/doctr:torch-viz-html-contrib-py3.12.7-v0.11.0

WORKDIR /app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH "${PYTHONPATH}:/app"

RUN apt-get update \
    && apt-get install --no-install-recommends git ffmpeg libsm6 libxext6 make vim-tiny poppler-utils exiftool libxml2-utils -y \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml  /app/pyproject.toml
COPY Makefile /app/Makefile

RUN pip install --upgrade pip setuptools wheel \
    && make lock \
    && pip install -r /app/requirements.txt \
    && pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 \
    && pip cache purge \
    && rm -rf /root/.cache/pip

# copy project
COPY app /app/app
