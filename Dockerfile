FROM python:3.13.2-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    ca-certificates \
    # python build dependencies \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    # gradio dependencies \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python -m ensurepip --upgrade && python -m pip install --upgrade pip

RUN useradd -m -u 1001 hf-space
USER hf-space

ENV HOME=/home/hf-space \
    PATH=/home/hf-space/.local/bin:${PATH} \
    PYTHONPATH=/home/hf-space/app \
    PYTHONUNBUFFERED=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_THEME=huggingface \
    SYSTEM=spaces \
    HF_HOME=/home/hf-space/app/hf-home

COPY --chown=hf-space:hf-space . ${HOME}/app

WORKDIR ${HOME}/app

RUN mkdir ${HF_HOME} && chmod a+rwx ${HF_HOME}

RUN pip install --no-cache-dir -r /home/hf-space/app/requirements.txt

CMD ["python", "app.py"]
