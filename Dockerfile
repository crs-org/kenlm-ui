FROM python:3.12.9-bookworm

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
    build-essential cmake libicu-dev libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev \
    libssl-dev \
    zlib1g-dev \
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

# Install ouch
RUN ARCHIVE="ouch_$( [ "$TARGETARCH" = "arm64" ] && echo 'aarch64-unknown-linux-gnu' || echo 'x86_64-unknown-linux-musl' ).zip" && \
    wget "https://github.com/crs-org/ouch-releases/releases/download/v0.2.0/$ARCHIVE" && \
    unzip "$ARCHIVE" && \
    mv ouch_*/ouch /tmp/ouch && \
    chmod +x /tmp/ouch && \
    /tmp/ouch --version && \
    rm -rf "$ARCHIVE" ouch_*

# Install KenLM module
RUN pip install --upgrade setuptools wheel
RUN pip install https://github.com/kpu/kenlm/archive/master.zip --no-build-isolation

# Install KenLM binaries
RUN wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz && \
    mkdir kenlm/build && \
    cd kenlm/build && \
    cmake .. && \
    make -j2 && \
    cd ../..

# Install app dependencies
RUN pip install --no-cache-dir -r /home/hf-space/app/requirements.txt

RUN mkdir ${HF_HOME} && chmod a+rwx ${HF_HOME}

CMD ["python", "app.py"]
