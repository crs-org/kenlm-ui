---
license: apache-2.0
title: Evaluate ASR outputs
sdk: docker
emoji: 👀
colorFrom: green
colorTo: gray
short_description: 'Calculate WER/CER values from JSONL files made by ASR models'
---

## Install

```shell
uv venv --python 3.13.2

source .venv/bin/activate

uv pip install -r requirements.txt

# in development mode
uv pip install -r requirements-dev.txt
```

## Build image

```shell
docker build -t evaluate-asr-outputs .
```

## Run

```shell
docker run -it --rm -p 8888:7860 evaluate-asr-outputs
```
