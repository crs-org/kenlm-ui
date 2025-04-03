---
license: apache-2.0
title: KenLM UI
sdk: docker
emoji: 📖
colorFrom: green
colorTo: gray
short_description: 'Score texts and build KenLMs'
---

# KenLM UI

## Install

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.12.9

source .venv/bin/activate

uv pip install --upgrade setuptools wheel
uv pip install https://github.com/kpu/kenlm/archive/master.zip --no-build-isolation

uv pip install -r requirements.txt
```

## Run

```shell
uv run app.py
```

### Docker

### Pull from a Registry

```shell
docker run -it --rm -p 8888:7860 --name kenlm-ui ghcr.io/crs-org/kenlm-ui:0.2.0
```

### Build locally

#### Build image

```shell
docker build --platform linux/arm64 -t kenlm-trainer-gradio .
```

#### Run

```shell
docker run -it --rm -p 8888:7860 --name kenlm-trainer kenlm-trainer-gradio
```

#### Access

```shell
docker exec -it kenlm-trainer bash
```
