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

uv pip install -r requirements.txt

uv pip install https://github.com/kpu/kenlm/archive/master.zip
```

## Build KenLM in a container

```
git clone https://github.com/kpu/kenlm/

mkdir kenlm/build
cd kenlm/build

cmake ..
make -j2
```

## Build image

```shell
docker build -t kenlm-trainer-gradio .
```

## Run

```shell
docker run -it --rm -p 8888:7860 --name kenlm-trainer kenlm-trainer-gradio
```
