## Install

```shell
uv venv --python 3.13.2

source .venv/bin/activate

uv pip install setuptools

CXX=clang++ CC=clang uv pip install -r requirements.txt --no-build-isolation
```

## Build image

```shell
docker build -t kenlm-gradio .
```

## Run

```shell
docker run -it --rm -p 8888:7860 kenlm-gradio
```
