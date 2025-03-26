## Install

```shell
uv venv --python 3.13.2

source .venv/bin/activate

uv pip install setuptools

CXX=clang++ CC=clang uv pip install -r requirements.txt --no-build-isolation
```

## Build KenLM

```
set -xg CXX clang++ 
set -xg CC clang 

git clone https://github.com/kpu/kenlm/

cd kenlm

mkdir -p build
cd build
cmake ..
make -j 4
```

## Build image

```shell
docker build -t kenlm-gradio .
```

## Run

```shell
docker run -it --rm -p 8888:7860 kenlm-gradio
```
