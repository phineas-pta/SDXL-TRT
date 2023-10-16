# Test: run Stable Diffusion XL with TensorRT natively on windows

refs:
- official guide using docker: https://huggingface.co/stabilityai/stable-diffusion-xl-1.0-tensorrt/blob/main/README.md
- original code requiring 24gb vram (coz load both base & refiner): https://github.com/rajeevsrao/TensorRT/blob/release/8.6/demo/Diffusion

my fork is an attempt to run natively on windows with 9gb vram by disable refiner

only a proof-of-concept, no plan to maintain this

i know jack shit how to convert custom models, watch this repo for update: https://github.com/Amblyopius/Stable-Diffusion-ONNX-FP16

## License

![LICENSE](https://www.gnu.org/graphics/gplv3-with-text-136x68.png)

## Benchmark

to be added: normal pytorch (comyfui) vs tensorrt

## 1️⃣ prepare environment

test environment:
- cuda 12.1
- cudnn 8.9.5
- tensorrt 8.6.1.6
- visual studio 17 (2022)
- python 3.11

requirements:
- 25gb free space for models (13gb downloaded + 12gb converted)
- 9gb vram if not use refiner, 12gb otherwise
- 32gb ram: to be verified coz ram usage rise up then go down

follow my guide to install TensorRT: https://github.com/phineas-pta/NVIDIA-win/blob/main/NVIDIA-win.md

download this https://github.com/NVIDIA/NVTX/archive/refs/heads/release-v3.zip then unzip,<br />
navigate into `c\include` then copy folder `nvtx3` to any of `%INCLUDE%`,<br />
for e.g. `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\VS\include`

download/clone this repo

inside, create those folders:
- `onnx-ckpt/` to contain official onnx files
- `trt-engine/` to contain tensorrt engine files

prepare a fresh env (venv/conda/virtualenv) then `pip install -r requirements.txt`<br />
also install the tensorrt wheel in the tensorrt folder downloaded during TensorRT installation<br />
(coz `pip install tensorrt` only available on linux)

## 2️⃣ run

download checkpoints: `python download_ckpt.py`<br />
if slow internet (download interrupted), edit file `download_ckpt.py`: un-comment line 7

1st run may take >1h to build tensorrt engine from onnx (float16 by default)

refiner require 12gb vram, still possible with 9gb if ram swap but much slower
```
set CUDA_MODULE_LOADING=LAZY

python my_demo.py
	--prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
	--negative-prompt="ugly, deformed"
	--scheduler="DPM"
	--denoising-steps=50
	--guidance=7
	--use-refiner
	--output-dir="output"
```
a bit faster inference if enable `--build-static-batch --use-cuda-graph`
