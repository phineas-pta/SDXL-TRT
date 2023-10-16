# -*- coding: utf-8 -*-

import argparse
import tensorrt as trt
from utilities import TRT_LOGGER
from txt2img_xl_pipeline import Txt2ImgXLPipeline
from img2img_xl_pipeline import Img2ImgXLPipeline


def parseArgs():
	parser = argparse.ArgumentParser(description="demo: Stable Diffusion XL txt2img TensorRT", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	# Stable Diffusion configuration
	grp_sd = parser.add_argument_group("Stable Diffusion configuration")
	grp_sd.add_argument("--prompt", help="Text prompt(s) to guide image generation")
	grp_sd.add_argument("--negative-prompt", default="", help="The negative prompt(s) to guide the image generation.")
	grp_sd.add_argument("--height", type=int, default=1024, help="Height of image to generate (must be multiple of 8)")
	grp_sd.add_argument("--width", type=int, default=1024, help="Height of image to generate (must be multiple of 8)")
	grp_sd.add_argument("--denoising-steps", type=int, default=50, help="Number of denoising steps")
	grp_sd.add_argument("--scheduler", type=str, default="DDIM", choices=["PNDM", "LMSD", "DPM", "DDIM", "EulerA"], help="Scheduler for diffusion process")
	grp_sd.add_argument("--guidance", type=float, default=7., help="CFG scale")
	grp_sd.add_argument("--seed", type=int, default=None, help="Seed (but somehow has no effect in this implementation)")
	grp_sd.add_argument("--use-refiner", action="store_true", help="enable refiner")

	# ONNX export
	grp_onnx = parser.add_argument_group("ONNX export")
	grp_onnx.add_argument("--onnx-opset", type=int, default=17, choices=range(7,18), help="Select ONNX opset version to target for exported models")
	grp_onnx.add_argument("--onnx-base-dir", default="onnx-ckpt/sdxl-1.0-base", help="Directory for SDXL-Base ONNX models")
	grp_onnx.add_argument("--onnx-refiner-dir", default="onnx-ckpt/sdxl-1.0-refiner", help="Directory for SDXL-Refiner ONNX models")

	# TensorRT engine build
	grp_trt = parser.add_argument_group("TensorRT engine build & inference")
	grp_trt.add_argument("--build-static-batch", action="store_true", help="Build TensorRT engines with fixed batch size.")
	grp_trt.add_argument("--build-dynamic-shape", action="store_true", help="Build TensorRT engines with dynamic image shapes.")
	grp_trt.add_argument("--build-preview-features", action="store_true", help="Build TensorRT engines with preview features.")
	grp_trt.add_argument("--build-all-tactics", action="store_true", help="Build TensorRT engines using all tactic sources.")
	grp_trt.add_argument("--timing-cache", default=None, type=str, help="Path to the precached timing measurements to accelerate build.")
	grp_trt.add_argument("--engine-base-dir", default="trt-engine/xl_base", help="Directory for SDXL-Base TensorRT engines")
	grp_trt.add_argument("--engine-refiner-dir", default="trt-engine/xl_refiner", help="Directory for SDXL-Refiner TensorRT engines")
	grp_trt.add_argument("--num-warmup-runs", type=int, default=5, help="Number of warmup runs before benchmarking performance")
	grp_trt.add_argument("--use-cuda-graph", action="store_true", help="Enable cuda graph")

	# miscellaneous
	grp_misc = parser.add_argument_group("miscellaneous")
	grp_misc.add_argument("--framework-model-dir", default="pytorch_model", help="Directory for HF saved models")
	grp_misc.add_argument("--output-dir", default="output", help="Output directory for logs and image artifacts")
	grp_misc.add_argument("--hf-token", type=str, help="HuggingFace API access token for downloading model checkpoints")
	grp_misc.add_argument("-v", "--verbose", action="store_true", help="Show verbose output")

	return parser.parse_args()

args = parseArgs()

BATCH_SIZE = 1
pos_prompt, neg_prompt = [args.prompt], [args.negative_prompt]

# Validate image dimensions
if args.height % 8 != 0 or args.width % 8 != 0:
	raise ValueError(f"Image height and width have to be divisible by 8 but specified as: {args.height} and {args.width}.")

if args.use_cuda_graph and (not args.build_static_batch or args.build_dynamic_shape):
	raise ValueError("Using CUDA graph requires static dimensions. Enable `--build-static-batch` and do not specify `--build-dynamic-shape`")

def init_pipeline(pipeline_class, refiner, onnx_dir, engine_dir):
	demo = pipeline_class(
		scheduler=args.scheduler,
		denoising_steps=args.denoising_steps,
		output_dir=args.output_dir,
		version="xl-1.0",
		hf_token=args.hf_token,
		verbose=args.verbose,
		max_batch_size=BATCH_SIZE,
		use_cuda_graph=args.use_cuda_graph,
		refiner=refiner,
		framework_model_dir=args.framework_model_dir
	)
	demo.loadEngines(
		engine_dir=engine_dir,
		framework_model_dir=args.framework_model_dir,
		onnx_dir=onnx_dir,
		onnx_opset=args.onnx_opset,
		opt_batch_size=BATCH_SIZE,
		opt_image_height=args.height,
		opt_image_width=args.width,
		static_batch=args.build_static_batch,
		static_shape=not args.build_dynamic_shape,
		enable_preview=args.build_preview_features,
		enable_all_tactics=args.build_all_tactics,
		timing_cache=args.timing_cache
	)
	demo.activateEngines()
	demo.loadResources(args.height, args.width, BATCH_SIZE, args.seed)
	return demo

# Register TensorRT plugins
trt.init_libnvinfer_plugins(TRT_LOGGER, "")


print("[I] SDXL base: init")
demo_base = init_pipeline(Txt2ImgXLPipeline, False, args.onnx_base_dir, args.engine_base_dir)

print("[I] SDXL base: warm up")
for i in range(args.num_warmup_runs):
	_ = demo_base.infer(pos_prompt, neg_prompt, args.height, args.width, warmup=True, verbose=args.verbose)

print("[I] SDXL base: generate image")
images_base, _ = demo_base.infer(
	prompt=pos_prompt, negative_prompt=neg_prompt,
	image_height=args.height, image_width=args.width,
	guidance=args.guidance, seed=args.seed,
	warmup=False, verbose=args.verbose,
	return_type="latents" if args.use_refiner else "image"
)

# no need this if u have 24gb vram
demo_base.teardown()
del demo_base

if args.use_refiner:
	print("[I] SDXL refiner: init")
	demo_refiner = init_pipeline(Img2ImgXLPipeline, True, args.onnx_refiner_dir, args.engine_refiner_dir)

	print("[I] SDXL refiner: warm up")
	for i in range(args.num_warmup_runs):
		_ = demo_refiner.infer(pos_prompt, neg_prompt, images_base, args.height, args.width, warmup=True, verbose=args.verbose)

	print("[I] SDXL refiner: generate image")
	images_refiner, _ = demo_refiner.infer(
		prompt=pos_prompt, negative_prompt=neg_prompt, init_image=images_base,
		image_height=args.height, image_width=args.width,
		guidance=args.guidance, seed=args.seed,
		warmup=False, verbose=args.verbose, return_type="image"
	)

	demo_refiner.teardown()
	del demo_refiner
