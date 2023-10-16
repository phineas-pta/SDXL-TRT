# -*- coding: utf-8 -*-

import time
import torch
import tensorrt as trt
from utilities import TRT_LOGGER, PIPELINE_TYPE
from stable_diffusion_pipeline import StableDiffusionPipeline
from models import make_tokenizer

class Txt2ImgXLPipeline(StableDiffusionPipeline):
	def __init__(self, scheduler="DDIM", refiner=False, *args, **kwargs):
		super(Txt2ImgXLPipeline, self).__init__(*args, **kwargs, scheduler=scheduler, stages=["clip", "clip2", "unetxl", "vae"], pipeline_type=PIPELINE_TYPE.SD_XL_BASE, vae_scaling_factor=0.13025)
		self.tokenizer2 = make_tokenizer(self.version, self.pipeline_type, self.hf_token, self.framework_model_dir, subfolder="tokenizer_2")
		self.refiner = refiner

	def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
		add_time_ids = list(original_size + crops_coords_top_left + target_size)
		add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
		return add_time_ids

	def infer(self, prompt:list, negative_prompt:list, image_height=1024, image_width=1024, guidance=5.0, seed=None, warmup=False, verbose=False, return_type="latents"):
		assert len(prompt) == len(negative_prompt)

		original_size = (1024, 1024)
		crops_coords_top_left = (0, 0)
		target_size = (image_height, image_width)
		batch_size = len(prompt)

		with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER):
			# Pre-initialize latents
			latents = self.initialize_latents(batch_size=batch_size, unet_channels=4, latent_height=image_height//8, latent_width=image_width//8)

			torch.cuda.synchronize()
			e2e_tic = time.perf_counter()

			# CLIP text encoder
			text_embeddings = self.encode_prompt(prompt, negative_prompt, encoder="clip", tokenizer=self.tokenizer, output_hidden_states=True)
			# CLIP text encoder 2
			text_embeddings2, pooled_embeddings2 = self.encode_prompt(prompt, negative_prompt, encoder="clip2", tokenizer=self.tokenizer2, pooled_outputs=True, output_hidden_states=True)

			# Merged text embeddings
			text_embeddings = torch.cat([text_embeddings, text_embeddings2], dim=-1)

			# Time embeddings
			add_time_ids = self._get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype=text_embeddings.dtype)
			add_time_ids = add_time_ids.repeat(batch_size, 1)
			add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0).to(self.device)

			add_kwargs = {"text_embeds": pooled_embeddings2, "time_ids": add_time_ids}

			# UNet denoiser
			latents = self.denoise_latent(latents, text_embeddings, denoiser="unetxl", guidance=guidance, add_kwargs=add_kwargs)

		# FIXME - SDXL/VAE torch fallback
		with torch.inference_mode():
			# VAE decode latent
			if return_type == "latents":
				images = latents * self.vae_scaling_factor
			else:
				images = self.decode_latent(latents)

			torch.cuda.synchronize()
			e2e_toc = time.perf_counter()

			if not warmup:
				print("SD-XL Base Pipeline")
				self.print_summary(self.denoising_steps, e2e_tic, e2e_toc, batch_size)
				if return_type == "image":
					self.save_image(images, "txt2img-xl", prompt)

		return images, (e2e_toc - e2e_tic)*1000.
