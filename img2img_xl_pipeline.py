# -*- coding: utf-8 -*-

import time
import torch
import tensorrt as trt
from utilities import TRT_LOGGER, PIPELINE_TYPE
from stable_diffusion_pipeline import StableDiffusionPipeline
from models import make_tokenizer

class Img2ImgXLPipeline(StableDiffusionPipeline):
	def __init__(self, scheduler="DDIM", refiner=True, *args, **kwargs):
		super(Img2ImgXLPipeline, self).__init__(*args, **kwargs, scheduler=scheduler, stages=["clip2", "unetxl", "vae"], pipeline_type=PIPELINE_TYPE.SD_XL_REFINER, vae_scaling_factor=0.13025)
		self.tokenizer2 = make_tokenizer(self.version, self.pipeline_type, self.hf_token, self.framework_model_dir, subfolder="tokenizer_2")
		self.requires_aesthetics_score = True
		self.refiner = refiner

	def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, aesthetic_score, negative_aesthetic_score, dtype):
		if self.requires_aesthetics_score:
			add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
			add_neg_time_ids = list(original_size + crops_coords_top_left + (negative_aesthetic_score,))
		else:
			add_time_ids = list(original_size + crops_coords_top_left + target_size)
			add_neg_time_ids = list(original_size + crops_coords_top_left + target_size)
		add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
		add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)
		add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0).to(device=self.device)
		return add_time_ids

	def infer(self, prompt:list, negative_prompt:list, init_image, image_height=1024, image_width=1024, guidance=5.0, seed=None, warmup=False, verbose=False, return_type="image"):
		assert len(prompt) == len(negative_prompt)

		original_size = (1024, 1024)
		crops_coords_top_left = (0, 0)
		target_size = (image_height, image_width)
		aesthetic_score = 6.0
		negative_aesthetic_score = 2.5
		strength = 0.3

		with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER):
			batch_size = len(prompt)

			torch.cuda.synchronize()
			e2e_tic = time.perf_counter()

			# Initialize timesteps
			timesteps, t_start = self.initialize_timesteps(self.denoising_steps, strength)
			latent_timestep = timesteps[:1].repeat(batch_size)

			# CLIP text encoder 2
			text_embeddings, pooled_embeddings2 = self.encode_prompt(prompt, negative_prompt, encoder="clip2", tokenizer=self.tokenizer2, pooled_outputs=True, output_hidden_states=True)

			# Time embeddings
			add_time_ids = self._get_add_time_ids(original_size, crops_coords_top_left, target_size, aesthetic_score, negative_aesthetic_score, dtype=text_embeddings.dtype)

			add_time_ids = add_time_ids.repeat(batch_size, 1)

			add_kwargs = {"text_embeds": pooled_embeddings2, "time_ids": add_time_ids}

			# Pre-process input image
			init_image = self.preprocess_images(batch_size, (init_image,))[0]

			# VAE encode init image
			if init_image.shape[1] == 4:
				init_latents = init_image
			else:
				init_latents = self.encode_image(init_image)

			# Add noise to latents using timesteps
			noise = torch.randn(init_latents.shape, device=self.device, dtype=torch.float32)
			latents = self.scheduler.add_noise(init_latents, noise, t_start, latent_timestep)

			# UNet denoiser
			latents = self.denoise_latent(latents, text_embeddings, timesteps=timesteps, step_offset=t_start, denoiser="unetxl", guidance=guidance, add_kwargs=add_kwargs)

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
				print("SD-XL Refiner Pipeline")
				self.print_summary(self.denoising_steps, e2e_tic, e2e_toc, batch_size)
				self.save_image(images, "txt2img-xl", prompt)

		return images, (e2e_toc - e2e_tic)*1000.
