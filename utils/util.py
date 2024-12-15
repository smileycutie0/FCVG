import os
import imageio
import numpy as np
from typing import Union

import torch
import torchvision
import torch.distributed as dist

from safetensors import safe_open
from tqdm import tqdm
from einops import rearrange


def interpolate_matches_linear(p0, p1, frac):
    interped = np.zeros_like(p0)
    N, _, _ = p0.shape
    for i in range(N):
        pair1_distance = np.linalg.norm(p0[i, 0]-p1[i, 0]) + np.linalg.norm(p0[i, 1]-p1[i, 1])
        pair2_distance = np.linalg.norm(p0[i, 0]-p1[i, 1]) + np.linalg.norm(p0[i, 1]-p1[i, 0])
        if pair1_distance < pair2_distance:
            interped[i] = p0[i] + (p1[i] - p0[i]) * frac
        else:
            interped[i, 0] = p0[i, 0] + (p1[i, 1] - p0[i, 0]) * frac
            interped[i, 1] = p0[i, 1] + (p1[i, 0] - p0[i, 1]) * frac
    return interped

def interpolate_linear(p0, p1, frac):
    return p0 + (p1 - p0) * frac
    
def match_bodies(candidates1, candidates2, subset1, subset2):
    group_size = min(len(subset1[0]), len(subset2[0]))
    num_groups = min(len(subset1), len(subset2))
    new_match = []
    matched_list = list(range(num_groups))
    for i in range(num_groups):
        group1 = candidates1[i * group_size:(i + 1) * group_size]
        best_match = matched_list[0]
        min_distance = float('inf')

        for j in matched_list:
            group2 = candidates2[j * group_size:(j + 1) * group_size]
            distance = np.linalg.norm(np.mean(group1, axis=0) - np.mean(group2, axis=0))  
            
            if distance < min_distance:
                min_distance = distance
                best_match = j
        new_match.append(best_match)
        matched_list.remove(best_match)
    return new_match
    
def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents
    

