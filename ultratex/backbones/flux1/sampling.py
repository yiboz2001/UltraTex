# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Copyright (c) 2024 Black Forest Labs and The XLabs-AI Team. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Literal

import torch
from einops import rearrange, repeat
from torch import Tensor
from tqdm import tqdm

from .model import Flux
from .modules.conditioner import HFEmbedder
import os

CACHE_DIR = "outputs/.cache_caption"
os.makedirs(CACHE_DIR, exist_ok=True)


def cache_load_or_compute(prompt, t5_model, clip_model, bs):
    """
    Single entry point:
    - prompt -> cache key
    - cache hit: load it and skip the text encoders
    - cache miss: compute txt / txt_ids / vec, cache, then free the encoders
    """
    # normalise prompt to list[str]
    if isinstance(prompt, str):
        prompt_list = [prompt]
    else:
        prompt_list = prompt

    # cache path, one file per prompt
    cache_key = "_".join(prompt_list).replace("/", "_")
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")

    # ----------------------------
    # 1) try the cache
    # ----------------------------
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)

        txt = data["txt"]
        txt_ids = data["txt_ids"]
        vec = data["vec"]

        # repeat when the cache holds one sample but bs > 1
        if txt.shape[0] == 1 and bs > 1:
            txt = repeat(txt, "1 ... -> bs ...", bs=bs)
        if vec.shape[0] == 1 and bs > 1:
            vec = repeat(vec, "1 ... -> bs ...", bs=bs)

        return txt, txt_ids, vec

    # ----------------------------
    # 2) cache miss: compute
    # ----------------------------

    # ---- T5 ----
    txt = t5_model(prompt_list)            # (1, L, D)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)

    # original behaviour: txt_ids are all zeros
    txt_ids = torch.zeros(bs, txt.shape[1], 3, device=txt.device, dtype=txt.dtype)

    # ---- CLIP ----
    vec = clip_model(prompt_list)          # (1, D)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    # ----------------------------
    # 3) cache a single sample
    # ----------------------------
    to_save = {
        "txt": txt[0:1].cpu(),        # store one sample to save space
        "txt_ids": txt_ids[0:1].cpu(),
        "vec": vec[0:1].cpu(),
    }
    with open(cache_path, "wb") as f:
        pickle.dump(to_save, f)

    # ----------------------------
    # 4) free the encoders to release VRAM
    # ----------------------------
    del t5_model
    del clip_model
    torch.cuda.empty_cache()

    return txt, txt_ids, vec

def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def prepare(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: Tensor,
    prompt: str | list[str],
    ref_img: None | Tensor=None,
    pe: Literal['d', 'h', 'w', 'o'] ='d'
) -> dict[str, Tensor]:
    assert pe in ['d', 'h', 'w', 'o']
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if ref_img is not None:
        _, _, ref_h, ref_w = ref_img.shape
        ref_img = rearrange(ref_img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if ref_img.shape[0] == 1 and bs > 1:
            ref_img = repeat(ref_img, "1 ... -> bs ...", bs=bs)
        ref_img_ids = torch.zeros(ref_h // 2, ref_w // 2, 3)
        # offset img ids by the max extent in H and W
        h_offset = h // 2 if pe in {'d', 'h'} else 0
        w_offset = w // 2 if pe in {'d', 'w'} else 0
        ref_img_ids[..., 1] = ref_img_ids[..., 1] + torch.arange(ref_h // 2)[:, None] + h_offset
        ref_img_ids[..., 2] = ref_img_ids[..., 2] + torch.arange(ref_w // 2)[None, :] + w_offset
        ref_img_ids = repeat(ref_img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]

    # txt = t5(prompt)
    # if txt.shape[0] == 1 and bs > 1:
    #     txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    # txt_ids = torch.zeros(bs, txt.shape[1], 3)

    # vec = clip(prompt)
    # if vec.shape[0] == 1 and bs > 1:
    #     vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    txt, txt_ids, vec = cache_load_or_compute(prompt, t5, clip, bs)
    if ref_img is not None:
        return {
            "img": img,
            "img_ids": img_ids.to(img.device),
            "ref_img": ref_img,
            "ref_img_ids": ref_img_ids.to(img.device),
            "txt": txt.to(img.device),
            "txt_ids": txt_ids.to(img.device),
            "vec": vec.to(img.device),
        }
    else:
        return {
            "img": img,
            "img_ids": img_ids.to(img.device),
            "txt": txt.to(img.device),
            "txt_ids": txt_ids.to(img.device),
            "vec": vec.to(img.device),
        }

def prepare_multi_ip(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: Tensor,
    prompt: str | list[str],
    ref_imgs: list[Tensor] | None = None,
    pe: Literal['d', 'h', 'w', 'o'] = 'd'
) -> dict[str, Tensor]:
    assert pe in ['d', 'h', 'w', 'o']
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    ref_img_ids = []
    ref_imgs_list = []
    pe_shift_w, pe_shift_h = w // 2, h // 2
    
    for ref_img in ref_imgs:
        _, _, ref_h1, ref_w1 = ref_img.shape
        ref_img = rearrange(ref_img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if ref_img.shape[0] == 1 and bs > 1:
            ref_img = repeat(ref_img, "1 ... -> bs ...", bs=bs)
        ref_img_ids1 = torch.zeros(ref_h1 // 2, ref_w1 // 2, 3)
        
        # offset img ids by the max extent in H and W
        h_offset = pe_shift_h if pe in {'d', 'h'} else 0
        w_offset = pe_shift_w if pe in {'d', 'w'} else 0
        ref_img_ids1[..., 1] = ref_img_ids1[..., 1] + torch.arange(ref_h1 // 2)[:, None] + h_offset
        ref_img_ids1[..., 2] = ref_img_ids1[..., 2] + torch.arange(ref_w1 // 2)[None, :] + w_offset
        ref_img_ids1 = repeat(ref_img_ids1, "h w c -> b (h w) c", b=bs)
        ref_img_ids.append(ref_img_ids1)
        ref_imgs_list.append(ref_img)

        # update the pe shift
        pe_shift_h += ref_h1 // 2
        pe_shift_w += ref_w1 // 2

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "ref_img": tuple(ref_imgs_list),
        "ref_img_ids": [ref_img_id.to(img.device) for ref_img_id in ref_img_ids],
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    ref_img: Tensor=None,
    ref_img_ids: Tensor=None,
    masks=None
):
    i = 0
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in tqdm(zip(timesteps[:-1], timesteps[1:]), total=len(timesteps) - 1):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=img,
            img_ids=img_ids,
            ref_img=ref_img,
            ref_img_ids=ref_img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            masks=masks,
        )
        img = img + (t_prev - t_curr) * pred
        i += 1
    return img

def denoise_test(
    model: Flux,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    ref_img: Tensor=None,
    ref_img_ids: Tensor=None,
    masks=None,
    background=None,
    global_i=None,
    step_callback=None,
    step_callback_rgb=None,
    update_all=False,
):
    if update_all:
        masks = [[torch.ones_like(sample[0]), *sample[1:]] for sample in masks]
    masks_noise = torch.cat([masks[0][0].reshape(-1)]) 
    mask_noise_idx = masks_noise.nonzero(as_tuple=True)[0]
    i = 0
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    if step_callback is not None:
        step_callback(0, img)
    if step_callback_rgb is not None:
        step_callback_rgb(0, img)
    
    # os.makedirs(f"outputs/attention_vis", exist_ok=True)
    for t_curr, t_prev in tqdm(zip(timesteps[:-1], timesteps[1:]), total=len(timesteps) - 1):
        # os.makedirs(f"outputs/attention_vis", exist_ok=True)
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model.forward(
            img=img,
            img_ids=img_ids,
            ref_img=ref_img,
            ref_img_ids=ref_img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            masks=masks,
            # attention_save_path = f"outputs/attention_vis"
        )
        if update_all:
            img = img + (t_prev - t_curr) * pred
        else:
            img[:, mask_noise_idx] = img[:, mask_noise_idx] + (t_prev - t_curr) * pred
        i += 1
        if step_callback is not None:
            step_callback(i, img)
        if step_callback_rgb is not None:
            step_callback_rgb(i, img)
    
    if background is not None and not update_all:
        background[:, mask_noise_idx] = img[:, mask_noise_idx]
        img = background
        
    return img

def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
