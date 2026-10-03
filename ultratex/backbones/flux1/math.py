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

import torch
from einops import rearrange
from torch import Tensor
import math
from torchvision.io import write_png, write_jpeg  
from ultratex.sparse_attention import SparseLinearAttention

sparse_attn = SparseLinearAttention(
    head_dim=128,
    topk=0.20,              
    feature_map="softmax",
    BLKQ=128,
    BLKK=64,
)

def attention_masked(q, k, v, pe, masks=None, attention_save_path="", index_map=None):
    global sparse_attn
    q, k = apply_rope(q, k, pe)
    B,H,L,D = q.shape
    if masks is None:
        attention = sparse_attn(q, k, v)
        # with torch.backends.cuda.sdp_kernel(
        #     enable_flash=True,
        #     enable_math=False,
        # ):
        #     attention = torch.nn.functional.scaled_dot_product_attention(q,k,v)
        return rearrange(attention, "B H L D -> B L (H D)")

    masks = torch.cat([m.reshape(-1) for m in masks[0]])
    M = masks.shape[0]

    mask_idx  = masks.nonzero(as_tuple=True)[0]
    tail_idx = torch.arange(M, L, device=q.device)
    idx = torch.cat([mask_idx, tail_idx])
    q_sel, k_sel, v_sel = q[:,:,idx], k[:,:,idx], v[:,:,idx] 
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_mem_efficient=True,
        enable_math=False,
    ):
        out_sel = torch.nn.functional.scaled_dot_product_attention(q_sel, k_sel, v_sel)

    out_full = torch.zeros((B,H,L,D), device=q.device, dtype=q.dtype)
    out_full[:,:,idx,:] = out_sel
    
    return rearrange(out_full, "B H L D -> B L (H D)")


# def attention_masked(q, k, v, pe, masks=None, attention_save_path="", index_map=None):
#     q, k = apply_rope(q, k, pe)
#     B,H,L,D = q.shape

#     if masks is None:
#         vmax_fixed = 1e-3
#         vmin_fixed = 1e-7
#         PAD = 50

#         with torch.no_grad():
#             q_ = q.float()
#             k_ = k.float()
#             scores = torch.matmul(q_, k_.transpose(-1, -2)) / math.sqrt(D)
#             attn = torch.softmax(scores, dim=-1)[0] # [H, L, L]

#             H = attn.shape[0]
#             L = attn.shape[-1]
#             denom = float(vmax_fixed - vmin_fixed)

#             lengths = [
#                 int(index_map["text"]),
#                 int(index_map["nosie"][0]),
#                 int(index_map["nosie"][1]),
#                 int(index_map["cond_image"][0]),
#                 int(index_map["cond_image"][1]),
#                 int(index_map["ref_image"]),
#             ]
#             assert sum(lengths) == L, f"sum(lengths)={sum(lengths)} != L={L}"
            
#             COLORS = torch.tensor([
#                 [255,   0,   0],   # text: red
#                 [197, 216, 157],   # noise1: green
#                 [  0,   0, 255],   # noise2: blue
#                 [255, 255,   0],   # cond1: yellow
#                 [255,   0, 255],   # cond2: magenta
#                 [227, 116, 52],   # ref: cyan
#             ], dtype=torch.uint8, device=attn.device)  # [6,3]
            
#             strip = torch.zeros(3, L, dtype=torch.uint8, device=attn.device)
#             ptr = 0
#             for i, seg_len in enumerate(lengths):
#                 if seg_len > 0:
#                     strip[:, ptr:ptr + seg_len] = COLORS[i].view(3, 1)
#                     ptr += seg_len
                    
#             def add_rowcol_tags(img_uint8: torch.Tensor) -> torch.Tensor:
#                 canvas = torch.zeros(3, L + PAD, L + PAD, dtype=torch.uint8, device=img_uint8.device)
#                 canvas[:, :PAD, :PAD] = 255
#                 canvas[:, PAD:, PAD:] = img_uint8
#                 canvas[:, :PAD, PAD:] = strip.unsqueeze(1).expand(3, PAD, L)
#                 canvas[:, PAD:, :PAD] = strip.unsqueeze(2).expand(3, L, PAD)
#                 return canvas            
        
#             for h in range(H):
#                 a = attn[h]
#                 a = a.clamp(min=vmin_fixed, max=vmax_fixed)
#                 intensity = (a - vmin_fixed) / denom
#                 intensity = intensity.clamp(0, 1)

#                 red   = torch.full_like(intensity, 255.0)
#                 green = 255.0 * (1.0 - intensity)
#                 blue  = 255.0 * (1.0 - intensity)

#                 img = torch.stack([red, green, blue], dim=0)
#                 img = img.round().clamp(0, 255).to(torch.uint8)
#                 img = add_rowcol_tags(img)
                
#                 write_jpeg(img.cpu(),
#                         f"{attention_save_path}_h{h}.jpg",
#                         quality=90)

#             a_mean = attn.mean(dim=0)  # [L, L]
#             a_mean = a_mean.clamp(min=vmin_fixed, max=vmax_fixed)
#             intensity = (a_mean - vmin_fixed) / denom
#             intensity = intensity.clamp(0, 1)

#             red   = torch.full_like(intensity, 255.0)
#             green = 255.0 * (1.0 - intensity)
#             blue  = 255.0 * (1.0 - intensity)

#             img_mean = torch.stack([red, green, blue], dim=0)
#             img_mean = img_mean.round().clamp(0, 255).to(torch.uint8)
#             img_mean = add_rowcol_tags(img_mean)
            
#             write_jpeg(img_mean.cpu(),
#                     f"{attention_save_path}_mean.jpg",
#                     quality=90)
        
#         with torch.backends.cuda.sdp_kernel(
#             enable_flash=True,
#             enable_math=False,
#         ):
#             attention = torch.nn.functional.scaled_dot_product_attention(q,k,v)
#         return rearrange(attention, "B H L D -> B L (H D)")

#     masks = torch.cat([m.reshape(-1) for m in masks[0]])
#     M = masks.shape[0]  

#     mask_idx  = masks.nonzero(as_tuple=True)[0]
#     tail_idx = torch.arange(M, L, device=q.device)
#     idx = torch.cat([mask_idx, tail_idx])
#     q_sel, k_sel, v_sel = q[:,:,idx], k[:,:,idx], v[:,:,idx] 
#     with torch.backends.cuda.sdp_kernel(
#         enable_flash=True,
#         enable_mem_efficient=True,
#         enable_math=False,
#     ):
#         out_sel = torch.nn.functional.scaled_dot_product_attention(q_sel, k_sel, v_sel)

#     out_full = torch.zeros((B,H,L,D), device=q.device, dtype=q.dtype)
#     out_full[:,:,idx,:] = out_sel
    
#     return rearrange(out_full, "B H L D -> B L (H D)")


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None)
    x = rearrange(x, "B H L D -> B L (H D)")
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
