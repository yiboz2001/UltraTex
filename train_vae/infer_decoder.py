# ============================================================
# TRAIN + EVAL (ONE FILE) + DeepSpeed ZeRO-2
# - Background latent is FIXED (buffer, not trainable)
# - Train VAE DECODER ONLY
# - Loss: foreground L2
# ============================================================

import os
os.environ["WANDB_MODE"] = "offline"
os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "18000000000000"
import cv2 
import warnings
warnings.filterwarnings(
    "ignore",
    message="The torch.cuda.*DtypeTensor constructors are no longer recommended"
)

import argparse
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from PIL import Image
from tqdm import tqdm
from einops import rearrange
from skimage import color

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from accelerate import Accelerator

from ultratex.data.dataset import FluxPairedDatasetV2
from ultratex.backbones.flux1.sampling import unpack
from ultratex.backbones.flux1.util import load_ae


# ============================================================
# Utils
# ============================================================

def parse_step_from_ckpt(path: str) -> int:
    fname = os.path.basename(path)
    m = re.search(r"step[_\-]?(\d+)", fname)
    # Released checkpoints are named decoder.pt and carry no step number.
    return int(m.group(1)) if m else 0


def save_foreground_debug(
    pred, target, masks_full,
    out_dir, step, tag="tile"
):
    """
    pred, target: [1, 3, H, W] in [-1, 1]
    masks_full:   [H, W] or [1, H, W]
    """
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        # ---------- match foreground_l2_loss ----------
        if masks_full.dim() == 2:
            masks_full = masks_full[None, None]   # [1,1,H,W]
        elif masks_full.dim() == 3:
            masks_full = masks_full[:, None]

        fg = masks_full != 0                      # [1,1,H,W]

        # ---------- to HWC ----------
        pred_hw = pred.permute(0, 2, 3, 1)        # [1,H,W,3]
        tgt_hw  = target.permute(0, 2, 3, 1)

        # ---------- init white background ----------
        H, W = fg.shape[-2:]
        white = torch.ones((1, H, W, 3), device=pred.device)

        pred_vis = white.clone()
        tgt_vis  = white.clone()

        # ---------- fill foreground pixels only ----------
        pred_vis[fg.squeeze(1)] = pred_hw[fg.squeeze(1)]
        tgt_vis [fg.squeeze(1)] = tgt_hw [fg.squeeze(1)]

        # ---------- [-1,1] → [0,255] ----------
        pred_vis = ((pred_vis + 1) * 127.5).clamp(0, 255)
        tgt_vis  = ((tgt_vis  + 1) * 127.5).clamp(0, 255)

        pred_vis = pred_vis[0].cpu().numpy().astype(np.uint8)
        tgt_vis  = tgt_vis [0].cpu().numpy().astype(np.uint8)

        # ---------- concatenate horizontally ----------
        concat = np.concatenate([tgt_vis, pred_vis], axis=1)

        Image.fromarray(concat).save(
            os.path.join(out_dir, f"{tag}_step{step}.png")
        )


def mean_deltaE2000(imgA, imgB, mask=None):
    labA = color.rgb2lab(imgA)
    labB = color.rgb2lab(imgB)
    deltaE = color.deltaE_ciede2000(labA, labB)
    if mask is not None:
        deltaE = deltaE[mask]
    return float(deltaE.mean())


def foreground_l2_loss(pred, target, masks_full):
    if masks_full.dim() == 2:
        masks_full = masks_full[None, None]
    elif masks_full.dim() == 3:
        masks_full = masks_full[:, None]

    fg = masks_full != 0
    pred_fg = pred.permute(0, 2, 3, 1)[fg.squeeze(1)]
    target_fg = target.permute(0, 2, 3, 1)[fg.squeeze(1)]
    return F.mse_loss(pred_fg, target_fg, reduction="mean")


# ============================================================
# Background latent (FIXED)
# ============================================================

def init_base_latent_tensor(vae, device, dataset_transform, pad, dtype=torch.float32):
    black = Image.new("RGB", (pad, pad), color=(0, 0, 0))
    x = dataset_transform(black).unsqueeze(0).to(device, dtype)
    with torch.no_grad():
        z = vae.encode(x)
    return z


class BaseLatentModule(nn.Module):
    def __init__(self, init_tensor: torch.Tensor):
        super().__init__()
        # <<< CHANGED >>> fixed buffer, NOT trainable
        self.register_buffer("base_latent", init_tensor)

    def build_bg_latent(self, resolution: int):
        H_lat = (2 * resolution) // 8
        W_lat = (3 * resolution) // 8
        h0, w0 = self.base_latent.shape[-2:]
        assert H_lat % h0 == 0 and W_lat % w0 == 0
        return self.base_latent.repeat(1, 1, H_lat // h0, W_lat // w0)


# ============================================================
# Latent ops
# ============================================================

def mix_latent(img_latent, bg_latent, masks_noise, height, width):
    img_tok = rearrange(img_latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    bg_tok  = rearrange(bg_latent,  "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    idx = (masks_noise.reshape(-1) != 0).nonzero(as_tuple=True)[0]
    mixed = bg_tok.clone()
    mixed[:, idx] = img_tok[:, idx]
    return unpack(mixed, height, width)


def encode_grid(img, vae, rows=2, cols=3):
    # <<< decoder-only training: encode WITHOUT grad >>>
    assert img.dim() == 4
    B, C, H, W = img.shape
    h = H // rows
    w = W // cols
    patches = rearrange(img, "b c (r h) (co w) -> (b r co) c h w", r=rows, co=cols, h=h, w=w)
    with torch.no_grad():
        z = torch.cat([vae.encode(p.unsqueeze(0)) for p in patches], dim=0)
    z = rearrange(z, "(b r co) c h w -> b c (r h) (co w)", b=B, r=rows, co=cols)
    return z


def decode_grid(vae, z):
    rows = torch.chunk(z, 2, dim=-2)
    x = torch.cat(
        [torch.cat([vae.decode(c) for c in torch.chunk(r, 3, dim=-1)], dim=-1)
         for r in rows],
        dim=-2
    )
    return x


# ============================================================
# TRAIN + EVAL
# ============================================================

def train_and_eval(
    train_json,
    eval_json,
    out_dir,
    decoder_ckpt,
    resolution=1024,
    eval_resolution=1024,
    lr=1e-5,
    train_steps=300000,
    batch_size=1,
    grad_accum_steps=1,
    num_workers=24,
):
    os.makedirs(out_dir, exist_ok=True)

    accelerator = Accelerator(
        project_dir=out_dir,
        mixed_precision="no",
        gradient_accumulation_steps=grad_accum_steps,
        log_with="tensorboard",
    )
    accelerator.init_trackers(project_name="")

    # ========================================================
    # eval dir: out_dir/eval_{step}
    # ========================================================

    step = parse_step_from_ckpt(decoder_ckpt)
    eval_dir = os.path.join(out_dir, f"eval_{step}")

    if accelerator.is_local_main_process:
        os.makedirs(eval_dir, exist_ok=True)
    accelerator.wait_for_everyone()



    print(
        f"[rank {accelerator.process_index}] "
        f"world_size={accelerator.num_processes}"
    )
    
    eval_ds  = FluxPairedDatasetV2(eval_json, eval_resolution, None)
    eval_sampler = None
    if accelerator.num_processes > 1:
        eval_sampler = DistributedSampler(
            eval_ds,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=False,
            drop_last=False,
        )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=1,
        shuffle=False if eval_sampler is not None else False,
        sampler=eval_sampler,
        collate_fn=eval_ds.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ========================================================
    # VAE
    # ========================================================

    vae = load_ae("flux-dev", device=accelerator.device)
    vae.train()

    # <<< CHANGED >>> decoder only
    vae.encoder.requires_grad_(False)
    vae.decoder.requires_grad_(True)

    # ========================================================
    # Load decoder weights (inference)
    # ========================================================

    state = torch.load(decoder_ckpt, map_location="cpu")
    missing, unexpected = vae.decoder.load_state_dict(state, strict=False)
    if accelerator.is_local_main_process:
        print(f"[Decoder ckpt] {decoder_ckpt}")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

    vae.eval()

    # ========================================================
    # Fixed background latent
    # ========================================================

    init_tensor = init_base_latent_tensor(
        vae, accelerator.device, eval_ds.transform, pad=resolution
    )

    bg_module = BaseLatentModule(init_tensor)
    bg_module = bg_module.to(accelerator.device)

    eval_loader = eval_loader

    H = resolution * 2
    W = resolution * 3
    h_lat_tile = resolution // 8
    w_lat_tile = resolution // 8

    # ========================================================
    # EVAL
    # ========================================================

    for batch in tqdm(eval_loader):
        id_ = batch["id"][0]
        img = batch["img"][0].to(accelerator.device)
        raw_img = batch["raw_img"][0]
        
        masks_noise = batch["masks"][0][0].reshape(-1).to(accelerator.device)
        masks_full = batch["masks_full"][0][0]

        with torch.no_grad():
            z = encode_grid(img[None], vae)
            bg = bg_module.build_bg_latent(eval_resolution)
            mixed = mix_latent(z, bg, masks_noise, eval_resolution * 2, eval_resolution * 3)
            pred = decode_grid(vae, mixed)[0]

        pred = ((pred + 1) * 127.5).clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        gt   = ((raw_img  + 1) * 127.5).clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        fg = masks_full.cpu().numpy().astype(bool)
        fg = np.repeat(fg[..., None], 3, axis=2)
        # pred[~fg] = 255
        # gt[~fg] = 255

        gt_h, gt_w = gt.shape[:2]                                                                                                                                                                                                                                                                                           
        pred_h, pred_w = pred.shape[:2]                                                                                                                                                                                                                                             
        if gt_h != pred_h or gt_w != pred_w:
            gt = cv2.resize(gt, (pred_w, pred_h), interpolation=cv2.INTER_LINEAR) 
            
            
        loss = mean_deltaE2000(gt.astype(np.float32), pred.astype(np.float32))
        # ===== ensure output dir exists =====
        all_dir = os.path.join(eval_dir, "all")
        single_dir = os.path.join(eval_dir, "single")
        os.makedirs(all_dir, exist_ok=True)
        os.makedirs(single_dir, exist_ok=True)

        # ===== 1. save the full 2x3 grid, gt | pred =====
        img_all = np.concatenate([gt, pred], axis=1)  # concatenate horizontally
        Image.fromarray(img_all).save(
            os.path.join(all_dir, f"{id_}_{round(loss, 4)}.png")
        )

        # ===== 2. crop the first row / first column tile =====
        H = gt.shape[0] // 2
        W = gt.shape[1] // 3

        gt_00 = gt[0:H, 0:W]       # row 1, column 1
        pred_00 = pred[0:H, 0:W]            
        img_single = np.concatenate([gt_00, pred_00], axis=1)

        Image.fromarray(img_single).save(
            os.path.join(single_dir, f"{id_}_{round(loss, 4)}.png")
        )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a FLUX.1 foreground-aware VAE decoder.")
    parser.add_argument("--decoder_ckpt", default="checkpoints/flux1/decoder.pt")
    parser.add_argument("--eval_json", default="data/eval_demo.json")
    parser.add_argument("--out_dir", default="outputs/eval_vae_flux1")
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    train_and_eval(
        train_json=args.eval_json,
        eval_json=args.eval_json,
        out_dir=args.out_dir,
        decoder_ckpt=args.decoder_ckpt,
        resolution=args.resolution,
        eval_resolution=args.resolution,
        lr=1e-5,
        train_steps=0,
        batch_size=1,
        grad_accum_steps=1,
        num_workers=args.num_workers,
    )
