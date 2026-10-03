# ============================================================
# TRAIN + EVAL (ONE FILE) + DeepSpeed ZeRO-2
# - Background latent is FIXED (buffer, not trainable)
# - Train FLUX.2 VAE DECODER ONLY
# - Loss: foreground L2
# ============================================================

import os
os.environ["WANDB_MODE"] = "offline"
os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "18000000000000"
import warnings
warnings.filterwarnings(
    "ignore",
    message="The torch.cuda.*DtypeTensor constructors are no longer recommended"
)

from torch.utils.data.distributed import DistributedSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from accelerate import Accelerator, DeepSpeedPlugin
from ultratex.data.dataset import FluxPairedDatasetV2
from ultratex.backbones.flux2.util import load_ae

from skimage import color

# ============================================================
# Utils
# ============================================================

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
    # Keep the model in bf16 while accumulating the pixel loss in fp32.
    return F.mse_loss(pred_fg.float(), target_fg.float(), reduction="mean")


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
        # FLUX.2 AE encode() returns [B, 128, H/16, W/16].
        H_lat = (2 * resolution) // 16
        W_lat = (3 * resolution) // 16
        h0, w0 = self.base_latent.shape[-2:]
        assert H_lat % h0 == 0 and W_lat % w0 == 0
        return self.base_latent.repeat(1, 1, H_lat // h0, W_lat // w0)


# ============================================================
# Latent ops
# ============================================================

def mix_latent(img_latent, bg_latent, masks_noise):
    """Keep foreground cells and replace background in native FLUX.2 latent space."""
    if img_latent.shape != bg_latent.shape:
        raise ValueError(
            f"Image/background latent shapes differ: "
            f"{tuple(img_latent.shape)} vs {tuple(bg_latent.shape)}"
        )

    batch, _, height, width = img_latent.shape
    expected = batch * height * width
    if masks_noise.numel() not in (height * width, expected):
        raise ValueError(
            f"Mask has {masks_noise.numel()} cells, expected "
            f"{height * width} (shared) or {expected} (batched)"
        )

    if masks_noise.numel() == height * width:
        mask = masks_noise.reshape(1, 1, height, width)
        mask = mask.expand(batch, -1, -1, -1)
    else:
        mask = masks_noise.reshape(batch, 1, height, width)

    return torch.where(mask.to(device=img_latent.device) != 0, img_latent, bg_latent)


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
    resolution=1024,
    eval_resolution=1024,
    lr=1e-5,
    train_steps=300000,
    batch_size=1,
    grad_accum_steps=1,
    num_workers=24,
    model_name="flux.2-klein-base-4b",
    ae_model_path=None,
    save_every=100,
):
    os.makedirs(out_dir, exist_ok=True)

    # Avoid a Hugging Face metadata request on every distributed rank when a
    # shared local FLUX.2 AE checkpoint is available.
    if ae_model_path is not None:
        if not os.path.isfile(ae_model_path):
            raise FileNotFoundError(f"FLUX.2 AE checkpoint not found: {ae_model_path}")
        os.environ["AE_MODEL_PATH"] = ae_model_path

    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "reduce_scatter": True,
        },
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},
    }

    accelerator = Accelerator(
        project_dir=out_dir,
        mixed_precision="bf16",
        gradient_accumulation_steps=grad_accum_steps,
        deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=ds_config),
        log_with="tensorboard",
    )
    accelerator.init_trackers(project_name="")

    # ========================================================
    # Dataset
    # ========================================================

    train_ds = FluxPairedDatasetV2(train_json, resolution, None)
    eval_ds  = FluxPairedDatasetV2(eval_json, eval_resolution, None)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=train_ds.collate_fn, num_workers=num_workers, pin_memory=True
    )

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

    vae = load_ae(model_name, device=accelerator.device)
    vae.train()

    # FLUX.2 AE encoder and latent-normalization statistics stay frozen.
    vae.requires_grad_(False)
    vae.decoder.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        vae.decoder.parameters(), lr=lr, weight_decay=0.0
    )

    # ========================================================
    # Fixed background latent
    # ========================================================

    init_tensor = init_base_latent_tensor(
        vae, accelerator.device, train_ds.transform, pad=resolution
    )
    
    bg_module = BaseLatentModule(init_tensor)
    bg_module = bg_module.to(accelerator.device)
    
    vae, optimizer, train_loader, eval_loader = accelerator.prepare(
        vae, optimizer, train_loader, eval_loader
    )
    model_dtype = next(vae.parameters()).dtype

    H = resolution * 2
    W = resolution * 3
    h_lat_tile = resolution // 16
    w_lat_tile = resolution // 16

    # ========================================================
    # TRAIN
    # ========================================================

    pbar = tqdm(range(train_steps), disable=not accelerator.is_local_main_process)
    it = iter(train_loader)
    train_loss = 0.0

    for step in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        img = batch["img"][0].to(accelerator.device)
        masks_noise = batch["masks"][0][0].to(accelerator.device)
        masks_full = batch["masks_full"][0][0].to(accelerator.device)

        with accelerator.accumulate(vae):
            img_latent = encode_grid(torch.stack([img]).to(model_dtype), vae)
            bg_latent = bg_module.build_bg_latent(resolution).to(model_dtype)

            loss_log = 0.0

            for r in range(2):
                for c in range(3):
                    z_tile = img_latent[:, :, r*h_lat_tile:(r+1)*h_lat_tile,
                                         c*w_lat_tile:(c+1)*w_lat_tile]
                    bg_tile = bg_latent[:, :, r*h_lat_tile:(r+1)*h_lat_tile,
                                        c*w_lat_tile:(c+1)*w_lat_tile]
                    
                    mask_tile = masks_noise[
                        r*h_lat_tile:(r+1)*h_lat_tile,
                        c*w_lat_tile:(c+1)*w_lat_tile,
                    ].reshape(-1)
                    z_mix = mix_latent(z_tile, bg_tile, mask_tile)

                    pred = vae.decode(z_mix)

                    hs, ws = r * resolution, c * resolution
                    tgt = img[None, :, hs:hs+resolution, ws:ws+resolution]

                    m  = masks_full[hs:hs+resolution, ws:ws+resolution]
                    l = foreground_l2_loss(pred, tgt, m)

                    accelerator.backward(l)
                    loss_log += l.detach()
                    
                    # if (
                    #     accelerator.is_main_process
                    #     and step % 100 == 0    # <<< tune as needed
                    # ):
                    #     save_foreground_debug(
                    #         pred.detach(),
                    #         tgt.detach(),
                    #         m.detach(),
                    #         out_dir=os.path.join(out_dir, "log_fg_debug"),
                    #         step=step,
                    #         tag=f"r{r}_c{c}"
                    #     )

            if accelerator.sync_gradients:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss += accelerator.gather(loss_log).mean().item()

        if accelerator.sync_gradients:
            accelerator.log({"train_loss": train_loss}, step=step)
            if accelerator.is_local_main_process:
                pbar.set_postfix(train_loss=f"{train_loss:.6f}")
            train_loss = 0.0

        if accelerator.is_local_main_process and step % save_every == 0:
            torch.save(
                accelerator.unwrap_model(vae).decoder.state_dict(),
                os.path.join(out_dir, f"decoder_step{step}.pt")
            )
            
        accelerator.wait_for_everyone()
        if step % 1000 == 0:
            vae.eval()
            eval_dir = os.path.join(out_dir, f"eval_{step}")
            
            if accelerator.is_local_main_process:
                os.makedirs(eval_dir, exist_ok=True)
            accelerator.wait_for_everyone()

            for batch in tqdm(eval_loader):
                id_ = batch["id"][0]
                img = batch["img"][0].to(accelerator.device)
                masks_noise = batch["masks"][0][0].reshape(-1).to(accelerator.device)
                masks_full = batch["masks_full"][0][0]

                with torch.no_grad():
                    z = encode_grid(img[None].to(model_dtype), vae)
                    bg = bg_module.build_bg_latent(eval_resolution).to(model_dtype)
                    mixed = mix_latent(z, bg, masks_noise)
                    pred = decode_grid(vae, mixed)[0]

                pred = ((pred + 1) * 127.5).clamp(0, 255).permute(1, 2, 0).float().cpu().numpy().astype(np.uint8)
                gt   = ((img  + 1) * 127.5).clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

                fg = masks_full.cpu().numpy().astype(bool)
                fg = np.repeat(fg[..., None], 3, axis=2)
                pred[~fg] = 255
                gt[~fg] = 255

                loss = mean_deltaE2000(gt.astype(np.float32), pred.astype(np.float32))

                # ===== make sure output dirs exist =====
                all_dir = os.path.join(eval_dir, "all")
                single_dir = os.path.join(eval_dir, "single")
                os.makedirs(all_dir, exist_ok=True)
                os.makedirs(single_dir, exist_ok=True)

                # ===== 1. save the full 2x3 grid (GT | pred) =====
                img_all = np.concatenate([gt, pred], axis=1)  # side by side
                Image.fromarray(img_all).save(
                    os.path.join(all_dir, f"{id_}_{round(loss, 4)}.png")
                )

                # ===== 2. save just the first view (row 0, col 0) =====
                H = gt.shape[0] // 2
                W = gt.shape[1] // 3

                gt_00 = gt[0:H, 0:W]       # row 0, col 0
                pred_00 = pred[0:H, 0:W]

                img_single = np.concatenate([gt_00, pred_00], axis=1)

                Image.fromarray(img_single).save(
                    os.path.join(single_dir, f"{id_}_{round(loss, 4)}.png")
                )
                
            accelerator.wait_for_everyone()
            vae.train()

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        torch.save(
            accelerator.unwrap_model(vae).decoder.state_dict(),
            os.path.join(out_dir, f"decoder_step{train_steps}.pt"),
        )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    # Settings come from the environment so scripts/train_vae_flux2.sh can set
    # them; the defaults train on the bundled demo objects.
    env = os.environ.get
    train_and_eval(
        train_json=env("TRAIN_JSON", "data/train_demo.json"),
        eval_json=env("EVAL_JSON", "data/eval_demo.json"),
        out_dir=env("OUTPUT_DIR", "outputs/train_vae_flux2"),
        resolution=int(env("RESOLUTION", "2048")),
        eval_resolution=int(env("RESOLUTION", "2048")),
        lr=float(env("LR", "1e-5")),
        train_steps=int(env("MAX_TRAIN_STEPS", "100000")),
        batch_size=1,
        grad_accum_steps=1,
        num_workers=int(env("NUM_WORKERS", "2")),
        model_name=env("MODEL_NAME", "flux.2-klein-base-9b"),
        ae_model_path=env("AE_MODEL_PATH", "checkpoints/base/flux2/ae.safetensors"),
        save_every=int(env("SAVE_EVERY", "100")),
    )
