# ============================================================
# Evaluate VAE reconstruction (4 conditions, foreground metrics)
#   pred1: Orig VAE + noise bg (no replacement)
#   pred2: Orig VAE + black bg replacement
#   pred3: Finetuned VAE + black bg replacement
#   pred4: Orig VAE direct encode-decode (no bg manipulation)
# Metrics: foreground PSNR / SSIM / LPIPS (per view, 6 views)
# Supports multi-GPU via accelerate.
# ============================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "18000000000000"
os.environ["TORCH_HOME"] = os.environ.get(
    "TORCH_HOME", os.path.expanduser("~/.cache/torch")
)

import copy
import json
import argparse

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange

from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
import lpips

from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from accelerate import Accelerator

from ultratex.backbones.flux1.sampling import unpack
from ultratex.backbones.flux1.util import load_ae


# ============================================================
# Grid view order: row0=[0,1,3], row1=[2,4,5]
# ============================================================

def split_grid_to_views(img_np, resolution):
    """Split 2x3 grid [2*res, 3*res, 3] into 6 views [res, res, 3] in order 0-5."""
    h, w = resolution, resolution
    v0 = img_np[0:h, 0:w]
    v1 = img_np[0:h, w:2*w]
    v3 = img_np[0:h, 2*w:3*w]
    v2 = img_np[h:2*h, 0:w]
    v4 = img_np[h:2*h, w:2*w]
    v5 = img_np[h:2*h, 2*w:3*w]
    return [v0, v1, v2, v3, v4, v5]


def split_mask_to_views(mask_np, resolution):
    """Split 2x3 mask grid [2*res, 3*res] into 6 masks [res, res]."""
    h, w = resolution, resolution
    m0 = mask_np[0:h, 0:w]
    m1 = mask_np[0:h, w:2*w]
    m3 = mask_np[0:h, 2*w:3*w]
    m2 = mask_np[h:2*h, 0:w]
    m4 = mask_np[h:2*h, w:2*w]
    m5 = mask_np[h:2*h, 2*w:3*w]
    return [m0, m1, m2, m3, m4, m5]


# ============================================================
# VAE encode/decode (grid)
# ============================================================

def encode_grid(img, vae, rows=2, cols=3):
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
# Background latent
# ============================================================

def init_black_latent(vae, device, transform, resolution):
    black = Image.new("RGB", (resolution, resolution), color=(0, 0, 0))
    x = transform(black).unsqueeze(0).to(device, torch.float32)
    with torch.no_grad():
        z = vae.encode(x)
    return z


def build_bg_latent(base_latent, resolution):
    H_lat = (2 * resolution) // 8
    W_lat = (3 * resolution) // 8
    h0, w0 = base_latent.shape[-2:]
    return base_latent.repeat(1, 1, H_lat // h0, W_lat // w0)


def mix_latent(img_latent, bg_latent, masks_noise, height, width):
    img_tok = rearrange(img_latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    bg_tok = rearrange(bg_latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    idx = (masks_noise.reshape(-1) != 0).nonzero(as_tuple=True)[0]
    mixed = bg_tok.clone()
    mixed[:, idx] = img_tok[:, idx]
    return unpack(mixed, height, width)


# ============================================================
# Dataset (simplified loader for eval)
# ============================================================

class EvalAlbedoDataset(Dataset):
    def __init__(self, json_file, resolution, kernel_pad=2):
        with open(json_file, "r") as f:
            self.data_list = json.load(f)
        self.resolution = resolution
        self.kernel_pad = kernel_pad
        self.transform = Compose([ToTensor(), Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_dict = self.data_list[idx]
        image_dir = data_dict["image_dir"]
        sample_id = data_dict["id"]

        paths = [os.path.join(image_dir, "albedo", f"{i:03d}.webp") for i in range(6)]

        imgs_big = []
        masks_full = []
        masks_small = []

        for p in paths:
            img = Image.open(p).convert("RGBA")

            # small mask for latent space
            w_small = h_small = self.resolution // 16
            img_small = img.resize((w_small, h_small), Image.Resampling.BILINEAR)
            alpha_small = np.array(img_small.split()[3])
            mask_small = (alpha_small > 0).astype(np.uint8)
            kernel = np.ones((self.kernel_pad, self.kernel_pad), np.uint8)
            mask_small = cv2.dilate(mask_small, kernel, iterations=1)
            masks_small.append(mask_small)

            # full resolution image (white bg)
            img_big = img.resize((self.resolution, self.resolution), Image.Resampling.BILINEAR)
            bg = Image.new("RGB", img_big.size, (255, 255, 255))
            bg.paste(img_big, mask=img_big.split()[3])
            imgs_big.append(np.array(bg))

            # full resolution mask
            alpha_full = np.array(img_big.split()[3])
            mask_full = (alpha_full > 0).astype(np.uint8)
            masks_full.append(mask_full)

        # Build 2x3 grid
        row1_img = np.concatenate([imgs_big[0], imgs_big[1], imgs_big[3]], axis=1)
        row2_img = np.concatenate([imgs_big[2], imgs_big[4], imgs_big[5]], axis=1)
        grid_img = Image.fromarray(np.concatenate([row1_img, row2_img], axis=0))

        row1_ms = np.concatenate([masks_small[0], masks_small[1], masks_small[3]], axis=1)
        row2_ms = np.concatenate([masks_small[2], masks_small[4], masks_small[5]], axis=1)
        grid_mask_small = np.concatenate([row1_ms, row2_ms], axis=0)

        row1_mf = np.concatenate([masks_full[0], masks_full[1], masks_full[3]], axis=1)
        row2_mf = np.concatenate([masks_full[2], masks_full[4], masks_full[5]], axis=1)
        grid_mask_full = np.concatenate([row1_mf, row2_mf], axis=0)

        img_tensor = self.transform(grid_img)  # [3, 2*res, 3*res]
        masks_noise_tensor = torch.from_numpy(grid_mask_small).float().reshape(-1)

        return {
            "id": sample_id,
            "img_tensor": img_tensor,
            "masks_noise": masks_noise_tensor,
            "masks_full": grid_mask_full,            # [2*res, 3*res] uint8 numpy
            "gt_views": np.stack(imgs_big, axis=0),  # [6, res, res, 3] uint8
            "mask_views": np.stack(masks_full, axis=0),  # [6, res, res] uint8
        }


# ============================================================
# Metrics (full resolution, white bg)
# ============================================================

def compute_image_psnr(gt, pred):
    """gt, pred: [H,W,3] uint8, full resolution with white bg."""
    return float(compute_psnr(gt, pred, data_range=255))


def compute_image_ssim(gt, pred):
    """gt, pred: [H,W,3] uint8, full resolution with white bg."""
    return float(compute_ssim(gt, pred, channel_axis=2, data_range=255))


def compute_image_lpips(gt, pred, lpips_fn, device):
    """gt, pred: [H,W,3] uint8, full resolution with white bg."""
    gt_t = torch.from_numpy(gt).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
    pred_t = torch.from_numpy(pred).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
    with torch.no_grad():
        val = lpips_fn(gt_t.to(device), pred_t.to(device))
    return val.item()


def apply_white_bg(pred_uint8, mask):
    """Set background pixels to white."""
    out = np.ones_like(pred_uint8) * 255
    out[mask] = pred_uint8[mask]
    return out


# ============================================================
# Main
# ============================================================

def main(args):
    accelerator = Accelerator()
    device = accelerator.device
    is_main = accelerator.is_local_main_process

    # Load eval json
    with open(args.eval_json, "r") as f:
        data_list = json.load(f)

    if args.max_samples > 0:
        data_list = data_list[:args.max_samples]

    if is_main:
        print(f"Eval samples: {len(data_list)}, Resolution: {args.resolution}")
        print(f"Decoder ckpt: {args.decoder_ckpt}")
        print(f"Num GPUs: {accelerator.num_processes}")

    # Dataset & DataLoader
    dataset = EvalAlbedoDataset(args.eval_json, args.resolution)
    if args.max_samples > 0:
        dataset.data_list = dataset.data_list[:args.max_samples]

    sampler = None
    if accelerator.num_processes > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=False,
            drop_last=False,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=4,
        pin_memory=False,
    )

    # Load VAE
    vae = load_ae("flux-dev", device=device)
    vae.requires_grad_(False)
    vae.eval()

    # Save original decoder, load finetuned
    orig_decoder_state = copy.deepcopy(vae.decoder.state_dict())

    ft_state = torch.load(args.decoder_ckpt, map_location="cpu")
    missing, unexpected = vae.decoder.load_state_dict(ft_state, strict=False)
    if is_main:
        print(f"  Missing keys: {missing}")
        print(f"  Unexpected keys: {unexpected}")
    ft_decoder_state = copy.deepcopy(vae.decoder.state_dict())

    # Restore original
    vae.decoder.load_state_dict(orig_decoder_state)

    # Black background latent
    transform = Compose([ToTensor(), Normalize([0.5], [0.5])])
    black_latent = init_black_latent(vae, device, transform, args.resolution)

    # LPIPS
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    # Output dir
    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Metrics accumulators (per GPU)
    metrics = {
        "pred1": {"psnr": [], "ssim": [], "lpips": []},
        "pred2": {"psnr": [], "ssim": [], "lpips": []},
        "pred3": {"psnr": [], "ssim": [], "lpips": []},
        "pred4": {"psnr": [], "ssim": [], "lpips": []},
    }
    per_sample_results = []  # per-sample metrics

    def _to_uint8(t):
        if t.dim() == 4:
            t = t[0]
        return ((t + 1) * 127.5).clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    for batch_idx, batch in enumerate(tqdm(dataloader, disable=not is_main, desc="Evaluating")):
        sample_id = batch["id"][0]
        img_tensor = batch["img_tensor"].to(device)       # [1, 3, 2*res, 3*res]
        masks_noise = batch["masks_noise"][0].to(device)  # [N]
        gt_views = batch["gt_views"][0].numpy()           # [6, res, res, 3]
        mask_views = batch["mask_views"][0].numpy()       # [6, res, res]

        with torch.no_grad():
            # Encode
            z = encode_grid(img_tensor, vae)
            bg_black = build_bg_latent(black_latent, args.resolution)

            # Mixed latents
            mixed_black = mix_latent(z, bg_black, masks_noise,
                                     args.resolution * 2, args.resolution * 3)
            bg_noise = torch.randn_like(bg_black)
            mixed_noise = mix_latent(z, bg_noise, masks_noise,
                                     args.resolution * 2, args.resolution * 3)

            # pred1: orig decoder + noise bg
            vae.decoder.load_state_dict(orig_decoder_state)
            pred1 = decode_grid(vae, mixed_noise)

            # pred2: orig decoder + black bg
            pred2 = decode_grid(vae, mixed_black)

            # pred3: finetuned decoder + black bg
            vae.decoder.load_state_dict(ft_decoder_state)
            pred3 = decode_grid(vae, mixed_black)

            # pred4: orig decoder, direct encode-decode
            vae.decoder.load_state_dict(orig_decoder_state)
            pred4 = decode_grid(vae, z)

        p1_grid = _to_uint8(pred1)
        p2_grid = _to_uint8(pred2)
        p3_grid = _to_uint8(pred3)
        p4_grid = _to_uint8(pred4)

        # Split to 6 views
        p1_views = split_grid_to_views(p1_grid, args.resolution)
        p2_views = split_grid_to_views(p2_grid, args.resolution)
        p3_views = split_grid_to_views(p3_grid, args.resolution)
        p4_views = split_grid_to_views(p4_grid, args.resolution)

        # Per-view metrics
        sample_metrics = {"id": sample_id, "views": []}
        for vi in range(6):
            gt_v = gt_views[vi]
            mask_v = mask_views[vi].astype(bool)

            # Apply white bg for all predictions
            p1_v = apply_white_bg(p1_views[vi], mask_v)
            p2_v = apply_white_bg(p2_views[vi], mask_v)
            p3_v = apply_white_bg(p3_views[vi], mask_v)
            p4_v = apply_white_bg(p4_views[vi], mask_v)

            view_metrics = {"view": vi}
            for name, pv in [("pred1", p1_v), ("pred2", p2_v), ("pred3", p3_v), ("pred4", p4_v)]:
                psnr_val = compute_image_psnr(gt_v, pv)
                ssim_val = compute_image_ssim(gt_v, pv)
                lpips_val = compute_image_lpips(gt_v, pv, lpips_fn, device)
                metrics[name]["psnr"].append(psnr_val)
                metrics[name]["ssim"].append(ssim_val)
                metrics[name]["lpips"].append(lpips_val)
                view_metrics[name] = {"psnr": round(psnr_val, 4), "ssim": round(ssim_val, 4), "lpips": round(lpips_val, 4)}
            sample_metrics["views"].append(view_metrics)

        # Per-sample average
        sample_avg = {"id": sample_id}
        for name in ["pred1", "pred2", "pred3", "pred4"]:
            vals = sample_metrics["views"]
            sample_avg[name] = {
                "psnr": round(np.mean([v[name]["psnr"] for v in vals]), 4),
                "ssim": round(np.mean([v[name]["ssim"] for v in vals]), 4),
                "lpips": round(np.mean([v[name]["lpips"] for v in vals]), 4),
            }
        sample_avg["views"] = sample_metrics["views"]
        per_sample_results.append(sample_avg)

        # Save visualization (like ablation script)
        if args.save_vis:
            all_dir = os.path.join(args.out_dir, "all")
            single_dir = os.path.join(args.out_dir, "single")
            os.makedirs(all_dir, exist_ok=True)
            os.makedirs(single_dir, exist_ok=True)

            # Apply white bg to full grids
            grid_mask_full = mask_views  # [6, res, res]
            # Rebuild grid masks for full grid white bg
            mask_grid = batch["masks_full"][0].numpy().astype(bool)  # [2*res, 3*res]

            p1_wb = apply_white_bg(p1_grid, mask_grid)
            p2_wb = apply_white_bg(p2_grid, mask_grid)
            p3_wb = apply_white_bg(p3_grid, mask_grid)
            p4_wb = apply_white_bg(p4_grid, mask_grid)

            # Reconstruct GT grid
            gt_grid_r1 = np.concatenate([gt_views[0], gt_views[1], gt_views[3]], axis=1)
            gt_grid_r2 = np.concatenate([gt_views[2], gt_views[4], gt_views[5]], axis=1)
            gt_grid = np.concatenate([gt_grid_r1, gt_grid_r2], axis=0)

            # Full grid: GT | pred4 | pred1 | pred2 | pred3
            img_all = np.concatenate([gt_grid, p4_wb, p1_wb, p2_wb, p3_wb], axis=1)
            fname = f"{sample_id}.png"
            Image.fromarray(img_all).save(os.path.join(all_dir, fname))

            # Single view (first tile, top-left): GT | pred4 | pred1 | pred2 | pred3
            res = args.resolution
            img_single = np.concatenate([
                gt_views[0],
                p4_views[0],
                apply_white_bg(p1_views[0], mask_views[0].astype(bool)),
                apply_white_bg(p2_views[0], mask_views[0].astype(bool)),
                apply_white_bg(p3_views[0], mask_views[0].astype(bool)),
            ], axis=1)
            Image.fromarray(img_single).save(os.path.join(single_dir, fname))

    # ========================================================
    # Gather metrics from all GPUs
    # ========================================================
    accelerator.wait_for_everyone()

    # Convert to tensors for gathering
    all_metrics = {}
    for method in ["pred1", "pred2", "pred3", "pred4"]:
        for metric_name in ["psnr", "ssim", "lpips"]:
            vals = torch.tensor(metrics[method][metric_name], device=device)
            gathered = accelerator.gather(vals)
            if method not in all_metrics:
                all_metrics[method] = {}
            all_metrics[method][metric_name] = gathered.cpu().numpy()

    # ========================================================
    # Print results (main process only)
    # ========================================================
    if is_main:
        n_measurements = len(all_metrics["pred1"]["psnr"])
        print("\n" + "=" * 60)
        print(f"Results ({n_measurements} view measurements)")
        print("=" * 60)

        labels = {
            "pred1": "Orig VAE + Noise BG",
            "pred2": "Orig VAE + Black BG",
            "pred3": "Finetuned VAE + Black BG",
            "pred4": "Orig VAE direct recon (upper bound)",
        }

        results = {}
        for method in ["pred4", "pred1", "pred2", "pred3"]:
            psnr_mean = float(np.mean(all_metrics[method]["psnr"]))
            ssim_mean = float(np.mean(all_metrics[method]["ssim"]))
            lpips_mean = float(np.mean(all_metrics[method]["lpips"]))
            results[method] = {
                "PSNR": round(psnr_mean, 4),
                "SSIM": round(ssim_mean, 4),
                "LPIPS": round(lpips_mean, 4),
            }
            print(f"\n  [{method}] {labels[method]}")
            print(f"    PSNR  ↑: {psnr_mean:.4f}")
            print(f"    SSIM  ↑: {ssim_mean:.4f}")
            print(f"    LPIPS ↓: {lpips_mean:.4f}")

        # Save summary
        results_path = os.path.join(args.out_dir, "metrics.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSummary saved to: {results_path}")

    # Gather per-sample results from all GPUs
    accelerator.wait_for_everyone()
    if accelerator.num_processes > 1:
        all_per_sample = [None] * accelerator.num_processes
        torch.distributed.all_gather_object(all_per_sample, per_sample_results)
        merged = []
        for ps in all_per_sample:
            merged.extend(ps)
    else:
        merged = per_sample_results

    if is_main:
        per_sample_path = os.path.join(args.out_dir, "metrics_per_sample.json")
        with open(per_sample_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"Per-sample saved to: {per_sample_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_json", type=str,
                        default="data/eval.json")
    parser.add_argument("--decoder_ckpt", type=str,
                        default="checkpoints/flux1/decoder.pt")
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max number of samples to evaluate. -1 means all.")
    parser.add_argument("--out_dir", type=str,
                        default="./train_vae_log/eval_metrics")
    parser.add_argument("--save_vis", action="store_true",
                        help="Save visualization images (GT|pred4|pred1|pred2|pred3)")
    args = parser.parse_args()
    main(args)

# Usage (8 GPUs):
# accelerate launch --num_processes 8 train_vae/eval_metrics.py
# accelerate launch --num_processes 8 train_vae/eval_metrics.py --max_samples 50
