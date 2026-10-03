import argparse
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ultratex.backbones.flux2.util import load_ae
from ultratex.data.dataset import FluxPairedDatasetV2
from train_decoder_flux2 import (
    BaseLatentModule,
    decode_grid,
    encode_grid,
    init_base_latent_tensor,
    mean_deltaE2000,
    mix_latent,
)


def main(args):
    device = torch.device(args.device)
    if not os.path.isfile(args.decoder_ckpt):
        raise FileNotFoundError(args.decoder_ckpt)
    if not os.path.isfile(args.ae_model_path):
        raise FileNotFoundError(args.ae_model_path)

    os.environ["AE_MODEL_PATH"] = args.ae_model_path
    dataset = FluxPairedDatasetV2(args.eval_json, args.resolution, None)
    if not dataset.data_dicts:
        raise RuntimeError(f"No valid evaluation objects in {args.eval_json}")

    vae = load_ae(args.model_name, device=device)
    state = torch.load(args.decoder_ckpt, map_location="cpu", weights_only=True)
    result = vae.decoder.load_state_dict(state, strict=True)
    print(f"Loaded decoder: missing={result.missing_keys}, unexpected={result.unexpected_keys}")

    vae.requires_grad_(False)
    vae.to(dtype=torch.bfloat16)
    vae.eval()

    black_latent = init_base_latent_tensor(
        vae,
        device,
        dataset.transform,
        pad=args.resolution,
        dtype=torch.bfloat16,
    )
    bg_module = BaseLatentModule(black_latent).to(device)

    all_dir = os.path.join(args.out_dir, "all")
    single_dir = os.path.join(args.out_dir, "single")
    os.makedirs(all_dir, exist_ok=True)
    os.makedirs(single_dir, exist_ok=True)

    # FluxPairedDatasetV2 repeats each object four times for conditioning
    # variants. VAE reconstruction only depends on albedo, so infer once/object.
    object_count = len(dataset.data_dicts)
    if args.max_samples > 0:
        object_count = min(object_count, args.max_samples)

    for object_index in tqdm(range(object_count), desc="FLUX.2 VAE inference"):
        sample = dataset[object_index * 4]
        sample_id = sample["id"]
        img = sample["img"].to(device)
        masks_noise = sample["masks"][0].to(device)
        masks_full = sample["masks_full"][0]

        with torch.inference_mode():
            z = encode_grid(img[None].to(torch.bfloat16), vae)
            bg = bg_module.build_bg_latent(args.resolution).to(torch.bfloat16)
            pred = decode_grid(vae, mix_latent(z, bg, masks_noise))[0]

        pred_np = (
            ((pred + 1) * 127.5)
            .clamp(0, 255)
            .permute(1, 2, 0)
            .float()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        gt_np = (
            ((img + 1) * 127.5)
            .clamp(0, 255)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        foreground = masks_full.cpu().numpy().astype(bool)
        foreground_rgb = np.repeat(foreground[..., None], 3, axis=2)
        pred_np[~foreground_rgb] = 255
        gt_np[~foreground_rgb] = 255
        delta_e = mean_deltaE2000(gt_np, pred_np)
        filename = f"{sample_id}_{delta_e:.4f}.png"

        Image.fromarray(np.concatenate([gt_np, pred_np], axis=1)).save(
            os.path.join(all_dir, filename)
        )

        tile_h = gt_np.shape[0] // 2
        tile_w = gt_np.shape[1] // 3
        single = np.concatenate(
            [gt_np[:tile_h, :tile_w], pred_np[:tile_h, :tile_w]], axis=1
        )
        Image.fromarray(single).save(os.path.join(single_dir, filename))
        print(f"{sample_id}: deltaE2000={delta_e:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decoder_ckpt",
        default="checkpoints/flux2/decoder.pt",
    )
    parser.add_argument(
        "--eval_json",
        default="data/eval_demo.json",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/eval_vae_flux2",
    )
    parser.add_argument(
        "--ae_model_path",
        default=os.environ.get(
            "AE_MODEL_PATH", "checkpoints/base/flux2/ae.safetensors"
        ),
    )
    parser.add_argument("--model_name", default="flux.2-klein-base-9b")
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--device", default="cuda:0")
    main(parser.parse_args())
