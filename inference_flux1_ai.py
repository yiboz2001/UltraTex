# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from typing import Literal
import dataclasses
import gc
import itertools
import logging
import os
import re
import random
from copy import deepcopy
from typing import TYPE_CHECKING, Literal
os.environ["WANDB_MODE"]="offline"
os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "180000"
import time
import datetime
import json
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed, send_to_device
from diffusers.optimization import get_scheduler
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from ultratex.data.dataset_ai import FluxPairedDatasetAI
from ultratex.backbones.flux1.sampling import denoise, denoise_test, get_noise, get_schedule, prepare_multi_ip, unpack
from ultratex.backbones.flux1.util import load_ae, load_clip, load_flow_model, load_t5, set_lora

if TYPE_CHECKING:
    from ultratex.backbones.flux1.model import Flux
    from ultratex.backbones.flux1.modules.autoencoder import AutoEncoder
    from ultratex.backbones.flux1.modules.conditioner import HFEmbedder

log_dir = "outputs/profile" 
logger = get_logger(__name__)

import os
from PIL import Image
import numpy as np

def load_and_concat(image_dir, subdir, resolution):
    """Load 000-005.webp from dir/subdir, composite on white, tile into a 2x3 grid."""
    paths = [os.path.join(image_dir, subdir, f"{i:03d}.webp") for i in range(6)]
    imgs = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")
        
        img = Image.open(p).convert("RGBA")  # normalise to RGBA so alpha is available
        img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
        
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])  # composite using the alpha channel
        imgs.append(bg)

    arrays = [np.array(img, dtype=np.uint8) for img in imgs]
    row1 = np.concatenate([arrays[0], arrays[1], arrays[3]], axis=1)
    row2 = np.concatenate([arrays[2], arrays[4], arrays[5]], axis=1)
    concat_img = np.concatenate([row1, row2], axis=0)
    return Image.fromarray(concat_img)

def make_eval_composite(image_dir, result_img, render_path, resolution):
    """
    Layout:
      render occupies the left column (rows 1 and 3, first cell)
      right two columns: position and bump_normal, each a 2x3 tile
      result_img sits bottom-right, centred
    """
    # load the three base tiles
    pos_img = load_and_concat(image_dir, "position", resolution)
    # The renderer writes world-space normals as "bump_normal_recover" and then
    # renames the directory to "bump_normal_world"; released data only has the
    # latter (see render_single_recover_random.py in the rendering pipeline).
    bump_img = load_and_concat(image_dir, "bump_normal_world", resolution)
    albedo_img = load_and_concat(image_dir, "albedo", resolution)
    render_img = Image.open(render_path).convert("RGBA").resize((resolution, resolution))
    
    bg = Image.new("RGB", render_img.size, (255, 255, 255))
    bg.paste(render_img, mask=render_img.split()[3])
    render_img = bg

    # per-tile dimensions
    ref_w, ref_h = pos_img.size
    cell_w = render_img.width
    cell_h = render_img.height

    # overall canvas: 6x4 grid
    total_w = cell_w * 7
    total_h = cell_h * 4
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))

    # === layout ===
    result_resized = result_img.resize((ref_w, ref_h))
    
    canvas.paste(render_img, (0, 2 * cell_h))            

    canvas.paste(pos_img, (cell_w, 0))           
    canvas.paste(bump_img, (cell_w, 2 * cell_h))
    
    canvas.paste(albedo_img, (cell_w * 4, 0))
    canvas.paste(result_resized, (cell_w * 4, 2 * cell_h))

    return canvas


def get_models(name: str, device, offload: bool=False):
    t5 = load_t5(device, max_length=512)
    clip = load_clip(device)
    model = load_flow_model(name, device="cpu")
    vae = load_ae(name, device="cpu" if offload else device)
    return model, vae, t5, clip

def inference(
    batch: dict,
    model: "Flux", t5: "HFEmbedder", clip: "HFEmbedder", ae: "AutoEncoder",
    accelerator: Accelerator,
    seed: int = 0,
    pe: Literal["d", "h", "w", "o"] = "d",
    resolution: int = 0,
    background=None,
    denoise_vis_dir=None,
    denoise_rgb_vis_dir=None,
    baseline_vis_dir=None,
    baseline_mode=False,
) -> Image.Image:
    
    def encode_grid(img, vae, rows=2, cols=3):
        H, W = img.shape[-2:]
        h, w = H // rows, W // cols
        patches = rearrange(img, "b c (r h) (co w) -> (b r co) c h w", r=rows, co=cols)
        z_grid = []
        with torch.no_grad():
            for i, patch in enumerate(patches):
                patch = patch.unsqueeze(0)  # [1, 3, h, w]
                z = vae.encode(patch)
                z_grid.append(z)
        
        z_grid = torch.cat(z_grid, dim=0)
        z_grid = rearrange(z_grid, "(b r co) c h w -> b c (r h) (co w)", b=1, r=rows, co=cols)
        del patches
        torch.cuda.empty_cache()
        return z_grid
    
    ref_imgs = batch["ref_imgs"]
    masks = batch["masks"]
    masks_full = batch["masks_full"][0][0]
    prompt = batch["txt"]
    neg_prompt = ''
    num_steps = 25

    def decode_packed_latent(packed):
        decoded = unpack(packed.float(), height, width)
        rows = torch.chunk(decoded, 2, dim=-2)
        return torch.cat(
            [torch.cat([ae.decode(c) for c in torch.chunk(row, 3, dim=-1)], dim=-1)
             for row in rows],
            dim=-2,
        )

    def save_denoise_step(step, packed, output_dir=denoise_vis_dir):
        if output_dir is None:
            return
        with torch.no_grad():
            decoded = decode_packed_latent(packed)
            vis = decoded[0].clamp(-1, 1)
            vis = rearrange(vis, "c h w -> h w c")
            vis = (127.5 * (vis + 1.0)).cpu().numpy().astype(np.uint8)
            Image.fromarray(vis).save(os.path.join(output_dir, f"step_{step:02d}.png"))

    initial_noise_rgb = None

    def save_rgb_composite_step(step, packed):
        nonlocal initial_noise_rgb
        if denoise_rgb_vis_dir is None:
            return
        with torch.no_grad():
            if initial_noise_rgb is None:
                initial_noise = decode_packed_latent(packed)
                initial_noise = initial_noise[0].clamp(-1, 1)
                initial_noise = rearrange(initial_noise, "c h w -> h w c")
                initial_noise_rgb = (127.5 * (initial_noise + 1.0)).cpu().numpy().astype(np.uint8)

            if step == 0:
                Image.fromarray(initial_noise_rgb).save(
                    os.path.join(denoise_rgb_vis_dir, "step_00.png")
                )
                return

            foreground_latent = background.clone()
            masks_noise = torch.cat([masks[0][0].reshape(-1)])
            mask_noise_idx = masks_noise.nonzero(as_tuple=True)[0]
            foreground_latent[:, mask_noise_idx] = packed[:, mask_noise_idx]
            foreground = decode_packed_latent(foreground_latent)
            foreground = foreground[0].clamp(-1, 1)
            foreground = rearrange(foreground, "c h w -> h w c")
            foreground = (127.5 * (foreground + 1.0)).cpu().numpy().astype(np.uint8)

            foreground_mask = masks_full.cpu().numpy().astype(bool)
            composite = initial_noise_rgb.copy()
            composite[foreground_mask] = foreground[foreground_mask]
            Image.fromarray(composite).save(
                os.path.join(denoise_rgb_vis_dir, f"step_{step:02d}.png")
            )
    
    height = resolution * 2
    width = resolution * 3
    x = get_noise(
        1, height, width,
        device=accelerator.device,
        dtype=torch.bfloat16,
        seed=seed + accelerator.process_index
    )
    timesteps = get_schedule(
        num_steps,
        (width // 8) * (height // 8) // (16 * 16),
        shift=True,
    )
    callback_time = 0.0

    def timed_callback(callback):
        def wrapped(step, packed):
            nonlocal callback_time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            callback(step, packed)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            callback_time += time.perf_counter() - start
        return wrapped

    baseline_callback = None
    if baseline_vis_dir is not None:
        baseline_callback = timed_callback(
            lambda step, packed: save_denoise_step(step, packed, baseline_vis_dir)
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_start = time.perf_counter()

    with torch.no_grad():
        x_ref = []
        for ref_img_ in ref_imgs:
            H, W = ref_img_.shape[-2:]
            if H == W:
                x_ref.append(ae.encode(ref_img_.to(accelerator.device).to(torch.float32)).to(torch.bfloat16))
            else:
                x_ref.append(encode_grid(ref_img_.to(accelerator.device, torch.float32), ae).to(torch.bfloat16))

        # ref_imgs = [
        #     encode_grid(ref_img_.to(accelerator.device, torch.float32), ae).to(torch.bfloat16)
        #     for ref_img_ in ref_imgs
        # ]
        
        inp_cond = prepare_multi_ip(
            t5=t5, clip=clip, img=x, prompt=prompt,
            ref_imgs=x_ref,
            pe=pe
        )
        if background is not None:
            background = rearrange(background, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        x = denoise_test(
            model,
            **inp_cond,
            timesteps=timesteps,
            masks=masks,
            guidance=4,
            background=background,
            step_callback=baseline_callback if baseline_mode else (timed_callback(save_denoise_step) if denoise_vis_dir else None),
            step_callback_rgb=timed_callback(save_rgb_composite_step) if denoise_rgb_vis_dir and not baseline_mode else None,
            update_all=baseline_mode,
        )
        x = decode_packed_latent(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_elapsed = time.perf_counter() - inference_start - callback_time
    x1 = x.clamp(-1, 1)
    x1 = rearrange(x1[-1], "c h w -> h w c")   # [-1,1] → H W C
    img_np = (127.5 * (x1 + 1.0)).cpu().numpy().astype(np.uint8)   # scale to 0-255
    mask = (masks_full == 0).cpu().numpy()               # 0 = background
    mask = np.repeat(mask[..., None],3,axis=2)   # H W -> H W C, broadcast to 3 channels
    if not baseline_mode:
        img_np[mask] = 255
    output_img = Image.fromarray(img_np)
    return output_img, inference_elapsed

start_step = 0
def resume_from_checkpoint(
    resume_from_checkpoint: str | None | Literal["latest"],
    project_dir: str,
    accelerator: Accelerator,
    dit: "Flux",
    dit_ema_dict: dict | None = None,
) -> tuple["Flux", torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, dict | None, int]:
    
    global start_step
    if resume_from_checkpoint is None:
        return dit, dit_ema_dict, 0

    name = os.path.basename(resume_from_checkpoint)
    path = resume_from_checkpoint

    accelerator.print(f"Resuming from checkpoint {path}")
    lora_state = load_file(
        os.path.join(path, 'dit_lora.safetensors'),
        device=accelerator.device.__str__()
    )
    unwarp_dit = accelerator.unwrap_model(dit)
    unwarp_dit.load_state_dict(lora_state, strict=False)
    if dit_ema_dict is not None:
        dit_ema_dict = load_file(
            os.path.join(path, 'dit_lora_ema.safetensors'),
            device=accelerator.device.__str__()
        )
        if dit is not unwarp_dit:
            dit_ema_dict = {f"module.{k}": v for k, v in dit_ema_dict.items() if k in unwarp_dit.state_dict()}

    global_step = 0
    # A released LoRA folder (e.g. checkpoints/flux1/lora) has no step suffix.
    _m = re.fullmatch(r"checkpoint-(\d+)", name)
    start_step = int(_m.group(1)) if _m else 0
    
    return dit, dit_ema_dict, global_step

@dataclasses.dataclass
class TrainArgs:
    ## accelerator
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    gradient_accumulation_steps: int = 1
    seed: int = 42
    wandb_project_name: str | None = None
    wandb_run_name: str | None = None

    ## model
    model_name: Literal["flux-dev", "flux-schnell"] = "flux-dev"
    lora_rank: int = 64
    double_blocks_indices: list[int] | None = dataclasses.field(
        default=None,
        metadata={"help": "Indices of double blocks to apply LoRA. None means all double blocks."}
    )
    single_blocks_indices: list[int] | None = dataclasses.field(
        default=None,
        metadata={"help": "Indices of double blocks to apply LoRA. None means all single blocks."}
    )
    pe: Literal["d", "h", "w", "o"] = "d"
    gradient_checkpoint: bool = True
    ema: bool = False
    ema_interval: int = 1
    ema_decay: float = 0.99

    ## optimizer
    learning_rate: float = 1e-4
    adam_betas: list[float] = dataclasses.field(default_factory=lambda: [0.9, 0.999])
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    ## lr_scheduler
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 1

    ## dataloader
    train_data_json: str = "data/train.json"
    batch_size: int = 1
    text_dropout: float = 0.1
    resolution: int = 2048
    resolution_ref: int | None = None

    # eval_data_json: str = "data/ai_generated"
    eval_data_json: str = "data/sketchfab"

    eval_batch_size: int = 1

    decoder_ckpt: str =  "checkpoints/flux1/decoder.pt"

    denoise_vis_dir: str | None = None
    denoise_rgb_vis_dir: str | None = None
    baseline_vis_dir: str | None = None
    baseline_mode: bool = False
    timing_json: str | None = None
    benchmark_repeats: int = 0
        
    ## misc
    resume_from_checkpoint: str | None | Literal["latest"] = None
    checkpointing_steps: int = 500
    project_dir: str | None = None
    
    def __post_init__(self):
        if self.project_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.project_dir = (
                f"outputs/inference_flux1_ai_res={self.resolution}_lr={self.learning_rate}-lora"
            )

def main(
    args: TrainArgs,
):
    ## accelerator
    deepspeed_plugins = {
        "dit": DeepSpeedPlugin(hf_ds_config='config/deepspeed/zero2_config.json'),
        "t5": DeepSpeedPlugin(hf_ds_config='config/deepspeed/zero3_config.json'),
        "clip": DeepSpeedPlugin(hf_ds_config='config/deepspeed/zero3_config.json')
    }
    accelerator = Accelerator(
        project_dir=args.project_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        deepspeed_plugins=deepspeed_plugins,
        log_with="tensorboard",  
    )
    set_seed(args.seed, device_specific=True)
    accelerator.init_trackers(
        project_name="",
    )
    weight_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "no": torch.float32,
    }.get(accelerator.mixed_precision, torch.float32)

    ## logger
    logging.basicConfig(
        format=f"[RANK {accelerator.process_index}] " + "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        force=True
    )
    logger.info(accelerator.state)
    logger.info("Training script launched", main_process_only=False)

    ## model
    dit, vae, t5, clip = get_models(
        name=args.model_name,
        device=accelerator.device,
    )
    
    vae.requires_grad_(False)
    t5.requires_grad_(False)
    clip.requires_grad_(False)

    state = torch.load(args.decoder_ckpt, map_location="cpu")
    missing, unexpected = vae.decoder.load_state_dict(state, strict=False)
    
    dit.requires_grad_(False)
    dit = set_lora(dit, args.lora_rank, args.double_blocks_indices, args.single_blocks_indices, accelerator.device)
    dit.train()
    dit.gradient_checkpointing = args.gradient_checkpoint
    
    ## ema
    dit_ema_dict = {
        f"module.{k}": deepcopy(v).requires_grad_(False) for k, v in dit.named_parameters() if v.requires_grad
    } if args.ema else None


    ## optimizer and lr scheduler
    optimizer = torch.optim.AdamW(
        [p for p in dit.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=args.adam_betas,
        weight_decay=args.adam_weight_decay,
        eps=args.adam_eps,
    )
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    ## resume
    (
        dit,
        dit_ema_dict,
        global_step
    ) = resume_from_checkpoint(
        args.resume_from_checkpoint,
        project_dir=args.project_dir,
        accelerator=accelerator,
        dit=dit,
        dit_ema_dict=dit_ema_dict
    )

    eval_dataset = FluxPairedDatasetAI(
        json_file=args.eval_data_json,
        resolution=args.resolution, resolution_ref=args.resolution_ref
    )   
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=eval_dataset.collate_fn
    )

    eval_dataloader = accelerator.prepare_data_loader(eval_dataloader)

    ## parallel
    accelerator.state.select_deepspeed_plugin("dit")
    dit, optimizer, lr_scheduler = accelerator.prepare(dit, optimizer, lr_scheduler) 
    accelerator.state.select_deepspeed_plugin("t5")
    t5 = accelerator.prepare(t5)  # type: torch.nn.Module
    accelerator.state.select_deepspeed_plugin("clip")
    clip = accelerator.prepare(clip)  # type: torch.nn.Module

    bg_pil = Image.new("RGB", (args.resolution, args.resolution), color=(0, 0, 0))
    bg_tensor = eval_dataset.transform(bg_pil).unsqueeze(0).to(accelerator.device, torch.float32) 

    with torch.no_grad():
        bg_latent = vae.encode(bg_tensor).to(torch.bfloat16).repeat(1, 1, 2, 3)

    save_path = os.path.join(args.project_dir, f"checkpoint-{global_step}")
    os.makedirs(save_path, exist_ok=True)
    
    dit.eval()
    torch.set_grad_enabled(False)
    rank = accelerator.process_index
    local_len = len(eval_dataloader)  

    if args.benchmark_repeats > 0:
        if accelerator.num_processes != 1:
            raise ValueError("Benchmark requires one process")
        measurements = []
        for repetition in range(args.benchmark_repeats + 1):
            torch.cuda.synchronize()
            started = time.perf_counter()
            benchmark_batch = eval_dataset.collate_fn([eval_dataset[0]])
            benchmark_batch = send_to_device(benchmark_batch, accelerator.device)
            benchmark_background = None
            if not args.baseline_mode:
                background_image = Image.new("RGB", (args.resolution, args.resolution), (0, 0, 0))
                background_tensor = eval_dataset.transform(background_image).unsqueeze(0).to(accelerator.device, torch.float32)
                benchmark_background = vae.encode(background_tensor).to(torch.bfloat16).repeat(1, 1, 2, 3)
            benchmark_result, _ = inference(
                benchmark_batch, dit, t5, clip, vae, accelerator,
                seed=0, resolution=args.resolution, background=benchmark_background,
                baseline_mode=args.baseline_mode,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            print(f"BENCHMARK repetition={repetition} warmup={repetition == 0} seconds={elapsed:.6f}", flush=True)
            if repetition:
                measurements.append(elapsed)
        report = {
            "mode": "baseline" if args.baseline_mode else "ours",
            "id": benchmark_batch["id"][0],
            "seconds": measurements,
            "mean_seconds": sum(measurements) / len(measurements),
            "warmup_runs": 1,
            "num_steps": 25,
            "resolution": args.resolution,
            "gpu": torch.cuda.get_device_name(),
            "scope": "Warm model: data loading/preprocessing, background preparation when used, noise, reference/text encoding, denoising, final VAE decode and RGB/PIL conversion. No visualization callbacks or output writes. Model loading excluded.",
            "checkpoint": args.resume_from_checkpoint,
        }
        with open(args.timing_json or os.path.join(args.project_dir, "benchmark.json"), "w") as handle:
            json.dump(report, handle, indent=2)
        benchmark_result.save(os.path.join(args.project_dir, "benchmark_final.png"))
        return

    for local_i, batch in enumerate(eval_dataloader):
        image_id = batch["id"][0]
        step_vis_dir = None
        step_rgb_vis_dir = None
        baseline_step_vis_dir = None
        if args.denoise_vis_dir is not None:
            step_vis_dir = os.path.join(args.denoise_vis_dir, str(image_id))
            os.makedirs(step_vis_dir, exist_ok=True)
        if args.denoise_rgb_vis_dir is not None:
            step_rgb_vis_dir = os.path.join(args.denoise_rgb_vis_dir, str(image_id))
            os.makedirs(step_rgb_vis_dir, exist_ok=True)
        if args.baseline_vis_dir is not None:
            baseline_step_vis_dir = os.path.join(args.baseline_vis_dir, str(image_id))
            os.makedirs(baseline_step_vis_dir, exist_ok=True)
        result, inference_elapsed = inference(
            batch, dit, t5, clip, vae, accelerator, seed=0,
            resolution=args.resolution, background=bg_latent,
            denoise_vis_dir=step_vis_dir,
            denoise_rgb_vis_dir=step_rgb_vis_dir,
            baseline_vis_dir=baseline_step_vis_dir,
            baseline_mode=args.baseline_mode,
        )
        image_dir = batch["image_dir"][0] if isinstance(batch["image_dir"], list) else batch["image_dir"]
        render_path = batch["render_path"][0] if isinstance(batch["render_path"], list) else batch["render_path"]
        id_ = batch["id"][0]
        global_i = rank * local_len + local_i
        composite = make_eval_composite(image_dir, result, render_path, resolution=args.resolution)
        composite.save(os.path.join(save_path, f"eval_{id_}_{global_i}.png"))
        if args.timing_json is not None:
            timing_path = args.timing_json
            os.makedirs(os.path.dirname(timing_path) or ".", exist_ok=True)
            timing = {
                "id": id_,
                "mode": "baseline" if args.baseline_mode else "ours",
                "num_steps": 25,
                "resolution": args.resolution,
                "elapsed_seconds_excluding_intermediate_saves": inference_elapsed,
                "intermediate_saves_excluded": True,
            }
            with open(timing_path, "w") as f:
                json.dump(timing, f, indent=2)
            print(json.dumps(timing, indent=2))

if __name__ == "__main__":
    parser = transformers.HfArgumentParser([TrainArgs])
    args_tuple = parser.parse_args_into_dataclasses(args_file_flag="--config")
    main(*args_tuple)

# nohup accelerate launch inference_ai-generated.py &
