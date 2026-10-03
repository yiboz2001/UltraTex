"""UNO TexVerse 2048 roughness/metallic LoRA training ported to FLUX.2.

The data pipeline, masks, optimization hyperparameters, checkpoint cadence and
evaluation layout intentionally follow train_flux2.py.  The
model-facing pieces use the same native FLUX.2 path as UNO_FLUX2: Klein Base,
FLUX.2 AutoEncoder, Qwen3 text features, t-axis reference planes and PEFT LoRA.
"""

import dataclasses
import datetime
import gc
import logging
import math
import os
import re
from copy import deepcopy
from typing import Literal

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("TORCH_DISTRIBUTED_TIMEOUT", "180000")

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from einops import rearrange
from peft import LoraConfig, get_peft_model
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from ultratex.backbones.flux2.sampling import batched_prc_img, batched_prc_txt
from ultratex.backbones.flux2.model import configure_sparse_attention
from ultratex.backbones.flux2.util import load_ae, load_flow_model, load_text_encoder
from ultratex.data.dataset_mr import FluxPairedDatasetMR, LengthAwareBatchSampler

logger = get_logger(__name__)
start_step = 0


def uno_time_shift(mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
    """The timestep shift used by the original UNO/FLUX.1 trainer."""
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def uno_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    """Exact schedule formula from ultratex/backbones/flux1/sampling.py."""
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if shift:
        slope = (max_shift - base_shift) / (4096 - 256)
        intercept = base_shift - slope * 256
        mu = slope * image_seq_len + intercept
        timesteps = uno_time_shift(mu, 1.0, timesteps)
    return timesteps.tolist()


def load_and_concat(image_dir: str, subdir: str, resolution: int) -> Image.Image:
    """Keep the original TexVerse six-view 2x3 layout."""
    paths = [os.path.join(image_dir, subdir, f"{i:03d}.webp") for i in range(6)]
    imgs = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        img = Image.open(path).convert("RGBA").resize(
            (resolution, resolution), Image.Resampling.LANCZOS
        )
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        imgs.append(np.asarray(bg, dtype=np.uint8))

    row1 = np.concatenate([imgs[0], imgs[1], imgs[3]], axis=1)
    row2 = np.concatenate([imgs[2], imgs[4], imgs[5]], axis=1)
    return Image.fromarray(np.concatenate([row1, row2], axis=0))


def make_eval_composite(
    image_dir: str,
    result_img: Image.Image,
    render_path: str,
    resolution: int,
) -> Image.Image:
    """Keep the original MR evaluation canvas unchanged."""
    pos_img = load_and_concat(image_dir, "position", resolution)
    bump_img = load_and_concat(image_dir, "bump_normal_world", resolution)
    mr_img = load_and_concat(image_dir, "roughness_metallic", resolution)
    render_img = Image.open(render_path).convert("RGBA").resize((resolution, resolution))
    bg = Image.new("RGB", render_img.size, (255, 255, 255))
    bg.paste(render_img, mask=render_img.split()[3])
    render_img = bg

    ref_w, ref_h = pos_img.size
    cell_w, cell_h = render_img.size
    canvas = Image.new("RGB", (cell_w * 7, cell_h * 4), (255, 255, 255))
    canvas.paste(render_img, (0, 2 * cell_h))
    canvas.paste(pos_img, (cell_w, 0))
    canvas.paste(bump_img, (cell_w, 2 * cell_h))
    canvas.paste(mr_img, (cell_w * 4, 0))
    canvas.paste(result_img.resize((ref_w, ref_h)), (cell_w * 4, 2 * cell_h))
    return canvas


def get_models(model_name: str, device: torch.device):
    dit = load_flow_model(model_name, device="cpu")
    ae = load_ae(model_name, device=device)
    text_encoder = load_text_encoder(model_name, device=device)
    return dit, ae, text_encoder


def load_decoder_checkpoint(ae, decoder_ckpt: str) -> None:
    """Replace the native FLUX.2 decoder with the fine-tuned 2048 decoder."""
    if not os.path.isfile(decoder_ckpt):
        raise FileNotFoundError(f"FLUX.2 decoder checkpoint not found: {decoder_ckpt}")
    state = torch.load(decoder_ckpt, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(
            f"Expected a decoder state dict in {decoder_ckpt}, got {type(state).__name__}"
        )
    result = ae.decoder.load_state_dict(state, strict=True)
    logger.info(
        "Loaded fine-tuned FLUX.2 decoder from %s (missing=%d, unexpected=%d)",
        decoder_ckpt,
        len(result.missing_keys),
        len(result.unexpected_keys),
    )


@torch.no_grad()
def encode_grid(img: torch.Tensor, ae, rows: int = 2, cols: int = 3) -> torch.Tensor:
    """VAE-encode each TexVerse tile separately, preserving the original grid path."""
    batch_size = img.shape[0]
    patches = rearrange(
        img,
        "b c (r h) (co w) -> (b r co) c h w",
        r=rows,
        co=cols,
    )
    encoded = []
    for patch in patches:
        encoded.append(ae.encode(patch.unsqueeze(0).to(torch.float32)))
    z = torch.cat(encoded, dim=0)
    return rearrange(
        z,
        "(b r co) c h w -> b c (r h) (co w)",
        b=batch_size,
        r=rows,
        co=cols,
    )


@torch.no_grad()
def encode_conditions(batch: dict, ae, text_encoder, device: torch.device):
    """Map the original target/references onto FLUX.2 native token sequences."""
    img = batch["img"].to(device)
    x_1_grid = encode_grid(img, ae)
    x_1, x_ids = batched_prc_img(x_1_grid)

    ref_tokens_list = []
    ref_ids_list = []
    for ref_index, ref_img in enumerate(batch["ref_imgs"]):
        ref_img = ref_img.to(device)
        height, width = ref_img.shape[-2:]
        if height == width:
            ref_grid = ae.encode(ref_img.to(torch.float32))
        else:
            ref_grid = encode_grid(ref_img, ae)
        tokens, ids = batched_prc_img(ref_grid)
        ids = ids.clone()
        ids[..., 0] = 10 * (ref_index + 1)
        ref_tokens_list.append(tokens)
        ref_ids_list.append(ids)

    ref_tokens = torch.cat(ref_tokens_list, dim=1) if ref_tokens_list else None
    ref_ids = torch.cat(ref_ids_list, dim=1) if ref_ids_list else None
    ctx = text_encoder(list(batch["txt"])).to(torch.bfloat16)
    ctx, ctx_ids = batched_prc_txt(ctx)
    ref_token_lengths = [tokens.shape[1] for tokens in ref_tokens_list]
    return (
        x_1,
        x_ids,
        ref_tokens,
        ref_ids,
        ref_token_lengths,
        ctx,
        ctx_ids,
        x_1_grid.shape[-2:],
    )


def _foreground_indices(
    masks: list[torch.Tensor],
    token_count: int,
    device: torch.device,
    label: str,
) -> torch.Tensor:
    """Return batched foreground indices, exactly matching original token pruning.

    The original implementation indexes one sample at a time (`masks[0]`).  The
    training configuration also uses batch size 1 per device.  For completeness
    this supports larger batches when every sample has the same foreground-token
    count; otherwise padding plus an attention mask would change the old logic,
    so fail explicitly instead of silently keeping background tokens.
    """
    flat_masks = torch.stack([mask.reshape(-1) for mask in masks]).to(device).bool()
    if flat_masks.shape[1] != token_count:
        raise ValueError(
            f"{label} mask/token mismatch: mask has {flat_masks.shape[1]} values, "
            f"but FLUX.2 produced {token_count} tokens"
        )

    counts = flat_masks.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError(f"{label} contains a sample with no foreground tokens")
    if not torch.all(counts == counts[0]):
        raise ValueError(
            f"{label} foreground counts differ within a batch ({counts.tolist()}); "
            "use batch_size=1 to preserve exact background-token dropping"
        )
    return torch.stack(
        [mask.nonzero(as_tuple=True)[0] for mask in flat_masks], dim=0
    )


def foreground_token_indices(
    batch: dict,
    target_token_count: int,
    ref_token_lengths: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Build the target/ref `keep_idx` lists used by the original UNO model."""
    expected_masks = 1 + len(ref_token_lengths)
    if any(len(sample_masks) != expected_masks for sample_masks in batch["masks"]):
        raise ValueError(
            f"Expected {expected_masks} masks per sample (target + references)"
        )

    target_indices = _foreground_indices(
        [sample_masks[0] for sample_masks in batch["masks"]],
        target_token_count,
        device,
        "target",
    )
    ref_indices = [
        _foreground_indices(
            [sample_masks[ref_index + 1] for sample_masks in batch["masks"]],
            token_count,
            device,
            f"reference[{ref_index}]",
        )
        for ref_index, token_count in enumerate(ref_token_lengths)
    ]
    return target_indices, ref_indices


def gather_sequence(sequence: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather a variable sequence dimension with one index list per batch item."""
    return sequence.gather(
        1, indices[..., None].expand(-1, -1, sequence.shape[-1])
    )


def prune_reference_tokens(
    ref_tokens: torch.Tensor | None,
    ref_ids: torch.Tensor | None,
    ref_token_lengths: list[int],
    ref_indices: list[torch.Tensor],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Drop background tokens from every reference before Transformer attention."""
    if ref_tokens is None:
        if ref_token_lengths or ref_indices:
            raise ValueError("Reference masks were provided without reference tokens")
        return None, None

    token_chunks = ref_tokens.split(ref_token_lengths, dim=1)
    id_chunks = ref_ids.split(ref_token_lengths, dim=1)
    kept_tokens = [
        gather_sequence(chunk, indices)
        for chunk, indices in zip(token_chunks, ref_indices, strict=True)
    ]
    kept_ids = [
        gather_sequence(chunk, indices)
        for chunk, indices in zip(id_chunks, ref_indices, strict=True)
    ]
    return torch.cat(kept_tokens, dim=1), torch.cat(kept_ids, dim=1)


@torch.no_grad()
def inference(
    batch: dict,
    model,
    text_encoder,
    ae,
    accelerator: Accelerator,
    seed: int = 0,
    pe: Literal["d", "h", "w", "o"] = "d",
    resolution: int = 0,
    background: torch.Tensor | None = None,
    guidance: float = 4.0,
    remove_background: bool = True,
    drop_background_tokens: bool = False,
    num_steps: int = 25,
) -> Image.Image:
    del pe  # FLUX.2 uses native 4-axis IDs.
    (
        x_shape,
        x_ids,
        ref_tokens,
        ref_ids,
        ref_token_lengths,
        ctx,
        ctx_ids,
        latent_hw,
    ) = encode_conditions(batch, ae, text_encoder, accelerator.device)
    generator = torch.Generator(device=accelerator.device).manual_seed(
        seed + accelerator.process_index
    )
    x = torch.randn(
        x_shape.shape,
        generator=generator,
        device=accelerator.device,
        dtype=torch.bfloat16,
    )
    target_tokens = x.shape[1]
    if drop_background_tokens:
        target_indices, ref_indices = foreground_token_indices(
            batch,
            target_tokens,
            ref_token_lengths,
            accelerator.device,
        )
    else:
        # Explicit inference switch: retain complete target/reference token
        # sequences. Training token dropping remains unchanged.
        batch_size = x.shape[0]
        target_indices = torch.arange(
            target_tokens, device=accelerator.device
        )[None].expand(batch_size, -1)
        ref_indices = [
            torch.arange(token_count, device=accelerator.device)[None].expand(
                batch_size, -1
            )
            for token_count in ref_token_lengths
        ]
    target_ids = gather_sequence(x_ids, target_indices)
    ref_tokens, ref_ids = prune_reference_tokens(
        ref_tokens,
        ref_ids,
        ref_token_lengths,
        ref_indices,
    )

    # FLUX.2 Klein Base is undistilled and uses text CFG. Keep both branches
    # conditioned on the same selected image tokens; only the text differs.
    empty_ctx = text_encoder([""] * x.shape[0]).to(torch.bfloat16)
    empty_ctx, empty_ctx_ids = batched_prc_txt(empty_ctx)
    cfg_ctx = torch.cat([empty_ctx, ctx], dim=0)
    cfg_ctx_ids = torch.cat([empty_ctx_ids, ctx_ids], dim=0)
    # Preserve the original TexVerse inference schedule exactly. At R=1024,
    # height=2R and width=3R produce image_seq_len=384 (not the 24,576 FLUX.2
    # atlas-token count). Only the denoiser implementation has changed.
    height = resolution * 2
    width = resolution * 3
    schedule_seq_len = (width // 8) * (height // 8) // (16 * 16)
    timesteps = uno_schedule(num_steps, schedule_seq_len, shift=True)

    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full(
            (x.shape[0],), t_curr, dtype=torch.bfloat16, device=accelerator.device
        )
        # Match original denoise_test: only target foreground tokens and
        # foreground reference tokens enter every Transformer step.
        x_foreground = gather_sequence(x, target_indices)
        x_in, x_in_ids = x_foreground, target_ids
        if ref_tokens is not None:
            x_in = torch.cat([x_in, ref_tokens.to(torch.bfloat16)], dim=1)
            x_in_ids = torch.cat([x_in_ids, ref_ids], dim=1)
        cfg_x_in = torch.cat([x_in, x_in], dim=0)
        cfg_x_in_ids = torch.cat([x_in_ids, x_in_ids], dim=0)
        cfg_t_vec = torch.cat([t_vec, t_vec], dim=0)
        pred_cfg = model(
            x=cfg_x_in,
            x_ids=cfg_x_in_ids,
            timesteps=cfg_t_vec,
            ctx=cfg_ctx,
            ctx_ids=cfg_ctx_ids,
            guidance=None,
        )[:, : x_foreground.shape[1]]
        pred_uncond, pred_cond = pred_cfg.chunk(2)
        pred = pred_uncond + guidance * (pred_cond - pred_uncond)
        # PEFT/Accelerate may return the CFG result in fp32 even though the
        # sampler state is bf16. torch.scatter requires self/src dtypes to be
        # identical, so explicitly return the Euler update to the state dtype.
        updated_foreground = (
            x_foreground + (t_prev - t_curr) * pred
        ).to(dtype=x.dtype)
        x = x.scatter(
            1,
            target_indices[..., None].expand_as(updated_foreground),
            updated_foreground,
        )

    # Always reconstruct unused target-background positions from the same base
    # latent as training-era UNO inference. `remove_background` only controls
    # the final alpha-mask whitening below; it no longer changes model inputs.
    if background is not None:
        if background.shape[0] == 1 and x.shape[0] != 1:
            background = background.expand(x.shape[0], -1, -1, -1)
        bg_tokens, _ = batched_prc_img(background)
        if bg_tokens.shape[1] != target_tokens:
            raise ValueError(
                f"Background has {bg_tokens.shape[1]} tokens, expected {target_tokens}"
            )
        # Match original denoise_test: scatter generated foreground back into
        # the complete black-background latent before VAE decoding.
        generated_foreground = gather_sequence(x, target_indices)
        x = bg_tokens.to(x.dtype).scatter(
            1,
            target_indices[..., None].expand_as(generated_foreground),
            generated_foreground,
        )

    x_grid = rearrange(
        x.float(),
        "b (h w) c -> b c h w",
        h=latent_hw[0],
        w=latent_hw[1],
    )
    # Preserve the original TexVerse inference path: decode the six views one
    # tile at a time, then reconstruct the 2x3 canvas.  Besides matching the
    # old decoder semantics, this avoids a much larger full-canvas AE decode.
    decoded_rows = []
    for latent_row in torch.chunk(x_grid, 2, dim=-2):
        decoded_tiles = [ae.decode(tile) for tile in torch.chunk(latent_row, 3, dim=-1)]
        decoded_rows.append(torch.cat(decoded_tiles, dim=-1))
    decoded = torch.cat(decoded_rows, dim=-2).float().clamp(-1, 1)
    image = rearrange(decoded[0], "c h w -> h w c")
    image_np = (127.5 * (image + 1.0)).cpu().numpy().astype(np.uint8)

    if remove_background:
        full_mask = batch["masks_full"][0][0].cpu().numpy() == 0
        if full_mask.shape != image_np.shape[:2]:
            mask_image = Image.fromarray((~full_mask).astype(np.uint8) * 255)
            mask_image = mask_image.resize(
                (image_np.shape[1], image_np.shape[0]), Image.Resampling.NEAREST
            )
            full_mask = np.asarray(mask_image) == 0
        image_np[np.repeat(full_mask[..., None], 3, axis=2)] = 255
    return Image.fromarray(image_np)


def _lora_targets(dit, double_indices, single_indices) -> list[str]:
    double_indices = (
        list(range(len(dit.double_blocks))) if double_indices is None else double_indices
    )
    single_indices = (
        list(range(len(dit.single_blocks))) if single_indices is None else single_indices
    )
    targets = []
    for index in double_indices:
        if not 0 <= index < len(dit.double_blocks):
            raise ValueError(f"double block index {index} is invalid for FLUX.2")
        prefix = f"double_blocks.{index}"
        targets.extend(
            [
                f"{prefix}.img_attn.qkv",
                f"{prefix}.img_attn.proj",
                f"{prefix}.txt_attn.qkv",
                f"{prefix}.txt_attn.proj",
            ]
        )
    for index in single_indices:
        if not 0 <= index < len(dit.single_blocks):
            raise ValueError(f"single block index {index} is invalid for FLUX.2")
        prefix = f"single_blocks.{index}"
        targets.extend([f"{prefix}.linear1", f"{prefix}.linear2"])
    return targets


def apply_lora(dit, args):
    dit.requires_grad_(False)
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=_lora_targets(
            dit, args.double_blocks_indices, args.single_blocks_indices
        ),
        lora_dropout=0.0,
        bias="none",
    )
    dit = get_peft_model(dit, config)
    dit.print_trainable_parameters()
    return dit


def save_lora(dit, accelerator: Accelerator, save_path: str):
    os.makedirs(save_path, exist_ok=True)
    unwrapped = accelerator.unwrap_model(dit)
    state = {
        key: value.detach().to(torch.float32).cpu().contiguous()
        for key, value in unwrapped.state_dict().items()
        if "lora_" in key
    }
    save_file(state, os.path.join(save_path, "dit_lora.safetensors"))


def snapshot_lora_base_weights(model):
    """Keep exact CPU copies of weights that PEFT merge/unmerge mutates."""
    snapshots = []
    for module_name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "get_base_layer"):
            continue
        weight = module.get_base_layer().weight
        snapshots.append(
            (module_name, weight, weight.detach().to(device="cpu", copy=True))
        )
    return snapshots


@torch.no_grad()
def restore_lora_base_weights(snapshots):
    for _module_name, weight, original in snapshots:
        weight.copy_(original)


def resume_from_checkpoint(
    resume_from_checkpoint: str | None | Literal["latest"],
    project_dir: str,
    accelerator: Accelerator,
    dit,
    dit_ema_dict: dict | None = None,
):
    del project_dir
    global start_step
    if resume_from_checkpoint is None:
        return dit, dit_ema_dict, 0

    path = resume_from_checkpoint
    name = os.path.basename(path)
    accelerator.print(f"Resuming from checkpoint {path}")
    state = load_file(os.path.join(path, "dit_lora.safetensors"))
    accelerator.unwrap_model(dit).load_state_dict(state, strict=False)
    if dit_ema_dict is not None:
        ema_path = os.path.join(path, "dit_lora_ema.safetensors")
        if os.path.exists(ema_path):
            dit_ema_dict = load_file(ema_path)
    # A released LoRA folder (e.g. checkpoints/flux1/lora) has no step suffix.
    _m = re.fullmatch(r"checkpoint-(\d+)", name)
    global_step = int(_m.group(1)) if _m else 0
    start_step = global_step
    return dit, dit_ema_dict, global_step


@dataclasses.dataclass
class TrainArgs:
    ## accelerator -- unchanged
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    gradient_accumulation_steps: int = 1
    seed: int = 42
    wandb_project_name: str | None = None
    wandb_run_name: str | None = None

    ## model -- FLUX.2 model identity replaces FLUX.1; tuning parameters unchanged
    model_name: str = "flux.2-klein-base-4b"
    lora_rank: int = 64
    double_blocks_indices: list[int] | None = dataclasses.field(default=None)
    single_blocks_indices: list[int] | None = dataclasses.field(default=None)
    pe: Literal["d", "h", "w", "o"] = "d"
    gradient_checkpoint: bool = True
    # The project-local SLA backward kernel is fixed for irregular sequence
    # lengths; retain topk=0.20 acceleration as the 2048 default.
    sparse_attention: bool = True
    sparse_attention_topk: float = 0.20
    sparse_attention_blkq: int = 128
    sparse_attention_blkk: int = 64
    ema: bool = False
    ema_interval: int = 1
    ema_decay: float = 0.99

    ## optimizer -- unchanged
    learning_rate: float = 1e-4
    adam_betas: list[float] = dataclasses.field(default_factory=lambda: [0.9, 0.999])
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    ## lr scheduler -- unchanged
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 1000000

    ## dataloader -- unchanged
    train_data_json: str = "data/train.json"
    batch_size: int = 1
    text_dropout: float = 0.1
    resolution: int = 2048
    resolution_ref: int | None = None
    eval_data_json: str = "data/demo.json"
    eval_batch_size: int = 1
    num_workers: int = 24
    bucket_metadata_json: str | None = None
    inference_num_steps: int = 25
    # Independent inference-only switch. Training always drops background tokens.
    inference_drop_background_tokens: bool = True
    # Independent output-only switch: force pixels outside the alpha mask white.
    inference_remove_background: bool = False

    # Fine-tuned native FLUX.2 decoder used for 2048 validation inference.
    decoder_ckpt: str = "checkpoints/flux2/decoder.pt"

    ## misc -- unchanged
    resume_from_checkpoint: str | None | Literal["latest"] = None
    save_only_steps: int = 1000
    checkpointing_steps: int = 4000
    project_dir: str | None = None

    def __post_init__(self):
        if self.project_dir is None:
            model_tag = self.model_name.replace(".", "_")
            self.project_dir = (
                "outputs/"
                f"full_mr_{model_tag}_res={self.resolution}_lr={self.learning_rate}"
            )


def main(args: TrainArgs):
    accelerator = Accelerator(
        project_dir=args.project_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
    )
    set_seed(args.seed, device_specific=True)
    accelerator.init_trackers(project_name="")
    weight_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "no": torch.float32,
    }.get(accelerator.mixed_precision, torch.float32)

    logging.basicConfig(
        format=f"[RANK {accelerator.process_index}] %(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        force=True,
    )
    logger.info(accelerator.state)
    logger.info("FLUX.2 training script launched", main_process_only=False)

    dit, ae, text_encoder = get_models(args.model_name, accelerator.device)
    sparse_config = configure_sparse_attention(
        enabled=args.sparse_attention,
        head_dim=dit.hidden_size // dit.num_heads,
        topk=args.sparse_attention_topk,
        blkq=args.sparse_attention_blkq,
        blkk=args.sparse_attention_blkk,
    )
    logger.info("DiT attention backend: %s", sparse_config)
    load_decoder_checkpoint(ae, args.decoder_ckpt)
    ae.requires_grad_(False).eval()
    text_encoder.requires_grad_(False).eval()

    dit = apply_lora(dit, args)
    dit.base_model.model.gradient_checkpointing = args.gradient_checkpoint
    dit.to(accelerator.device)
    dit.train()

    dit_ema_dict = (
        {
            key: deepcopy(value).requires_grad_(False)
            for key, value in dit.named_parameters()
            if value.requires_grad
        }
        if args.ema
        else None
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in dit.parameters() if parameter.requires_grad],
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

    dit, dit_ema_dict, global_step = resume_from_checkpoint(
        args.resume_from_checkpoint,
        project_dir=args.project_dir,
        accelerator=accelerator,
        dit=dit,
        dit_ema_dict=dit_ema_dict,
    )

    dataset = FluxPairedDatasetMR(
        json_file=args.train_data_json,
        resolution=args.resolution,
        resolution_ref=args.resolution_ref,
        bucket_metadata_json=args.bucket_metadata_json,
    )
    batch_sampler = LengthAwareBatchSampler(
        dataset,
        batch_size=args.batch_size,
        drop_last=False,
        seed=args.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    eval_dataset = FluxPairedDatasetMR(
        json_file=args.eval_data_json,
        resolution=args.resolution,
        resolution_ref=args.resolution_ref,
        bucket_metadata_json=args.bucket_metadata_json,
    )
    eval_batch_sampler = LengthAwareBatchSampler(
        eval_dataset,
        batch_size=args.eval_batch_size,
        drop_last=False,
        seed=args.seed,
    )
    logger.info(
        "TexVerse v2 train samples=%d (skipped outside v2=%d)",
        len(dataset.data_dicts),
        dataset.skipped_not_in_v2,
    )
    logger.info(
        "MR foreground bucket ordering enabled: metadata=%s, entries=%d, "
        "missing_objects=%d, range=[%.4f, %.4f]",
        args.bucket_metadata_json,
        len(dataset.lengths),
        dataset.missing_bucket_metadata,
        min(dataset.lengths),
        max(dataset.lengths),
    )
    logger.info(
        "TexVerse v2 eval samples=%d (skipped outside v2=%d)",
        len(eval_dataset.data_dicts),
        eval_dataset.skipped_not_in_v2,
    )
    logger.info(
        "MR eval bucket ordering enabled: entries=%d, missing_objects=%d, "
        "range=[%.4f, %.4f]",
        len(eval_dataset.lengths),
        eval_dataset.missing_bucket_metadata,
        min(eval_dataset.lengths),
        max(eval_dataset.lengths),
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_sampler=eval_batch_sampler,
        collate_fn=eval_dataset.collate_fn,
    )

    dit, optimizer, lr_scheduler, dataloader = accelerator.prepare(
        dit, optimizer, lr_scheduler, dataloader
    )
    eval_dataloader = accelerator.prepare_data_loader(eval_dataloader)

    # Opt-in diagnostics for controlled sparse-vs-dense NaN experiments.
    diagnose_nonfinite = os.environ.get("DIAGNOSE_NONFINITE", "0") == "1"
    diagnostic_disable_checkpoints = (
        os.environ.get("DIAGNOSTIC_DISABLE_CHECKPOINTS", "0") == "1"
    )
    diagnostic_grad_norm_limit = float(
        os.environ.get("DIAGNOSTIC_GRAD_NORM_LIMIT", "inf")
    )

    def every_rank_ok(local_ok: bool) -> bool:
        flag = torch.tensor(
            int(local_ok), device=accelerator.device, dtype=torch.int32
        )
        return int(accelerator.reduce(flag, reduction="sum").item()) == accelerator.num_processes

    total_batch_size = (
        args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )
    logger.info("***** Running FLUX.2 training *****")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Total validation prompts = {len(eval_dataloader)}")
    logger.info(
        "  Inference alpha-background whitening = %s",
        "ON" if args.inference_remove_background else "OFF",
    )
    logger.info(
        "  Inference background-token dropping = %s",
        "ON" if args.inference_drop_background_tokens else "OFF",
    )
    inference_schedule_seq_len = (
        (args.resolution * 3 // 8)
        * (args.resolution * 2 // 8)
        // (16 * 16)
    )
    logger.info(
        "  Inference sampling = %d steps, original UNO schedule seq_len = %d",
        args.inference_num_steps,
        inference_schedule_seq_len,
    )

    # Exact original UNO training-time distribution: build a 999-step shifted
    # schedule using one view's packed latent length, then sample one of its
    # 1000 entries uniformly for every training example.
    training_schedule_seq_len = (args.resolution // 8) ** 2 // 4
    training_timesteps = torch.tensor(
        uno_schedule(999, training_schedule_seq_len, shift=True),
        device=accelerator.device,
    )
    logger.info(
        "  Training timestep schedule = original UNO shifted-999, seq_len = %d",
        training_schedule_seq_len,
    )

    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=start_step,
        total=args.max_train_steps,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    train_loss = 0.0

    bg_pil = Image.new("RGB", (args.resolution, args.resolution), color=(0, 0, 0))
    bg_tensor = dataset.transform(bg_pil).unsqueeze(0).to(accelerator.device)
    with torch.no_grad():
        bg_latent = ae.encode(bg_tensor.to(torch.float32)).repeat(1, 1, 2, 3)

    while global_step < args.max_train_steps:
        for batch in dataloader:
            (
                x_1,
                x_ids,
                ref_tokens,
                ref_ids,
                ref_token_lengths,
                ctx,
                ctx_ids,
                latent_hw,
            ) = encode_conditions(batch, ae, text_encoder, accelerator.device)
            target_indices, ref_indices = foreground_token_indices(
                batch,
                x_1.shape[1],
                ref_token_lengths,
                accelerator.device,
            )
            target_ids = gather_sequence(x_ids, target_indices)
            ref_tokens, ref_ids = prune_reference_tokens(
                ref_tokens,
                ref_ids,
                ref_token_lengths,
                ref_indices,
            )
            if global_step == start_step:
                full_token_count = x_1.shape[1] + sum(ref_token_lengths)
                kept_token_count = target_indices.shape[1] + sum(
                    indices.shape[1] for indices in ref_indices
                )
                logger.info(
                    "Background token dropping active: target=%d/%d, refs=%s, total=%d/%d (%.1f%% kept)",
                    target_indices.shape[1],
                    x_1.shape[1],
                    [
                        f"{indices.shape[1]}/{token_count}"
                        for indices, token_count in zip(
                            ref_indices, ref_token_lengths, strict=True
                        )
                    ],
                    kept_token_count,
                    full_token_count,
                    100.0 * kept_token_count / full_token_count,
                )

            batch_size = x_1.shape[0]
            timestep_indices = torch.randint(
                0,
                training_timesteps.shape[0],
                (batch_size,),
                device=accelerator.device,
            )
            t = training_timesteps[timestep_indices]
            x_0 = torch.randn_like(x_1)
            x_t = (1 - t[:, None, None]) * x_1 + t[:, None, None] * x_0

            # Original UNO behavior: target and every reference are pruned to
            # foreground before entering any Transformer block.
            x_1_foreground = gather_sequence(x_1, target_indices)
            x_0_foreground = gather_sequence(x_0, target_indices)
            x_in = gather_sequence(x_t, target_indices).to(weight_dtype)
            x_in_ids = target_ids
            if ref_tokens is not None:
                x_in = torch.cat([x_in, ref_tokens.to(weight_dtype)], dim=1)
                x_in_ids = torch.cat([x_in_ids, ref_ids], dim=1)

            grad_log = {}
            with accelerator.accumulate(dit):
                pred_full = dit(
                    x=x_in,
                    x_ids=x_in_ids,
                    timesteps=t.to(weight_dtype),
                    ctx=ctx.to(weight_dtype),
                    ctx_ids=ctx_ids,
                    guidance=None,
                )
                pred = pred_full[:, : x_1_foreground.shape[1]]
                loss = F.mse_loss(
                    pred.float(),
                    (x_0_foreground - x_1_foreground).float(),
                    reduction="mean",
                )

                avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                grad_dict = {}
                hooks = []
                for name, parameter in accelerator.unwrap_model(dit).named_parameters():
                    if parameter.requires_grad:
                        def make_hook(parameter_name):
                            def hook(gradient):
                                if gradient is not None:
                                    grad_dict[parameter_name] = (
                                        grad_dict.get(parameter_name, 0.0)
                                        + gradient.norm().item() ** 2
                                    )
                            return hook

                        hooks.append(parameter.register_hook(make_hook(name)))

                accelerator.backward(loss)
                for hook in hooks:
                    hook.remove()

                if diagnose_nonfinite and math.isfinite(diagnostic_grad_norm_limit):
                    gradient_norm_sq = 0.0
                    largest_gradient_name = None
                    largest_gradient_norm = -1.0
                    largest_gradient_max_abs = -1.0
                    for name, parameter in accelerator.unwrap_model(dit).named_parameters():
                        if parameter.grad is None:
                            continue
                        gradient_fp64 = parameter.grad.detach().to(torch.float64)
                        if not torch.isfinite(gradient_fp64).all():
                            parameter_norm = float("inf")
                            parameter_max_abs = float("inf")
                        else:
                            parameter_norm = gradient_fp64.norm().item()
                            parameter_max_abs = gradient_fp64.abs().max().item()
                        gradient_norm_sq += parameter_norm**2
                        if parameter_norm > largest_gradient_norm:
                            largest_gradient_name = name
                            largest_gradient_norm = parameter_norm
                            largest_gradient_max_abs = parameter_max_abs
                    accurate_gradient_norm = gradient_norm_sq**0.5
                    local_norm_ok = accurate_gradient_norm <= diagnostic_grad_norm_limit
                    if not every_rank_ok(local_norm_ok):
                        logger.error(
                            "DIAGNOSTIC MR gradient norm limit exceeded: next_step=%d "
                            "rank=%d ids=%s t=%s target_tokens=%d total_tokens=%d "
                            "norm_fp64=%.9e limit=%.9e largest_parameter=%s "
                            "largest_parameter_norm=%.9e largest_parameter_max_abs=%.9e",
                            global_step + 1,
                            accelerator.process_index,
                            batch.get("id"),
                            t.detach().float().cpu().tolist(),
                            x_1_foreground.shape[1],
                            x_in.shape[1],
                            accurate_gradient_norm,
                            diagnostic_grad_norm_limit,
                            largest_gradient_name,
                            largest_gradient_norm,
                            largest_gradient_max_abs,
                            main_process_only=False,
                        )
                        raise FloatingPointError(
                            f"MR gradient norm limit exceeded before step {global_step + 1}"
                        )

                if accelerator.sync_gradients:
                    block_sq = {}
                    for name, squared_norm in grad_dict.items():
                        if "double_blocks." in name:
                            index = name.split("double_blocks.")[1].split(".")[0]
                            key = f"grad_norm/double_blocks/{index}"
                        elif "single_blocks." in name:
                            index = name.split("single_blocks.")[1].split(".")[0]
                            key = f"grad_norm/single_blocks/{index}"
                        else:
                            key = f"grad_norm/other/{name}"
                        block_sq[key] = block_sq.get(key, 0.0) + squared_norm
                    grad_log.update({key: value**0.5 for key, value in block_sq.items()})
                    pre_clip = sum(grad_dict.values()) ** 0.5
                    clipped_grad_norm = accelerator.clip_grad_norm_(
                        dit.parameters(), args.max_grad_norm
                    )
                    local_clip_ok = bool(
                        torch.isfinite(torch.as_tensor(clipped_grad_norm)).all().item()
                    )
                    if not every_rank_ok(local_clip_ok):
                        logger.error(
                            "Nonfinite MR total gradient norm before optimizer: "
                            "next_step=%d rank=%d ids=%s norm=%s",
                            global_step + 1,
                            accelerator.process_index,
                            batch.get("id"),
                            clipped_grad_norm,
                            main_process_only=False,
                        )
                        optimizer.zero_grad()
                        raise FloatingPointError(
                            f"Nonfinite MR total gradient norm before step {global_step + 1}"
                        )
                    grad_log["grad_norm/total_pre_clip"] = pre_clip
                    grad_log["grad_norm/total_post_clip"] = min(
                        pre_clip, args.max_grad_norm
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, **grad_log}, step=global_step)
                train_loss = 0.0

            if (
                accelerator.sync_gradients
                and dit_ema_dict is not None
                and global_step % args.ema_interval == 0
            ):
                source = dict(accelerator.unwrap_model(dit).named_parameters())
                for name, target in dit_ema_dict.items():
                    target.data.lerp_(source[name].data.to(target), 1 - args.ema_decay)

            should_checkpoint = accelerator.sync_gradients and (
                global_step % args.checkpointing_steps == 0
                or global_step == 100 + start_step
            )
            should_save_only = (
                accelerator.sync_gradients
                and global_step % args.save_only_steps == 0
                and not should_checkpoint
            )
            if diagnostic_disable_checkpoints:
                should_checkpoint = False
                should_save_only = False
            if should_save_only:
                save_path = os.path.join(args.project_dir, f"checkpoint-{global_step}")
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_lora(dit, accelerator, save_path)
                    if dit_ema_dict is not None:
                        save_file(
                            {
                                key: value.detach().float().cpu().contiguous()
                                for key, value in dit_ema_dict.items()
                            },
                            os.path.join(save_path, "dit_lora_ema.safetensors"),
                        )
                    logger.info(f"Saved checkpoint-only state to {save_path}")
                accelerator.wait_for_everyone()
            if should_checkpoint:
                save_path = os.path.join(args.project_dir, f"checkpoint-{global_step}")
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_lora(dit, accelerator, save_path)
                    if dit_ema_dict is not None:
                        save_file(
                            {
                                key: value.detach().float().cpu().contiguous()
                                for key, value in dit_ema_dict.items()
                            },
                            os.path.join(save_path, "dit_lora_ema.safetensors"),
                        )
                    logger.info(f"Saved state to {save_path}")

                # All ranks write their validation images into this directory.
                # Wait until the main rank has created it and finished the LoRA save.
                accelerator.wait_for_everyone()
                dit.eval()
                torch.set_grad_enabled(False)
                inference_model = accelerator.unwrap_model(dit)
                # Merge the adapter for validation so PEFT does not materialize
                # a full sequence-sized LoRA branch at every 2048 linear layer.
                base_weight_snapshots = snapshot_lora_base_weights(inference_model)
                snapshot_gib = sum(
                    original.numel() * original.element_size()
                    for _name, _weight, original in base_weight_snapshots
                ) / 2**30
                logger.info(
                    "Backed up %.2f GiB of MR LoRA base weights on CPU for exact post-validation restore",
                    snapshot_gib,
                )
                inference_model.merge_adapter(safe_merge=True)
                torch.cuda.empty_cache()
                logger.info("Merged MR LoRA adapter for memory-efficient validation")
                # Accelerator.unwrap_model keeps its mixed-precision forward
                # wrapper by default. That wrapper autocasts the entire DiT
                # forward and converts its output to fp32. At 25-step CFG
                # sampling this is not numerically equivalent to the native
                # bf16 FLUX.2 forward: it reproducibly blurs atlas column 3.
                # Temporarily call the forward saved by Accelerator before the
                # wrapper was installed, then restore the training wrapper even
                # if validation raises.
                wrapped_validation_forward = inference_model.forward
                original_validation_forward = getattr(
                    inference_model, "_original_forward", None
                )
                if original_validation_forward is not None:
                    inference_model.forward = original_validation_forward
                    logger.info(
                        "Bypassing Accelerator mixed-precision forward wrapper "
                        "for numerically consistent MR validation"
                    )
                try:
                    rank = accelerator.process_index
                    local_len = len(eval_dataloader)
                    for local_i, eval_batch in enumerate(eval_dataloader):
                        image_dir = eval_batch["image_dir"][0]
                        render_path = eval_batch["render_path"][0]
                        sample_id = eval_batch["id"][0]
                        global_i = rank * local_len + local_i
                        output_path = os.path.join(
                            save_path, f"eval_{sample_id}_{global_i}.webp"
                        )
                        if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                            try:
                                with Image.open(output_path) as existing_image:
                                    existing_image.verify()
                            except (OSError, SyntaxError):
                                pass
                            else:
                                continue

                        result = inference(
                            eval_batch,
                            inference_model,
                            text_encoder,
                            ae,
                            accelerator,
                            seed=0,
                            resolution=args.resolution,
                            background=bg_latent,
                            remove_background=args.inference_remove_background,
                            drop_background_tokens=args.inference_drop_background_tokens,
                            num_steps=args.inference_num_steps,
                        )
                        composite = make_eval_composite(
                            image_dir, result, render_path, args.resolution
                        )
                        composite.save(
                            output_path,
                            format="WEBP",
                            lossless=True,
                        )
                finally:
                    if original_validation_forward is not None:
                        inference_model.forward = wrapped_validation_forward

                inference_model.unmerge_adapter()
                restore_lora_base_weights(base_weight_snapshots)
                del base_weight_snapshots
                torch.cuda.empty_cache()
                gc.collect()
                torch.set_grad_enabled(True)
                dit.train()
                accelerator.wait_for_everyone()

            progress_bar.set_postfix(
                step_loss=loss.detach().item(), lr=lr_scheduler.get_last_lr()[0]
            )
            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = transformers.HfArgumentParser([TrainArgs])
    args_tuple = parser.parse_args_into_dataclasses(args_file_flag="--config")
    main(*args_tuple)
