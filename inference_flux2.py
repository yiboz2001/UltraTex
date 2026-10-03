"""Distributed batch inference for FLUX.2 TexVerse 2048 albedo LoRA.

This is the FLUX.2 counterpart of inference_flux1.py.  The
dataset semantics and 7R x 4R comparison canvas are retained, while model
loading and sampling use the current FLUX.2 trainer.  The DiT is intentionally
never passed through Accelerator.prepare(): diagnostics proved that its mixed-
precision forward wrapper can corrupt the third atlas column over 25 CFG steps.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from datetime import timedelta
from pathlib import Path
from typing import Literal

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("TORCH_DISTRIBUTED_TIMEOUT", "180000")

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, set_seed
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ultratex.backbones.flux2.model import configure_sparse_attention
from ultratex.data.dataset import FluxPairedDatasetV2
from ultratex.data.dataset_ai import FluxPairedDatasetAI
from ultratex.data.dataset_mr import FluxPairedDatasetMR


logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
USER_ROOT = PROJECT_ROOT.parent


def load_and_concat(image_dir: str, subdir: str, resolution: int) -> Image.Image:
    paths = [os.path.join(image_dir, subdir, f"{index:03d}.webp") for index in range(6)]
    images = []
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing file: {path}")
        image = Image.open(path).convert("RGBA").resize(
            (resolution, resolution), Image.Resampling.LANCZOS
        )
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.getchannel("A"))
        images.append(np.asarray(background, dtype=np.uint8))
    row1 = np.concatenate([images[0], images[1], images[3]], axis=1)
    row2 = np.concatenate([images[2], images[4], images[5]], axis=1)
    return Image.fromarray(np.concatenate([row1, row2], axis=0))


def make_eval_composite(
    image_dir: str,
    result_img: Image.Image,
    render_path: str,
    resolution: int,
    bump_subdir: str,
    target_subdir: str,
) -> Image.Image:
    """Preserve the original 7R x 4R inference comparison layout."""
    position = load_and_concat(image_dir, "position", resolution)
    bump = load_and_concat(image_dir, bump_subdir, resolution)
    target = load_and_concat(image_dir, target_subdir, resolution)
    render_rgba = Image.open(render_path).convert("RGBA").resize(
        (resolution, resolution), Image.Resampling.LANCZOS
    )
    render = Image.new("RGB", render_rgba.size, (255, 255, 255))
    render.paste(render_rgba, mask=render_rgba.getchannel("A"))

    cell_w, cell_h = render.size
    canvas = Image.new("RGB", (cell_w * 7, cell_h * 4), (255, 255, 255))
    canvas.paste(render, (0, 2 * cell_h))
    canvas.paste(position, (cell_w, 0))
    canvas.paste(bump, (cell_w, 2 * cell_h))
    canvas.paste(target, (cell_w * 4, 0))
    canvas.paste(result_img.resize(position.size), (cell_w * 4, 2 * cell_h))
    return canvas


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        samples_per_object = getattr(self.dataset, "samples_per_object", 1)
        expected = self.dataset.data_dicts[index // samples_per_object]
        expected_id = expected.get("id") or expected.get("id_only")
        if expected_id is not None and item.get("id") != expected_id:
            raise RuntimeError(
                f"Dataset fallback changed index {index} from {expected_id!r} "
                f"to {item.get('id')!r}; fix the missing/corrupt source data"
            )
        return index, item

    def collate_fn(self, samples):
        indices, items = zip(*samples, strict=True)
        batch = self.dataset.collate_fn(list(items))
        batch["dataset_index"] = list(indices)
        return batch


@dataclasses.dataclass
class InferenceArgs:
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    seed: int = 42
    model_name: str = "flux.2-klein-base-9b"
    lora_rank: int = 64
    double_blocks_indices: list[int] | None = dataclasses.field(default=None)
    single_blocks_indices: list[int] | None = dataclasses.field(default=None)

    resolution: int = 2048
    resolution_ref: int | None = None
    eval_data_json: str = "data/eval.json"
    task: Literal["albedo", "mr"] = "albedo"
    dataset_type: Literal["standard", "ai"] = "standard"
    eval_batch_size: int = 1
    num_workers: int = 4
    max_samples: int | None = None

    inference_num_steps: int = 25
    guidance: float = 4.0
    inference_drop_background_tokens: bool = True
    inference_remove_background: bool = True
    sparse_attention: bool = True
    sparse_attention_topk: float = 0.20
    sparse_attention_blkq: int = 128
    sparse_attention_blkk: int = 64

    decoder_ckpt: str = "checkpoints/flux2/decoder.pt"
    resume_from_checkpoint: str | None = "checkpoints/flux2/lora"
    project_dir: str = "outputs/inference_flux2"
    output_format: Literal["png", "webp"] = "png"
    save_composite: bool = True
    save_raw_result: bool = True
    overwrite: bool = False
    dry_run: bool = False
    distributed_timeout_seconds: int = 7200

    def __post_init__(self):
        if self.eval_batch_size != 1:
            raise ValueError("eval_batch_size must be 1 to preserve variable foreground pruning")
        if self.resolution_ref is None:
            self.resolution_ref = self.resolution
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive")
        if self.distributed_timeout_seconds <= 0:
            raise ValueError("distributed_timeout_seconds must be positive")


def checkpoint_file_and_step(path: str | None) -> tuple[Path, int]:
    if not path:
        raise ValueError("--resume_from_checkpoint is required")
    checkpoint = Path(path).expanduser().resolve()
    checkpoint_file = checkpoint if checkpoint.is_file() else checkpoint / "dit_lora.safetensors"
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"LoRA checkpoint not found: {checkpoint_file}")
    match = re.search(r"checkpoint-(\d+)", str(checkpoint_file))
    return checkpoint_file, int(match.group(1)) if match else 0


def get_trainer(task: str):
    if task == "albedo":
        import train_flux2

        return train_flux2
    import train_flux2_mr

    return train_flux2_mr


def build_dataset(args: InferenceArgs):
    if args.dataset_type == "ai":
        return FluxPairedDatasetAI(
            json_file=args.eval_data_json,
            resolution=args.resolution,
            resolution_ref=args.resolution_ref,
            target_subdir="albedo" if args.task == "albedo" else "roughness_metallic",
        )
    dataset_class = FluxPairedDatasetV2 if args.task == "albedo" else FluxPairedDatasetMR
    return dataset_class(
        json_file=args.eval_data_json,
        resolution=args.resolution,
        resolution_ref=args.resolution_ref,
    )


def is_valid_image(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, SyntaxError):
        return False
    return True


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def finish_distributed(accelerator: Accelerator) -> None:
    """Synchronize workers and explicitly release the NCCL process group."""
    accelerator.wait_for_everyone()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def load_inference_models(args: InferenceArgs, accelerator: Accelerator, checkpoint_file: Path, trainer):
    model, ae, text_encoder = trainer.get_models(args.model_name, accelerator.device)
    sparse_config = configure_sparse_attention(
        enabled=args.sparse_attention,
        head_dim=model.hidden_size // model.num_heads,
        topk=args.sparse_attention_topk,
        blkq=args.sparse_attention_blkq,
        blkk=args.sparse_attention_blkk,
    )
    logger.info("DiT attention backend: %s", sparse_config)
    trainer.load_decoder_checkpoint(ae, args.decoder_ckpt)
    ae.requires_grad_(False).eval()
    text_encoder.requires_grad_(False).eval()

    model = trainer.apply_lora(model, args)
    state = load_file(str(checkpoint_file), device="cpu")
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing_lora = [key for key in incompatible.missing_keys if "lora_" in key]
    if unexpected or missing_lora:
        raise RuntimeError(
            f"LoRA load mismatch: unexpected={unexpected[:10]}, missing_lora={missing_lora[:10]}"
        )
    if not all(torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError(f"Checkpoint contains non-finite LoRA tensors: {checkpoint_file}")
    del state
    model.to(accelerator.device)
    model.merge_adapter(safe_merge=True)
    model.requires_grad_(False).eval()
    # Deliberately do not call accelerator.prepare(model). The isolated
    # third-column diagnosis proved the wrapped forward is numerically unsafe.
    if hasattr(model, "_original_forward"):
        raise RuntimeError("Inference model unexpectedly contains an Accelerator forward wrapper")
    return model, ae, text_encoder


def main(args: InferenceArgs) -> None:
    process_group_kwargs = InitProcessGroupKwargs(
        timeout=timedelta(seconds=args.distributed_timeout_seconds)
    )
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[process_group_kwargs],
    )
    set_seed(args.seed, device_specific=True)
    logging.basicConfig(
        format=f"[RANK {accelerator.process_index}] %(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        force=True,
    )

    checkpoint_file, checkpoint_step = checkpoint_file_and_step(args.resume_from_checkpoint)
    for required in (Path(args.eval_data_json), Path(args.decoder_ckpt)):
        if not required.exists():
            raise FileNotFoundError(required)

    trainer = get_trainer(args.task)
    base_dataset = build_dataset(args)
    total = len(base_dataset)
    limit = min(total, args.max_samples) if args.max_samples is not None else total
    all_indices = list(range(limit))
    rank_indices = all_indices[accelerator.process_index :: accelerator.num_processes]
    indexed_dataset = IndexedDataset(base_dataset)
    dataloader = DataLoader(
        indexed_dataset,
        batch_size=1,
        sampler=rank_indices,
        collate_fn=indexed_dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    output_dir = Path(args.project_dir) / f"checkpoint-{checkpoint_step}"
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "dataset_type": args.dataset_type,
            "task": args.task,
            "eval_data_json": str(Path(args.eval_data_json).resolve()),
            "dataset_items": total,
            "selected_items": limit,
            "world_size": accelerator.num_processes,
            "checkpoint": str(checkpoint_file),
            "resolution": args.resolution,
            "num_steps": args.inference_num_steps,
            "guidance": args.guidance,
            "drop_background_tokens": args.inference_drop_background_tokens,
            "remove_background": args.inference_remove_background,
            "native_unwrapped_forward": True,
            "output_format": args.output_format,
            "save_composite": args.save_composite,
            "save_raw_result": args.save_raw_result,
            "distributed_timeout_seconds": args.distributed_timeout_seconds,
        }
        (output_dir / "inference_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        )
    accelerator.wait_for_everyone()

    if args.dry_run:
        summary = {
            "rank": accelerator.process_index,
            "rank_items": len(rank_indices),
            "first_indices": rank_indices[:8],
            "output_dir": str(output_dir),
        }
        print(json.dumps(summary, ensure_ascii=False))
        finish_distributed(accelerator)
        return

    model, ae, text_encoder = load_inference_models(
        args, accelerator, checkpoint_file, trainer
    )
    black = Image.new("RGB", (args.resolution, args.resolution), (0, 0, 0))
    black_tensor = base_dataset.transform(black).unsqueeze(0).to(accelerator.device)
    with torch.no_grad():
        background = ae.encode(black_tensor.to(torch.float32)).repeat(1, 1, 2, 3)

    bump_subdir = "bump_normal_world"
    target_subdir = "albedo" if args.task == "albedo" else "roughness_metallic"
    extension = args.output_format
    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        dataset_index = int(batch["dataset_index"][0])
        sample_id = str(batch["id"][0])
        file_stem = f"eval_{safe_id(sample_id)}_{dataset_index}"
        composite_path = output_dir / f"{file_stem}.{extension}"
        raw_path = output_dir / f"{file_stem}_result.{extension}"
        requested_paths = []
        if args.save_composite:
            requested_paths.append(composite_path)
        if args.save_raw_result:
            requested_paths.append(raw_path)
        if requested_paths and not args.overwrite and all(is_valid_image(path) for path in requested_paths):
            continue

        result = trainer.inference(
            batch,
            model,
            text_encoder,
            ae,
            accelerator,
            seed=args.seed,
            resolution=args.resolution,
            background=background,
            guidance=args.guidance,
            remove_background=args.inference_remove_background,
            drop_background_tokens=args.inference_drop_background_tokens,
            num_steps=args.inference_num_steps,
        )
        save_kwargs = {"format": "WEBP", "lossless": True} if extension == "webp" else {"format": "PNG"}
        if args.save_raw_result:
            result.save(raw_path, **save_kwargs)
        if args.save_composite:
            composite = make_eval_composite(
                batch["image_dir"][0],
                result,
                batch["render_path"][0],
                args.resolution,
                bump_subdir,
                target_subdir,
            )
            composite.save(composite_path, **save_kwargs)

    finish_distributed(accelerator)
    if accelerator.is_main_process:
        logger.info("Batch inference complete: %s", output_dir)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser([InferenceArgs])
    (parsed_args,) = parser.parse_args_into_dataclasses(args_file_flag="--config")
    main(parsed_args)
