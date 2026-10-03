# Changes made when preparing this release

This repository is assembled from two internal codebases — one built on
FLUX.1-dev, one on FLUX.2-Klein. The model code, training loop, and the three
paper components (Background Token Dropping, Block-Sparse Attention,
Foreground-Aware VAE Decoding) are unchanged. The edits below are packaging
changes plus two functional fixes.

## Packaging

- Modules moved under a single `ultratex` package:
  `uno/flux/*` → `ultratex/backbones/flux1/*`,
  `flux2/*` → `ultratex/backbones/flux2/*`,
  `uno/dataset/*` → `ultratex/data/*`.
- Entry points renamed by backbone: `train_flux1.py`, `train_flux2.py`,
  `inference_flux1.py`, `inference_flux2.py`, `inference_flux1_ai.py`.
- VAE decoder training moved to `train_vae/`: the FLUX.1 trainer is
  `train_decoder.py` and the FLUX.2 trainer (internal `train_vae_vae.py` from
  the FLUX.2 tree, which loads the FLUX.2 AE and its 16x latent grid) is
  `train_decoder_flux2.py`. Launchers: `scripts/train_vae_flux1.sh`,
  `scripts/train_vae_flux2.sh`.
- Absolute cluster paths replaced by repo-relative defaults
  (`data/`, `checkpoints/`, `outputs/`) or environment variables
  (`FLUX_DEV_MODEL_PATH`, `AE_MODEL_PATH`, `T5`, `CLIP`,
  `KLEIN_9B_BASE_MODEL_PATH`, `QWEN3_8B_PATH`, `DECODER_CKPT`).
- Dropped from the internal trees: profiling artefacts, ablation variants
  (`train_vae_super_fft.py`, `train_vae_super_wavelet.py`), attention
  visualisation scripts, Gradio demo, and FLUX.2 helpers unused by UltraTex
  (`watermark.py`, `openrouter_api_client.py`).

## Functional fixes

### Running from the README layout

The launch scripts were checked from a clean copy holding only `checkpoints/`
(README layout) and the bundled `data/`. Fixes needed for that:

- Bundled training data: `data/train_demo.json` (4 objects) and
  `data/eval_demo.json` are the defaults for every training script.
- `train_flux2_9b_2048.sh`: train/eval JSON, resume checkpoint and bucket
  metadata are now configurable and optional. Without them it resumes from
  `checkpoints/flux2/lora` if present, otherwise trains from scratch.
  `TASK=mr` runs the metallic-roughness trainer.
- `ultratex/data/dataset*.py`: without bucket metadata every sample gets the
  same cost; previously `lengths` was empty and training crashed.
- Resuming from a released `lora/` folder crashed because the step number was
  parsed from a `checkpoint-N` name; non-matching names now start at step 0.
- `train_flux1.py` ignored `max_train_steps` until the epoch ended; it now
  stops at the limit.
- `inference_flux2.sh` called scripts that do not exist in the release
  (`inference.py`, `inference_ai-generated.py`); it now runs
  `inference_flux2.py` and accepts `--task mr`.
- FLUX.1 launchers default T5 / CLIP to `checkpoints/base/` and accept
  `MAIN_PROCESS_PORT` (also exported for DeepSpeed so it does not fall back
  to port 29500).
- `train_vae/`: the decoder trainers and evaluators read paths from the
  environment / CLI instead of internal JSONs; the FLUX.1 decoder now uses
  activation checkpointing so 2048 training fits in 80 GB (outputs and
  gradients are bitwise identical).

### `ultratex/backbones/flux1/modules/conditioner.py`

`HFEmbedder` accepted a `version` argument but ignored it, hardcoding
`"XLabs-AI/xflux_text_encoders"` and `"openai/clip-vit-large-patch14"`. This
worked internally because those repos were reachable or already cached. It now
honours `version`, so the `T5` and `CLIP` environment variables can point at a
local directory and the code runs offline.

### `inference_flux1.py`

Removed two stray characters accidentally typed at the start of line 1, which
made the module fail to import.

## Vendored dependency

`ultratex/sparse_attention/` vendors [SLA](https://github.com/thu-ml/SLA)
(upstream commit `552a51f`) with four changes: the linear-attention branch
(`proj_l`) is removed so only sparse block attention is used, and three
kernel correctness fixes for Triton 3.x and H100 (`other=0.0` on masked
loads, unconditional tail masking, explicit fp32 reduction). Upstream SLA
will not reproduce our numerics. Details in
`ultratex/sparse_attention/PATCHES.md`.

## Verification

Static checks: 27 modules and entry points import cleanly; all Python and shell
files parse; no absolute cluster paths remain.

End-to-end runs on 10 G-buffer TexVerse objects at the training resolution
(2048 per view, 25 sampling steps, guidance 4), one fixed reference render per
object (`render_ref_0/000.webp`; the loaders pick a reference at random, so
the test pinned it by patching the loader in memory, without editing this tree):

| Backbone | Checkpoint | Decoder | Views aligned with input normals |
|---|---|---|---|
| FLUX.1-dev | LoRA step 27000 | `decoder_step3000.pt` | 10/10 objects |
| FLUX.2-Klein 9B | LoRA step 38000 | `decoder_step600.pt` | 10/10 objects |

Alignment was checked per view against the input normals and the albedo ground
truth. An automatic edge-map correlation agrees on 9/10 objects; it flags the
tenth, a near-uniformly black loudspeaker whose side faces have flat, identical
normals, but side-by-side comparison with the ground truth shows the front
panel in the correct view for both backbones. Edge correlation is unreliable on
flat, textureless faces. The bundled demo asset is one of the 10 objects.
Background Token Dropping was active in every run (sequence `L=212992`, with
18K-101K image tokens kept per object depending on silhouette size).

The five issues below were found while running this tree and are fixed:

1. `config/deepspeed/*.json` was missing from the export.
2. `HFEmbedder` ignored its `version` argument (see above).
3. `inference_flux1.py` also needs `--train_data_json`; the launch script now
   passes it.
4. The dataset `data_root_v2` default overrode `image_dir` from the JSON.
5. `bump_normal_recover` does not exist in released data — see below.
6. `inference_flux2.py` imports its trainer at call time (`import train_lora`),
   which the rename to `train_flux2.py` broke. The metallic-roughness trainer
   was also missing from the export; it is now `train_flux2_mr.py`, reachable
   via `--task mr`. The thin `inference_MR.py` wrapper was dropped since
   `--task mr` covers it.

### `bump_normal_recover` → `bump_normal_world`

The renderer writes world-space normals to `bump_normal_recover` and then
renames that directory to `bump_normal_world`. The FLUX.1 entry points still
read the pre-rename name in five places, so they failed on any released asset.
All five now read `bump_normal_world`.
