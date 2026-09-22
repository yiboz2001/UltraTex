<div align="center">
<h1 align="center">
  UltraTex: Unleashing 2K Multi-View Diffusion <br> for 3D Texturing
</h1>

  <a href="https://yiboz2001.github.io/UltraTex/"><img src="https://img.shields.io/badge/Project%20Page-UltraTex-blue"></a> &nbsp;
  <a href="https://arxiv.org/abs/2609.23169"><img src="https://img.shields.io/badge/arXiv-2609.23169-b31b1b.svg?logo=arXiv"></a> &nbsp;
  <a href="https://huggingface.co/datasets/YiboZhang2001/G-buffer-TexVerse"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-G--buffer%20TexVerse-blue"></a> &nbsp;
  <a href="https://www.youtube.com/watch?v=9tOsRvGxfN4"><img src="https://img.shields.io/badge/YouTube-Video-red?logo=youtube"></a>
<br>
<strong>SIGGRAPH Asia 2026</strong>
<br>
<strong>Yibo Zhang<sup>1,2</sup></strong>,
Ze Yuan<sup>3</sup>,
Nan Cao<sup>4,2</sup>,
Li Zhang<sup>5,2</sup>,
Yan-Pei Cao<sup>6</sup>,
Yuan-Chen Guo<sup>6</sup>,
Rui Ma<sup>1 &dagger;</sup>
<br>
<sup>1</sup>Jilin University &nbsp;
<sup>2</sup>Shanghai Innovation Institute &nbsp;
<sup>3</sup>The University of Hong Kong
<br>
<sup>4</sup>Tongji University &nbsp;
<sup>5</sup>Fudan University &nbsp;
<sup>6</sup>VAST
<br>
<sup>&dagger;</sup> Corresponding author
</div>

![UltraTex](https://yiboz2001.github.io/UltraTex/assets/teaser.jpg)

**UltraTex** is an efficient end-to-end framework for high-resolution multi-view diffusion-based 3D texturing at **2048×2048**. Object-centric multi-view renderings contain two major sources of redundancy: background-induced sequence redundancy and sparse token interactions within the foreground. UltraTex addresses them with:

1. **Background Token Dropping (BTD)** — discard background tokens before the DiT backbone, keeping original RoPE indices so spatial coordinates and the multi-view geometric prior survive.
2. **Block-Sparse Attention (BSA)** — top-*k* sparse attention over the retained foreground sequence.
3. **Foreground-Aware VAE Decoding** — replace the undenoised background latent with a canonical in-distribution background and lightly fine-tune the decoder so 2K reconstruction stays artifact-free.

On common samples in G-buffer TexVerse, this yields **20.6×–91.1×** training speedup and **22.3×–74.6×** end-to-end inference speedup over the dense baseline.

Feel free to contact me ([ybzhang23@mails.jlu.edu.cn](mailto:ybzhang23@mails.jlu.edu.cn)) or open an issue if you have any questions or suggestions.

## News

- [2026-09-22] Paper available on [arXiv](https://arxiv.org/abs/2609.23169).
- [2026-09-22] Project page, code placeholder, and [G-buffer TexVerse](https://huggingface.co/datasets/YiboZhang2001/G-buffer-TexVerse) dataset card released.

## Code

Training and inference code will be released here. This repository currently hosts the [project page](https://yiboz2001.github.io/UltraTex/).

## G-buffer TexVerse

To train 2K multi-view diffusion we constructed **G-buffer TexVerse**, a large-scale ultra-high-resolution multi-view rendering dataset built on [TexVerse](https://github.com/yiboz2001/TexVerse). The public release is the **351,847**-asset BSDF rendering pool. UltraTex training applies two further filters (albedo entropy and AI-content removal) and uses a **268,365**-asset subset.

| Stage | Remaining |
|---|---|
| Raw TexVerse | 858K |
| Visual quality (GPT-5) | 402K |
| Non-BSDF filtering | 348K |
| Albedo entropy | 297K |
| AI-content removal | 268,365 |

The public release is larger than the training subset: **351,847 BSDF assets**. UltraTex training uses the 268,365-asset subset.

Every asset is rendered with Blender Cycles under two camera configurations that share intrinsics, object normalization, aspect-ratio-adaptive distance, and three sampled HDR lights (from a pool of 862 Poly Haven maps; index → asset in [`env_maps.json`](https://huggingface.co/datasets/YiboZhang2001/G-buffer-TexVerse/blob/main/env_maps.json)):

- **Canonical — 6 views** (used by UltraTex): azimuths 0°/90°/180°/270° at elevation 0°, plus top and bottom. Training reference images are rendered under the same three HDR maps.
- **Sphere — 36 views** (community release, not used by UltraTex): 12 azimuths × elevations {−40°, −20°, 30°}.

Per-view outputs include shading normals (camera & world), canonical coordinate maps, albedo, metallic/roughness where available, and shaded images, all with an alpha channel. Resolution is 2048² or 4096² according to the asset's native texture resolution. Source-texture split of the 351,847 assets: 1024 / 2048 / 4096 / 8192 = 102,254 / 147,261 / 79,198 / 23,134.

**Download:** [https://huggingface.co/datasets/YiboZhang2001/G-buffer-TexVerse](https://huggingface.co/datasets/YiboZhang2001/G-buffer-TexVerse)

## Citation

```bibtex
@inproceedings{zhang2026ultratex,
  author    = {Zhang, Yibo and Yuan, Ze and Cao, Nan and Zhang, Li and
               Cao, Yan-Pei and Guo, Yuan-Chen and Ma, Rui},
  title     = {UltraTex: Unleashing 2K Multi-View Diffusion for 3D Texturing},
  year      = {2026},
  isbn      = {979-8-4007-2842-6},
  publisher = {Association for Computing Machinery},
  address   = {Kuala Lumpur, Malaysia},
  url       = {https://doi.org/10.1145/3829340.3842299},
  doi       = {10.1145/3829340.3842299},
  booktitle = {SIGGRAPH Asia 2026 Conference Papers},
  series    = {SA Conference Papers '26},
  month     = dec
}
```

## License

This repo is released under the [MIT License](./LICENSE).
