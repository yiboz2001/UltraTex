# Data format

Four demo objects ship with the repository, so every script runs without
downloading data:

```
data/train_demo.json          # all four objects (training default)
data/eval_demo.json           # first object (evaluation / inference default)
data/demo.json                # same as eval_demo.json
data/<id>_<res>/              # one directory per object
```

They are canonical 6-view assets from
[G-buffer TexVerse](https://huggingface.co/datasets/YiboZhang2001/G-buffer-TexVerse)
and include `roughness_metallic/`.

Training and inference read a JSON list of objects. Each entry's `image_dir`
points at one asset directory, relative to the repository root:

```json
[
  {
    "prompt": "",
    "image_dir": "data/<id>_<res>",
    "id": "<id>_<res>",
    "id_only": "<id>"
  }
]
```

To use your own data, extract assets anywhere and write a JSON in this format.

| Field | Description |
|---|---|
| `prompt` | Text prompt. Empty string for image-guided texturing. |
| `image_dir` | Directory holding the multi-view renderings of this asset. |
| `id` | `<sha>_<source_texture_res>`, matches the directory name. |
| `id_only` | Asset SHA without the resolution suffix. |

## Expected layout of `image_dir`

Each asset directory follows the
[G-buffer TexVerse](https://huggingface.co/datasets/YiboZhang2001/G-buffer-TexVerse)
canonical (6-view) layout:

```
<id>_<res>/
├── albedo/000.webp … 005.webp             # target texture
├── bump_normal_camera/000.webp … 005.webp  # geometric condition
├── bump_normal_world/000.webp … 005.webp
├── position/000.webp … 005.webp
├── roughness_metallic/000.webp … 005.webp  # PBR assets only
├── render_0/  render_1/  render_2/         # shaded views, 3 HDR lightings
├── pose/000.npy … 005.npy
├── render_ref_0/000.webp … 003.webp, env_id.txt   # reference images
├── render_ref_1/  render_ref_2/
├── render_ref_{0,1,2}_poses/000.npy … 003.npy
├── env_indices.txt                         # 3 HDR ids, e.g. [686, 303, 134]
└── intrinsics.npy
```

`env_indices.txt` indexes the 862-entry Poly Haven pool; the id → asset mapping
is `env_maps.json` in the dataset repository.

## Preparing your own data

1. Download one or more buckets from
   [G-buffer TexVerse](https://huggingface.co/datasets/YiboZhang2001/G-buffer-TexVerse):
   ```bash
   python -c "
   from huggingface_hub import hf_hub_download
   hf_hub_download('YiboZhang2001/G-buffer-TexVerse',
                   'canonical/bsdf/00.zip', repo_type='dataset')"
   ```
2. Unpack the bucket, then the per-asset zips inside it.
3. Write a JSON list following the schema above, pointing `image_dir` at each
   unpacked asset directory.

`bucket_metadata_json` (default `data/average_percentages.json`) holds the
per-asset foreground ratio used by the length-aware batch sampler. It maps
`id_only` to a float in `[0, 1]`. If absent, all assets default to ratio 1.0.
