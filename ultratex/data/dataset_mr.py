# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Copyright (c) 2024 Black Forest Labs and The XLabs-AI Team. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from PIL import Image, ImageOps
import os
import numpy as np
import cv2
import numpy as np
import random
import os
import json
import os
import time
import random
import numpy as np
import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader, Dataset, Sampler 

TEXVERSE_V2_EXCLUDED_IDS = {
    "e259f7110da345dc990800bf82bd46ac_1024",
    "f99a437c70d84fd1a0b63c1a1b98cfa8_2048",
    "afa16f06fb4644449509d4251056e2b2_1024",
    "bfd5a78eb7e94da0a37be64967ccdfc3_1024",
    "b15a5308bd5f4fb2ae23d8d09517657c_2048",
    "d14c40220858437bb81e88ac70129446_1024",
    "47b3ddc50b1d490da1504be576ef904e_2048",
    "a34faba60e8f4bfc9c103d2c5fd660e4_8192",
    "7611739e781449c7a900eaf2c3b1b728_2048",
    "427f9ab917bc4e9593ffeacc33a187a8_2048",
    "85ca5fa99a7c4e09a465a1da3c0da486_2048",
    "40951513e3e04f9e81d34798c3327b78_2048",
    "00642e01e9884a17ad04978ad734d433_2048",
    "91d05683538f4eebbf9e76c828e7d329_2048",
    "873a2aa748fc41ac91106adf7ec1724e_4096",
    "1359166303204ab4bb48c0dea0d68e50_2048",
    "1523b5383de0446f8d90e38582a7e71f_1024",
    "1fbf12d99eae453d8a8d9c958e8c3033_2048",
    "28be6fe6296b4fac8e61565b5a586043_4096",
    "29054494117c4df5a9540917e03a67cb_2048",
    "4b4052286242484c83264752b0b1868a_2048",
    "4b7662bfd57c4be79b01c7b16819b06b_2048",
    "4cceeccd99c64931a5c903ba4628b1e8_2048",
    "4d2cccedab3d4f048216f1423d8786b7_2048",
    "59c460c3e6304086be4d9f209b2b4d74_2048",
    "67549616d13246cfb1a540f9d6969078_2048",
    "794a2591560144ad970d5b4fdfd06cf3_1024",
    "802719621f354e978ac4e5edaa4d3739_2048",
    "981f065141a34883aa667725cb2919d8_1024",
    "9981bfc75e35492ea105bc93fb20be01_1024",
    "9f349d3ec1dc4c3f9c3eefecc013ebb4_2048",
    "a2362ae283be4ecc86e3a18485c11aeb_1024",
    "a6179b5510594d71a7e48b20de128531_2048",
    "b5287e178b27450e890fa5ac21e9cc9c_1024",
    "c31d8295b60342c59931c94534f79e98_8192",
    "ddc5461f4e694010af175a8cdd1985fa_2048",
    "e1e7033e57684851b64716186a196f79_2048",
    "e2e4c642e696431699ef39c9a7d0352a_1024",
}

class LengthAwareBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=False, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.sorted_indices = self._sort_with_random_ties(dataset.lengths)
        self.sorted_lengths = np.array(dataset.lengths)[self.sorted_indices]
        self.groups = self._group_samples_by_length()

    def _sort_with_random_ties(self, lengths):
        # Accelerate shards this global order across ranks. A local RNG keeps
        # the order identical even when device-specific process seeds differ.
        rng = np.random.default_rng(self.seed)
        indices = np.argsort(lengths)
        sorted_lengths = np.array(lengths)[indices]
        unique_lengths = list(dict.fromkeys(sorted_lengths))
        final_indices = []
        for length in unique_lengths:
            same_length_indices = np.where(sorted_lengths == length)[0]
            rng.shuffle(same_length_indices)
            final_indices.extend(indices[same_length_indices])

        return final_indices
    
    def _group_samples_by_length(self):
        """
        Group samples by length so each group holds similar-length samples.
        """
        groups = []
        current_group = []

        for idx in self.sorted_indices:
            if len(current_group) < self.batch_size:
                current_group.append(idx)
            else:
                groups.append(current_group)
                current_group = [idx]

        if current_group:
            groups.append(current_group)

        return groups

    def __iter__(self):
        """
        Draw from each group so every batch keeps a similar length.
        """
        batch = []
        for group in self.groups:
            for idx in group:
                expanded_idx = idx
                batch.append(expanded_idx)

                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                    
            if batch and not self.drop_last:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        total_batches = sum([len(group) // self.batch_size for group in self.groups])

        if not self.drop_last:
            total_batches += sum([1 for group in self.groups if len(group) % self.batch_size != 0])

        return total_batches
    

class FluxPairedDatasetMR(Dataset):
    """Isolated TexVerse dataset whose prediction target is roughness_metallic."""
    def __init__(
        self,
        json_file: str,
        resolution: int,
        resolution_ref: int | None = None,
        image_dropout=-1,
        text_dropout=-1,
        kernel_pad=2,
        data_root=None,
        bucket_metadata_json: str | None = None,
    ):
        super().__init__()
        self.json_file = json_file
        self.resolution = resolution
        self.resolution_ref = resolution_ref if resolution_ref is not None else resolution
        self.image_root = os.path.dirname(json_file)
        self.samples_per_object = 4
        
        with open(self.json_file, "rt") as f:
            self.data_dicts = json.load(f)

        # Each entry's `image_dir` is used as given (relative to the working
        # directory). If `data_root` is set, objects are looked up as
        # <data_root>/<id> instead, e.g. for an extracted G-buffer TexVerse bucket.
        resolved_data_dicts = []
        skipped_not_in_v2 = []
        for data_dict in self.data_dicts:
            sample_id = data_dict.get("id") or os.path.basename(
                os.path.normpath(data_dict["image_dir"])
            )
            if sample_id in TEXVERSE_V2_EXCLUDED_IDS:
                skipped_not_in_v2.append(sample_id)
                continue

            if data_root is not None:
                image_dir = os.path.join(data_root, sample_id)
            else:
                image_dir = data_dict["image_dir"]
            data_dict["_resolved_image_dir"] = image_dir
            resolved_data_dicts.append(data_dict)

        self.data_dicts = resolved_data_dicts
        self.skipped_not_in_v2 = len(skipped_not_in_v2)

        # Roughness/metallic uses the same object alpha support as albedo, so
        # the shared foreground-percentage metadata is the correct cost key.
        self.lengths = []
        self.missing_bucket_metadata = 0
        if bucket_metadata_json is not None:
            with open(bucket_metadata_json, "rt") as f:
                length_by_id = json.load(f)
            for data_dict in self.data_dicts:
                sample_id = data_dict["id"]
                if sample_id in length_by_id:
                    sample_length = round(float(length_by_id[sample_id]), 4)
                else:
                    sample_length = 1.0
                    self.missing_bucket_metadata += 1
                self.lengths.extend([sample_length] * self.samples_per_object)
        else:
            # No metadata: every sample has the same cost, so bucketing is a no-op.
            self.lengths = [1.0] * (len(self.data_dicts) * self.samples_per_object)
        
        self.transform = Compose([
            ToTensor(),
            Normalize([0.5], [0.5]),
        ])

        self.kernel_pad = kernel_pad

    
    def __getitem__(self, idx):
        try:
            # data_dict = self.data_dicts[idx]
            data_dict = self.data_dicts[idx // self.samples_per_object]
            txt = data_dict["prompt"]
            image_dir = data_dict["_resolved_image_dir"]

            # === tiling helpers ===
            def load_and_concat(subdir, resolution):
                paths = [os.path.join(image_dir, subdir, f"{i:03d}.webp") for i in range(6)]

                imgs_big = []
                imgs_raw = []
                imgs_small = []
                masks_small = []
                masks_full = []
                masks_raw = []

                for p in paths:
                    img = Image.open(p).convert("RGBA")

                    # ========= small mask =========
                    w_small = h_small = resolution // 16
                    img_small = img.resize((w_small, h_small), Image.Resampling.BILINEAR)
                    imgs_small.append(img_small)
                    
                    alpha_small = np.array(img_small.split()[3])
                    mask_small = (alpha_small > 0).astype(np.uint8)
                    kernel = np.ones((self.kernel_pad, self.kernel_pad), np.uint8)
                    mask_small = cv2.dilate(mask_small, kernel, iterations=1)
                    masks_small.append(mask_small)

                    # ========= full image ---------
                    img_big = img.resize((resolution, resolution), Image.Resampling.BILINEAR)
                    bg = Image.new("RGB", img_big.size, (255,255,255))
                    bg.paste(img_big, mask=img_big.split()[3])
                    imgs_big.append(np.array(bg))

                    # ========= full resolution mask ========= 
                    alpha_full = np.array(img_big.split()[3])                        # original alpha, H x W
                    mask_full = (alpha_full > 0).astype(np.uint8)
                    masks_full.append(mask_full)

                    # ========= raw image =========
                    bg = Image.new("RGB", img.size, (255,255,255))
                    bg.paste(img, mask=img.split()[3])
                    imgs_raw.append(np.array(bg))
                    
                    alpha_raw = np.array(img.split()[3])                        # original alpha, H x W
                    mask_raw = (alpha_raw > 0).astype(np.uint8)
                    masks_raw.append(mask_raw)
                    
                row1_img = np.concatenate([imgs_small[0], imgs_small[1], imgs_small[3]], axis=1)
                row2_img = np.concatenate([imgs_small[2], imgs_small[4], imgs_small[5]], axis=1)
                final_img_small = Image.fromarray(np.concatenate([row1_img, row2_img], axis=0))
                                
                row1_img = np.concatenate([imgs_raw[0], imgs_raw[1], imgs_raw[3]], axis=1)
                row2_img = np.concatenate([imgs_raw[2], imgs_raw[4], imgs_raw[5]], axis=1)
                final_img_raw = Image.fromarray(np.concatenate([row1_img, row2_img], axis=0))

                row1_img = np.concatenate([imgs_big[0], imgs_big[1], imgs_big[3]], axis=1)
                row2_img = np.concatenate([imgs_big[2], imgs_big[4], imgs_big[5]], axis=1)
                final_img = Image.fromarray(np.concatenate([row1_img, row2_img], axis=0))
                
                row1_s = np.concatenate([masks_small[0], masks_small[1], masks_small[3]], axis=1)
                row2_s = np.concatenate([masks_small[2], masks_small[4], masks_small[5]], axis=1)
                final_mask_small = np.concatenate([row1_s, row2_s], axis=0)

                row1_f = np.concatenate([masks_full[0], masks_full[1], masks_full[3]], axis=1)
                row2_f = np.concatenate([masks_full[2], masks_full[4], masks_full[5]], axis=1)
                final_mask_full = np.concatenate([row1_f, row2_f], axis=0)     
                
                row1_f = np.concatenate([masks_raw[0], masks_raw[1], masks_raw[3]], axis=1)
                row2_f = np.concatenate([masks_raw[2], masks_raw[4], masks_raw[5]], axis=1)
                final_mask_raw = np.concatenate([row1_f, row2_f], axis=0)   

                # final_img_small.save(
                #     os.path.join("outputs/mask_vis", f"{subdir}_img_small.png")
                # )

                # final_img.save(
                #     os.path.join("outputs/mask_vis", f"{subdir}_img.png")
                # )

                # Image.fromarray(final_mask_small * 255).convert("L").save(
                #     os.path.join("outputs/mask_vis", f"{subdir}_mask_small.png")
                # )

                # Image.fromarray(final_mask_full * 255).convert("L").save(
                #     os.path.join("outputs/mask_vis", f"{subdir}_mask_full.png")
                # )
                
                return final_img, final_mask_small, final_mask_full, final_img_raw, final_mask_raw

            ref_imgs = []
            masks = []
            masks_full = []       
            masks_raw = [] 
            
            ref_id = idx % self.samples_per_object
            
            img, mask, mask_full, raw_img, mask_raw = load_and_concat(
                "roughness_metallic", self.resolution
            )
            
            masks.append(torch.from_numpy(np.array(mask)).float())
            masks_full.append(torch.from_numpy(np.array(mask_full)).float())
            masks_raw.append(torch.from_numpy(np.array(mask_raw)).float())
            
            # img, mask, mask_full = load_and_concat(f"render_{ref_id}")
            # masks.append(torch.from_numpy(np.array(mask)).float())
            # masks_full.append(torch.from_numpy(np.array(mask_full)).float())
            
            # img_ref, mask_ref = load_and_concat("position")
            # ref_imgs.append(img_ref)         
            # masks.append(torch.from_numpy(np.array(mask_ref)).float())
   
            img_ref, mask_ref, _, _, _ = load_and_concat("bump_normal_world", self.resolution)
            ref_imgs.append(img_ref)
            masks.append(torch.from_numpy(np.array(mask_ref)).float())

            # --------------------------
            
            if ref_id == 3:
                rand = random.randint(0, 2)
                render_path = os.path.join(image_dir, f"render_{rand}", "000.webp")
            else:
                rand = random.randint(0, 3)
                rand_str = f"{rand:03d}"
                render_path = os.path.join(image_dir, f"render_ref_{ref_id}", f"{rand_str}.webp")
                
            # rand = random.randint(0, 3)
            # rand_str = f"{rand:03d}"
            # render_path = os.path.join(image_dir, f"render_ref_{ref_id}", f"{rand_str}.webp")

            if not os.path.exists(render_path):
                raise FileNotFoundError(f"Missing render file: {render_path}")
            
            render_img = Image.open(render_path).convert("RGBA")
            render_img = render_img.resize((self.resolution, self.resolution), Image.Resampling.BILINEAR)
            bg = Image.new("RGB", render_img.size, (255, 255, 255))
            bg.paste(render_img, mask=render_img.split()[3])
            ref_imgs.append(bg)
    
            img_small = render_img.resize((self.resolution // 16, self.resolution // 16), Image.Resampling.BILINEAR)
            alpha_small = np.array(img_small.split()[3])
            mask_small = (alpha_small > 0).astype(np.uint8)
            kernel = np.ones((2, 2), np.uint8)
            mask_small = cv2.dilate(mask_small, kernel, iterations=1)                    
            masks.append(torch.from_numpy(np.array(mask_small)).float())

            # Empty alpha masks cannot be pruned into foreground tokens.  If one
            # reaches the training loop, only that rank raises while every other
            # rank enters the next collective, which turns the data error into a
            # 10-minute NCCL timeout.  Treat it like the other per-sample loading
            # failures handled below and use the known-good fallback sample.
            mask_labels = (
                "target/roughness_metallic",
                "reference[0]/bump_normal_world",
                "reference[1]/render",
            )
            for mask_label, foreground_mask in zip(mask_labels, masks, strict=True):
                if not torch.any(foreground_mask):
                    raise ValueError(
                        f"{mask_label} has no foreground after mask resize "
                        f"for sample {data_dict['id']}"
                    )
            
            ref_imgs = [self.transform(im) for im in ref_imgs]
            
            img = self.transform(img)
            raw_img = self.transform(raw_img)

            return {
                "id": data_dict["id"],
                "prompt": txt,
                "ref_imgs": ref_imgs,
                "masks": masks,
                "masks_full": masks_full, 
                "masks_raw": masks_raw,
                "img": img,
                "raw_img": raw_img,
                "image_dir": image_dir,
                "render_path": render_path,
                "ref_id": ref_id,
            }

        except Exception as e:
            if idx == 0:
                raise RuntimeError(f"Failed to process fallback data (idx=0): {e}") from e
            print(f"[Dataset] Error loading idx {idx}: {e}, falling back to idx=0")
            return self.__getitem__(0)  # retry with idx=0

    def __len__(self):
        return len(self.data_dicts) * self.samples_per_object
    
    def collate_fn(self, batch):
        id_list = [data["id"] for data in batch] 
        img = [data["img"] for data in batch]
        raw_img = [data["raw_img"] for data in batch]
        ref_imgs = [data["ref_imgs"] for data in batch]
        masks = [data["masks"] for data in batch]
        masks_full = [data["masks_full"] for data in batch]
        masks_raw = [data["masks_raw"] for data in batch]
        
        txt = [data["prompt"] for data in batch]
        image_dir = [data["image_dir"] for data in batch]
        render_path = [data["render_path"] for data in batch]
        ref_id = [data["ref_id"] for data in batch]
        
        assert all([len(ref_imgs[0]) == len(ref_imgs[i]) for i in range(len(ref_imgs))])
        n_ref = len(ref_imgs[0])
        
        img = torch.stack(img, dim=0)
        # raw_img = torch.stack(raw_img, dim=0)
        
        
        ref_imgs_new = []
        for i in range(n_ref):
            ref_imgs_i = [refs[i] for refs in ref_imgs]
            ref_imgs_i = torch.stack(ref_imgs_i)
            ref_imgs_new.append(ref_imgs_i)
    
        return {
            "id": id_list,
            "txt": txt,
            "img": img,
            "raw_img": raw_img,
            "ref_imgs": ref_imgs_new,
            "masks": masks,
            "masks_full": masks_full,
            "masks_raw": masks_raw,
            "image_dir": image_dir,
            "render_path": render_path,
            "ref_id": ref_id
        }


if __name__ == '__main__':
    import argparse
    from pprint import pprint
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str, default="data/train.json")
    args = parser.parse_args()
    dataset = FluxPairedDatasetMR(args.json_file, 1024, image_dropout=-1)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)
    print(len(dataset))
    for i, data_dict in enumerate(dataloder):
        pprint(i)
        pprint(data_dict["img"])   
        pprint(data_dict["ref_imgs"])
        
    # for i, data_dict in enumerate(dataloader):
    #     pprint(i)

    #     ref_imgs = data_dict["ref_imgs"]
    #     imgs = data_dict["img"]

    #     # flatten nested lists, e.g. [[img1, img2, img3]] -> [img1, img2, img3]
    #     if isinstance(ref_imgs, list) and len(ref_imgs) == 1 and isinstance(ref_imgs[0], list):
    #         ref_imgs = ref_imgs[0]

    #     save_root = "data"
    #     os.makedirs(save_root, exist_ok=True)

    #     for j, img in enumerate(ref_imgs):
    #         if not isinstance(img, Image.Image):
    #             # convert back to PIL if already a tensor
    #             from torchvision.transforms.functional import to_pil_image
    #             img = to_pil_image(img)

    #         save_path = os.path.join(save_root, f"sample_{i:04d}_ref_{j}.jpg")
    #         img.save(save_path)

    #     for j, img in enumerate(imgs):
    #         if not isinstance(img, Image.Image):
    #             # convert back to PIL if already a tensor
    #             from torchvision.transforms.functional import to_pil_image
    #             img = to_pil_image(img)

    #         save_path = os.path.join(save_root, f"sample_{i:04d}_{j}.jpg")
    #         img.save(save_path)

    #     print(f"✅ Saved sample {i} ({len(ref_imgs)} ref_imgs)")
