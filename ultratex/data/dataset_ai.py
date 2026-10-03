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


class FluxPairedDatasetAI(Dataset):
    def __init__(self, json_file: str, resolution: int, resolution_ref: int | None = None, image_dropout=-1, text_dropout=-1, kernel_pad=2, target_subdir="albedo"):
        super().__init__()
        self.json_file = json_file
        self.resolution = resolution
        self.resolution_ref = resolution_ref if resolution_ref is not None else resolution
        self.image_root = os.path.dirname(json_file)
        
        with open(self.json_file, "rt") as f:
            self.data_dicts = json.load(f)
        
        self.transform = Compose([
            ToTensor(),
            Normalize([0.5], [0.5]),
        ])

        self.kernel_pad = kernel_pad
        self.target_subdir = target_subdir

    @staticmethod
    def _resolve_shared_path(path):
        """Resolve stale TexVerse_Compare mount prefixes in archived case JSONs."""
        if path and os.path.exists(path):
            return path
        marker = "/TexVerse_Compare/"
        if path and marker in path:
            suffix = path.split(marker, 1)[1]
            local_path = os.path.join(
                "outputs",
                suffix,
            )
            if os.path.exists(local_path):
                return local_path
        return path

    
    def __getitem__(self, idx):
        try:
            data_dict = self.data_dicts[idx]
            txt = data_dict["prompt"]
            image_dir = self._resolve_shared_path(data_dict["image_dir"])

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
                    
                row1_img = np.concatenate([imgs_small[0], imgs_small[1], imgs_small[3]], axis=1)
                row2_img = np.concatenate([imgs_small[2], imgs_small[4], imgs_small[5]], axis=1)
                final_img_small = Image.fromarray(np.concatenate([row1_img, row2_img], axis=0))

                row1_img = np.concatenate([imgs_big[0], imgs_big[1], imgs_big[3]], axis=1)
                row2_img = np.concatenate([imgs_big[2], imgs_big[4], imgs_big[5]], axis=1)
                final_img = Image.fromarray(np.concatenate([row1_img, row2_img], axis=0))
                
                row1_s = np.concatenate([masks_small[0], masks_small[1], masks_small[3]], axis=1)
                row2_s = np.concatenate([masks_small[2], masks_small[4], masks_small[5]], axis=1)
                final_mask_small = np.concatenate([row1_s, row2_s], axis=0)

                row1_f = np.concatenate([masks_full[0], masks_full[1], masks_full[3]], axis=1)
                row2_f = np.concatenate([masks_full[2], masks_full[4], masks_full[5]], axis=1)
                final_mask_full = np.concatenate([row1_f, row2_f], axis=0)     

                return final_img, final_mask_small, final_mask_full

            ref_imgs = []
            masks = []
            masks_full = []
            
            ref_id = idx % 4
            
            img, mask, mask_full = load_and_concat(self.target_subdir, self.resolution)
            masks.append(torch.from_numpy(np.array(mask)).float())
            masks_full.append(torch.from_numpy(np.array(mask_full)).float())
   
            img_ref, mask_ref, _ = load_and_concat("bump_normal_world", self.resolution)
            # img_ref, mask_ref, _ = load_and_concat("bump_normal", self.resolution)
            
            ref_imgs.append(img_ref)
            masks.append(torch.from_numpy(np.array(mask_ref)).float())

            render_path = data_dict.get("ref_image_path")
            if render_path:
                render_path = self._resolve_shared_path(render_path)
            else:
                render_path = os.path.join(image_dir, data_dict["id"] + "_resize.webp")
                
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
            
            ref_imgs = [self.transform(im) for im in ref_imgs]
            img = self.transform(img)

            return {
                "id": data_dict["id"],
                "prompt": txt,
                "ref_imgs": ref_imgs,
                "masks": masks,
                "masks_full": masks_full, 
                "img": img,
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
        return len(self.data_dicts)
    
    
    def collate_fn(self, batch):
        id_list = [data["id"] for data in batch] 
        img = [data["img"] for data in batch]
        ref_imgs = [data["ref_imgs"] for data in batch]
        masks = [data["masks"] for data in batch]
        masks_full = [data["masks_full"] for data in batch]
        txt = [data["prompt"] for data in batch]
        image_dir = [data["image_dir"] for data in batch]
        render_path = [data["render_path"] for data in batch]
        ref_id = [data["ref_id"] for data in batch]
        
        assert all([len(ref_imgs[0]) == len(ref_imgs[i]) for i in range(len(ref_imgs))])
        n_ref = len(ref_imgs[0])
        
        img = torch.stack(img, dim=0)
        
        ref_imgs_new = []
        for i in range(n_ref):
            ref_imgs_i = [refs[i] for refs in ref_imgs]
            ref_imgs_i = torch.stack(ref_imgs_i)
            ref_imgs_new.append(ref_imgs_i)
    
        return {
            "id": id_list,
            "txt": txt,
            "img": img,
            "ref_imgs": ref_imgs_new,
            "masks": masks,
            "masks_full": masks_full,
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
    dataset = FluxPairedDatasetV2(args.json_file, 1024, image_dropout=-1)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)
    print(len(dataset))
    for i, data_dict in enumerate(dataloder):
        pprint(i)
        pprint(data_dict["img"])   
        pprint(data_dict["ref_imgs"])
