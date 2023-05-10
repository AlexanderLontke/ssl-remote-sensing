from pathlib import Path
import os
from typing import Optional

import numpy as np
import rasterio
from torch.utils.data import Dataset
import torch
import glob
from rasterio.enums import Resampling
from tqdm import tqdm
from ssl_remote_sensing.data.bigearthnet.constants import ALL_BANDS, BAND_STATS

def normalize(img, mean, std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


class InMemoryBigearthnet(Dataset):
    def __init__(
        self,
        dataset_dir,
        bands=ALL_BANDS,
        transform=None,
        max_samples: Optional[int] = None,
    ):
        self.bands = bands
        self.transform = transform
        self.samples = glob.glob(os.path.join(dataset_dir, "*"))
        if max_samples:
            self.samples = self.samples[:max_samples]
        self.image_size = 120
        self.images = []
        for path in tqdm(self.samples, desc="Loading samples"):
            patch_id = path.split("/")[-1]

            channels = []
            for b in self.bands:
                ch = rasterio.open(Path(path, f"{patch_id}_{b}.tif")).read(
                    1,
                    out_shape=(1, self.image_size, self.image_size),
                    resampling=Resampling.bilinear,
                )
                ch = normalize(ch, mean=BAND_STATS["mean"][b], std=BAND_STATS["std"][b])
                channels.append(ch)
            img = np.dstack(channels)
            img = torch.from_numpy(img)
            img = torch.permute(img, (2, 0, 1))

            if self.transform is not None:
                img = self.transform(img)

            self.images += [img]

    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return len(self.samples)


# def normalize_func(img, mean, std):
#     min_value = mean - 2 * std
#     max_value = mean + 2 * std
#     img = (img - min_value) / (max_value - min_value) * 255.0
#     img = np.clip(img, 0, 255).astype(np.uint8)
#     return img

# # def normalize(img, mean, std):
# #     img = np.array(img)
# #     min_value = np.percentile(img, 2,  axis=(0,1))
# #     max_value = np.percentile(img, 98, axis=(0,1))
# #     img = (img - min_value) / (max_value - min_value)
# #     # img = np.clip(img, 0, 255).astype(np.float64)
# #     return img


# class InMemoryBigearthnet(Dataset):
#     def __init__(self, dataset_dir, bands=ALL_BANDS, transform=None, normalize = False, max_samples: Optional[int] = None):
#         self.bands = bands
#         self.transform = transform
#         self.normalize = normalize
#         self.samples = glob.glob(os.path.join(dataset_dir, "*"))
#         if max_samples:
#             self.samples = self.samples[:max_samples]
#         self.image_size = 120
#         self.images = []
#         for path in tqdm(self.samples, desc="Loading samples"):
#             patch_id = path.split("/")[-1]

#             channels = []
#             for b in self.bands:
#                 ch = rasterio.open(Path(path, f"{patch_id}_{b}.tif")).read(
#                     1,
#                     out_shape=(1, self.image_size, self.image_size),
#                     resampling=Resampling.bilinear,
#                 )
#                 if self.normalize:
#                     ch = normalize_func(ch, mean=BAND_STATS["mean"][b], std=BAND_STATS["std"][b])
#                     ch = ch/ch.max()
#                 else:
#                     ch = np.array(ch).astype(np.unit8)
#                     # print("Debug: ch type", type(ch))
#                     ch = ch/ch.max()
#                 channels.append(ch)
#             img = np.dstack(channels)
#             img = torch.from_numpy(img)
#             img = torch.permute(img, (2, 0, 1))

#             if self.transform is not None:
#                 img = self.transform(img)

#             self.images += [img]

#     def __getitem__(self, index):
#         return self.images[index]

#     def __len__(self):
#         return len(self.samples)
