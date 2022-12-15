from pathlib import Path
import os

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
    def __init__(self, dataset_dir, bands=ALL_BANDS, transform=None):
        self.bands = bands
        self.transform = transform
        self.samples = glob.glob(os.path.join(dataset_dir, "*"))
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
        return self.images[index], patch_id

    def __len__(self):
        return len(self.samples)
