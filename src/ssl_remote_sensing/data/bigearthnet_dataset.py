from pathlib import Path
import os

import numpy as np
import rasterio
from torch.utils.data import Dataset
import torch
import glob
from rasterio.enums import Resampling


ALL_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']


# Stats from https://github.com/ServiceNow/seasonal-contrast/blob/main/datasets/bigearthnet_dataset.py
BAND_STATS = {
    'mean': {
        'B01': 340.76769064,
        'B02': 429.9430203,
        'B03': 614.21682446,
        'B04': 590.23569706,
        'B05': 950.68368468,
        'B06': 1792.46290469,
        'B07': 2075.46795189,
        'B08': 2218.94553375,
        'B8A': 2266.46036911,
        'B09': 2246.0605464,
        'B11': 1594.42694882,
        'B12': 1009.32729131
    },
    'std': {
        'B01': 554.81258967,
        'B02': 572.41639287,
        'B03': 582.87945694,
        'B04': 675.88746967,
        'B05': 729.89827633,
        'B06': 1096.01480586,
        'B07': 1273.45393088,
        'B08': 1365.45589904,
        'B8A': 1356.13789355,
        'B09': 1302.3292881,
        'B11': 1079.19066363,
        'B12': 818.86747235
    }
}


def normalize(img, mean, std):
    min_value = mean - 2 * std
    max_value = mean + 2 * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


class Bigearthnet(Dataset):

    def __init__(self, dataset_dir, bands=ALL_BANDS, transform=None):
        self.bands = bands
        self.transform = transform
        self.samples = glob.glob(os.path.join(dataset_dir, "*"))
        self.image_size = 120


    def __getitem__(self, index):
        path = self.samples[index]
        patch_id = path.split("/")[-1]

        channels = []
        for b in self.bands:
            ch = rasterio.open(Path(path, f'{patch_id}_{b}.tif')).read(
                1,
                out_shape=(
                        1,
                        self.image_size, 
                        self.image_size
                    ),
                resampling=Resampling.bilinear
                )
            ch = normalize(ch, mean=BAND_STATS['mean'][b], std=BAND_STATS['std'][b])
            channels.append(ch)   
        img = np.dstack(channels)
        img = torch.from_numpy(img)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.samples)