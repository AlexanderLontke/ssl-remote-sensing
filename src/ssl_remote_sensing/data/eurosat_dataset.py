import numpy as np
import os
from typing import Union, Callable
from glob import glob
import rasterio as rio
from rasterio.plot import reshape_as_image
import torch
from torch.utils.data import Dataset

# Define classes
classes_to_int = {
    "AnnualCrop": 0,
    "Forest": 1,
    "HerbaceousVegetation": 2,
    "Highway": 3,
    "Industrial": 4,
    "Pasture": 5,
    "PermanentCrop": 6,
    "Residential": 7,
    "River": 8,
    "SeaLake": 9,
}
classes_to_label = {
    0: "AnnualCrop",
    1: "Forest",
    2: "HerbaceousVegetation",
    3: "Highway",
    4: "Industrial",
    5: "Pasture",
    6: "PermanentCrop",
    7: "Residential",
    8: "River",
    9: "SeaLake",
}

# dataset = EuroSAT(
#     "/Users/alexanderlontke/datasets/",
#     #transform=Augment(64),
#     download=True
# )
# all_images = np.array([np.array(dataset.__getitem__(i)[0]) for i in range(len(dataset))])
# n, h, w, c = all_images.shape
# means = [np.mean(all_images[:, :, :, i]) for i in range(c)]
means = [87.81586935763889, 96.97416420717593, 103.98142336697049]
stds = [51.67849701591506, 34.908630837585186, 29.465280593587384]


def euro_sat_target_transform(label_str: str) -> int:
    return classes_to_int[label_str]


class EuroSATDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        transform: Union[Callable, None],
        target_transform=Union[Callable, None],
    ):
        self.samples = glob(os.path.join(dataset_dir, "*", "*.tif"))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # Extract bands
        with rio.open(sample, "r") as d:
            ms_channels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            image = d.read(ms_channels)
            image = reshape_as_image(image)
            image = image.astype(np.float)

        # Extract label
        label = sample.split("/")[-1].split("_")[0]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
