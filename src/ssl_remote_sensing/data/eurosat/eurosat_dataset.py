import numpy as np
import os
from typing import Union, Callable
from glob import glob
import rasterio as rio
from tqdm import tqdm
from rasterio.plot import reshape_as_image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

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
means_tuple = (
    1353.7269257269966,
    1117.2022923538773,
    1041.8847248444733,
    946.5542548737702,
    1199.1886644965277,
    2003.0067999222367,
    2374.008444688585,
    2301.2204385489003,
    732.1819500777633,
    1820.6963775318286,
    1118.2027229275175,
    2599.7829373281975,
)
stds_tuple = (
    65.29657739037496,
    153.77375864458085,
    187.69931299271406,
    278.1246366855392,
    227.92409611864002,
    355.9331571735718,
    455.13290021052626,
    530.7795614455541,
    98.92998227431653,
    378.16138952053035,
    303.10651348740964,
    502.16376466306053
)

# def batch_mean_and_sd(loader):
#     imgs = []
#     for item in tqdm(loader.dataset):
#       imgs += [item[0]] # item[0] and item[1] are image and its label
#     imgs = torch.stack(imgs, dim=0).numpy()
#     print(imgs.shape)
#     b, c, h, w = imgs.shape
#     m = []
#     s = []
#     for i in range(c):
#         m += [imgs[:,i,:,:].mean()]
#         s += [imgs[:,i,:,:].std()]
#     return m, s

train_means = [
    1354.0996,
    1117.529,
    1042.4503,
    946.56146,
    1199.1011,
    2005.2098,
    2377.4353,
    2304.2144,
    731.7947,
    12.091442,
    1117.7218,
    2603.26
]

train_stds = [
    245.4707,
    333.48648,
    395.30338,
    594.32806,
    566.45123,
    862.2661,
    1089.3525,
    1120.208,
    403.872,
    4.717908,
    761.2879,
    1234.1173
]


def get_eurosat_normalizer():
    return T.Normalize(mean=train_means, std=train_stds)


def euro_sat_target_transform(label_str: str) -> int:
    return classes_to_int[label_str]


class EuroSATDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        transform: Union[Callable, None] = None,
        target_transform: Union[Callable, None] = None,
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
            ms_channels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]
            image = d.read(ms_channels)
            image = torch.tensor(image.astype(float))
            image = image.float()

        # Extract label
        label = sample.split("/")[-1].split("_")[0]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        else:
            label = euro_sat_target_transform(label)
        return image, label


class InMemoryEuroSATDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        transform: Union[Callable, None] = None,
        target_transform: Union[Callable, None] = None,
    ):
        self.samples = glob(os.path.join(dataset_dir, "*", "*.tif"))
        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        for sample in tqdm(self.samples, desc="Loading Images"):
            # Extract bands
            with rio.open(sample, "r") as d:
                ms_channels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]
                image = d.read(ms_channels)
                image = torch.tensor(image.astype(float))
                image = image.float()

            # Extract label
            label = sample.split("/")[-1].split("_")[0]

            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            else:
                label = euro_sat_target_transform(label)
            self.images += [(image, label)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]
