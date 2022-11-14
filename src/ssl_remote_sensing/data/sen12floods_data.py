import gdown
import tarfile
import os
import rasterio as rio
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import cv2
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch.nn as nn
import torch.nn.functional as F
import random


gdown.download(
    "https://drive.google.com/u/1/uc?id=1zGalZSCxgnmZM7zMm0qfRrnOXyD6IUX1&export=download"
)

data_base_path = "/content/"
data_folder = "lab-seg-data"
tar_path = os.path.join(data_base_path, data_folder + ".tar.gz")

with tarfile.open(tar_path, mode="r") as tar:
    tar.extractall(path=data_base_path)


class SEN12FLOODS:
    """SEN12FLOODS Segmentation Dataset."""

    def __init__(self, root="chips/", split="train", transforms=None, **kwargs):
        super(SEN12FLOODS, self).__init__()

        # Loop over available data and create pairs of Sentinel 1 and Sentinel 2 images, co-registered,
        # with corresponding groundtruth, and store the paths in lists.
        (
            self.s2_images,
            self.s2_masks,
            self.s1_images,
            self.s1_masks,
        ) = self._get_sen2flood_pairs(root, split)

        # Make sure that for each data point we have all the values we need.
        assert (
            len(self.s2_images)
            == len(self.s2_masks)
            == len(self.s1_images)
            == len(self.s1_masks)
        )
        if len(self.s2_images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")

        print(
            "Found {} images in the folder {}".format(len(self.s2_images), root + split)
        )

        self.transforms = transforms

        # Initialise the data augmentation we will use: horizontal and vertical flipping, random affine translation, resizing
        if self.transforms:
            augmentation = A.Compose(
                [
                    A.Resize(
                        height=256, width=256, p=1, interpolation=cv2.INTER_NEAREST
                    ),
                    A.Affine(scale=2, translate_px=5, rotate=20, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    ToTensorV2(),
                ],
                additional_targets={"image0": "image"},
            )
            self.augmentation = augmentation

        else:
            augmentation = A.Compose(
                [
                    A.Resize(
                        height=256, width=256, p=1, interpolation=cv2.INTER_NEAREST
                    ),
                    ToTensorV2(),
                ],
                additional_targets={"image0": "image"},
            )
            self.augmentation = augmentation

        # turn lists into arrays
        self.s2_images = np.array(self.s2_images)
        self.s1_images = np.array(self.s1_images)
        self.s2_masks = np.array(self.s2_masks)
        self.s1_masks = np.array(self.s1_masks)

    def __len__(self):
        return len(self.s2_images)

    def __getitem__(self, index):
        # Loop over all bands, and create a concatenated array for sentinel-2 data
        bands = []
        for file in [
            "B1.tif",
            "B2.tif",
            "B3.tif",
            "B4.tif",
            "B5.tif",
            "B6.tif",
            "B7.tif",
            "B8.tif",
            "B8A.tif",
            "B9.tif",
            "B10.tif",
            "B11.tif",
            "B12.tif",
        ]:
            band = rio.open(os.path.join(self.s2_images[index], file))
            bands.append(band.read())
        s2_img = np.concatenate(bands, axis=0)
        s2_img = np.array(s2_img, dtype=np.float32)
        # CHANGE TO RGB
        s2_img = s2_img[[3, 2, 1], :, :]
        # NORMALIZATION
        s2_img = (s2_img - s2_img.min()) / (s2_img.max() - s2_img.min())

        # Loop over both polarization, and create a concatenated array for sentinel-1 data
        bands = []
        for file in ["VH.tif", "VV.tif"]:
            band = rio.open(os.path.join(self.s1_images[index], file))
            band_array = band.read()
            if np.isfinite(band_array).all():
                bands.append(band.read())
            else:
                bands.append(np.zeros(band_array.shape))
        s1_img = np.concatenate(bands, axis=0)
        s1_img = np.array(s1_img, dtype=np.float32)

        # The two channels of Sentinel-1 (VV and VH) have both negative and positive values.
        # We normalize them to lie between 0 and 1 by applying [min-max normalization with min = -77 and max = 26.
        s1_img = np.clip(s1_img, a_min=-77, a_max=26)
        s1_img = (s1_img + 77) / (26 + 77)

        # The water labels for Sentinel 1 and Sentinel 2 can be slightly different (since scenes are taken around 3 days apart)
        # We read the water label mask associated to Sentinel 2.
        mask = rio.open(self.s2_masks[index])
        mask_img = mask.read().squeeze()

        # Apply same data augmentation for both sentinel 2 and sentinel 1 images, and the mask.
        augmented_data = self.augmentation(
            image=np.transpose(s2_img, (1, 2, 0)),
            image0=np.transpose(s1_img, (1, 2, 0)),
            mask=mask_img,
        )

        # Define output tensor
        output_tensor = {
            "s2_img": augmented_data["image"],
            "s1_img": augmented_data["image0"],
            "s2_imgfile": self.s2_images[index],
            "s1_imgfile": self.s1_images[index],
            "mask": augmented_data["mask"],
        }

        return output_tensor

    def _get_sen2flood_pairs(self, folder, split):
        """
        Constructs Sentinel2 and Sentinel1 pairs

        Arguments
        ----------
            folder : str
                Image folder name
            split : str
                train or val split
        Returns
        -------
            s2_img_paths : list
                List of Sentinel 2 image path
            s2_mask_paths : list
                List of Sentinel 2 water mask path
            s1_img_paths : list
                List of Sentinel 1 image path
            s1_mask_paths : list
                List of Sentinel 1 water mask path
        """
        s2_img_paths = []
        s2_mask_paths = []
        s1_img_paths = []
        s1_mask_paths = []

        img_folder = os.path.join(folder, split)

        # loop over the image folder (train or validation)
        for filename in os.listdir(img_folder):
            if filename not in ["._.DS_Store", ".DS_Store"]:
                for file in os.listdir(os.path.join(img_folder, filename, "s2")):
                    if file not in ["._.DS_Store", ".DS_Store"]:
                        # Get the Image ID (as explained in the dataset section)
                        image_id = file.split("_")[-1]

                        # Store Sentinel 2 image and mask paths in lists
                        s2_imgpath = os.path.join(img_folder, filename, "s2", file)
                        s2_maskpath = os.path.join(
                            img_folder, filename, "s2", file, "LabelWater.tif"
                        )

                        # Using the Image ID, store co-registered Sentinel 1 image and mask paths in lists
                        s1_files = os.listdir(os.path.join(img_folder, filename, "s1"))
                        s1_file = [
                            file for file in s1_files if file.endswith(image_id)
                        ][0]
                        s1_imgpath = os.path.join(img_folder, filename, "s1", s1_file)
                        s1_maskpath = os.path.join(
                            img_folder, filename, "s1", s1_file, "LabelWater.tif"
                        )

                        if os.path.isfile(s1_maskpath):
                            s2_img_paths.append(s2_imgpath)
                            s2_mask_paths.append(s2_maskpath)
                            s1_img_paths.append(s1_imgpath)
                            s1_mask_paths.append(s1_maskpath)
                        else:
                            print("cannot find the S1 Mask:", s1_maskpath)

        return s2_img_paths, s2_mask_paths, s1_img_paths, s1_mask_paths

    def visualize_observation(self, idx):
        """
        Visualise Sentinel1, Sentinel2, and water mask.

        Arguments
        ----------
            idx : int
                Data index
        """
        sample = self.__getitem__(idx)

        s2_image = sample.get("s2_img").squeeze()
        s1_image = sample.get("s1_img").squeeze()
        mask = sample.get("mask")

        print(sample.get("s2_imgfile"))
        print(sample.get("s1_imgfile"))

        fig, axs = plt.subplots(1, 3, figsize=(17, 6))

        s1_img_vh = s1_image[0, :, :]
        s1_img_vh = s1_img_vh / s1_img_vh.max()

        axs[0].imshow(s1_img_vh)
        axs[0].set_title("Sentinel-1 VH")
        axs[0].axis("off")

        s2_img_rgb = s2_image
        # s2_img_rgb = s2_image[[3, 2, 1], :, :]
        s2_img_rgb = np.transpose(s2_img_rgb, (1, 2, 0))
        s2_img_rgb = s2_img_rgb / s2_img_rgb.max()

        axs[1].imshow(s2_img_rgb)
        axs[1].set_title("Sentinel-2 RGB")
        axs[1].axis("off")

        mask = mask.squeeze()

        axs[2].imshow(mask, cmap="Blues")
        axs[2].set_title("Groundtruth Mask")
        axs[2].axis("off")

        plt.show()


trainset = SEN12FLOODS(root="/content/chips/", transforms=True, split="train")

valset = SEN12FLOODS(root="/content/chips/", split="val")


train_loader = DataLoader(trainset, batch_size=8, pin_memory=True)

val_loader = DataLoader(valset, batch_size=8, pin_memory=True)
