from torch import nn
from typing import Optional, List
from torchvision import transforms
from pytorch_lightning import LightningDataModule

from torch.utils.data import DataLoader
from ssl_remote_sensing.data.bigearthnet.bigearthnet_dataset import Bigearthnet
from ssl_remote_sensing.data.bigearthnet.constants import ALL_BANDS


class BigearthnetDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir,
        bands: Optional[List[str]] = None,
        batch_size: int = 32,
        num_workers: int = 16,
        seed: int = 42,
        train_transforms: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        if not bands:
            bands = ALL_BANDS
        self.bands = bands
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.train_dataset = None
        self.train_transforms = train_transforms

    def setup(self, stage: Optional[str] = None):
        train_transforms = (
            self.default_transform()
            if self.train_transforms is None
            else self.train_transforms
        )

        self.train_dataset = Bigearthnet(
            dataset_dir=self.data_dir,
            transform=train_transforms,
        )

    @staticmethod
    def default_transform():
        return transforms.Compose(
            [
                # transforms.RandomCrop(64),
            ]
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
