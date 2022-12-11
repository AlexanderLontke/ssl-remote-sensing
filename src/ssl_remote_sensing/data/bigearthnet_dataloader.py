
from torchvision import transforms
from pytorch_lightning import LightningDataModule

from ssl_remote_sensing.data.bigearthnet_dataset import Bigearthnet
from torch.utils.data import DataLoader


class BigearthnetDataModule(LightningDataModule):

    def __init__(self, data_dir, bands=None, batch_size=32, num_workers=16, seed=42):
        super().__init__()
        self.data_dir = data_dir
        self.bands = bands
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.train_dataset = None

    def setup(self):
        train_transforms = self.train_transform() if self.train_transforms is None else self.train_transforms

        self.train_dataset = Bigearthnet(
            root=self.data_dir,
            #bands=self.bands,
            transform=train_transforms
        )

    @staticmethod
    def train_transform():
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
