
from torchvision import transforms
from pytorch_lightning import LightningDataModule

from ssl_remote_sensing.data.bigearthnet_dataset import Bigearthnet
from torch.utils.data import DataLoader

ALL_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

class BigearthnetDataModule(LightningDataModule):

    def __init__(self, data_dir, bands=ALL_BANDS, batch_size=32, num_workers=16, seed=42, train_transforms=None):
        super().__init__()
        self.data_dir = data_dir
        self.bands = bands
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.train_dataset = None
        self.train_transforms = train_transforms

    def setup(self):
        train_transforms = self.train_transform() if self.train_transforms is None else self.train_transforms

        self.train_dataset = Bigearthnet(
            dataset_dir=self.data_dir,
            #bands=self.bands,
            transform=train_transforms
        )

    @staticmethod
    def train_transform():
        return transforms.Compose([
            transforms.RandomCrop(64),
            #transforms.ToTensor()
        ])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
