from pytorch_lightning import LightningDataModule
import torchvision.transforms as transforms
from torchvision.datasets import EuroSAT
from torch.utils.data import DataLoader

class EuroSATDataModule(LightningDataModule):
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.data_dir = config.eurosat_data_dir
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.dims = (3, 64, 64)
        self.num_classes = 10

    def prepare_data(self):
        # download
        EuroSAT(self.data_dir, download=True)


    def setup(self, stage = None):
        if stage == "fit":
            self.eurosat_train = EuroSAT(self.data_dir, transform=self.transform)


    def train_dataloader(self):
        return DataLoader(
            self.eurosat_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
