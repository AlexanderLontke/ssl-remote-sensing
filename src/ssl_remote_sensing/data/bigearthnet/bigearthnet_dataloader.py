import os
from pathlib import Path
from typing import Optional

from torch import nn
from torch.utils.data import DataLoader
from ssl_remote_sensing.data.bigearthnet.bigearthnet_dataset import Bigearthnet


def get_bigearthnet_dataloader(
    data_dir: Path,
    batch_size: int,
    num_workers: Optional[int] = None,
    dataset_transform: Optional[nn.Module] = None,
):
    bigearthnet_dataset = Bigearthnet(
        dataset_dir=data_dir,
        transform=dataset_transform,
    )
    if not num_workers:
        num_workers = os.cpu_count()
    return DataLoader(
        dataset=bigearthnet_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )



