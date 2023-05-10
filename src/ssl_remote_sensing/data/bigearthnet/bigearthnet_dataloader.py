import os
from pathlib import Path
from typing import Optional

from torch import nn
from torch.utils.data import DataLoader
from ssl_remote_sensing.data.bigearthnet.bigearthnet_in_memory_dataset import InMemoryBigearthnet

def get_bigearthnet_dataloader(
    data_dir: Path,
    batch_size: int,
    num_workers: Optional[int] = None,
    dataset_transform: Optional[nn.Module] = None,
    max_samples: Optional[int] = None,
):
    bigearthnet_dataset = InMemoryBigearthnet(
        dataset_dir=data_dir,
        transform=dataset_transform,
        max_samples=max_samples,
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
