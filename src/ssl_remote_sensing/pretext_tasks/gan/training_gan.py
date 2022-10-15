#!/usr/bin/env python
from ssl_remote_sensing.pretext_tasks.gan.gan import GAN
from pytorch_lightning import Trainer, WandbLogger
from ssl_remote_sensing.data.eurosat_torchvision import EuroSATDataModule
from ssl_remote_sensing.pretext_tasks.gan.config.core import config
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import torch

def main():
    data = EuroSATDataModule(config)
    model = GAN(config)
    wandb_logger = WandbLogger()
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=config.max_epochs,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
        logger=wandb_logger,
    )
    trainer.fit(model, data)

if __name__ == "__main__":
    main()