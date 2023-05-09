import os
from pathlib import Path

import torch
from pytorch_lightning.callbacks import GradientAccumulationScheduler, ModelCheckpoint
import torchvision.transforms as T
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from ssl_remote_sensing.pretext_tasks.simclr.utils import reproducibility
from ssl_remote_sensing.pretext_tasks.simclr.training import SimCLRTraining
from ssl_remote_sensing.pretext_tasks.simclr.augmentation import Augment
from ssl_remote_sensing.pretext_tasks.simclr.config import get_simclr_config
from ssl_remote_sensing.data.bigearthnet.bigearthnet_dataloader import (
    get_bigearthnet_dataloader,
)

# Machine setup
available_gpus = torch.cuda.device_count()
save_model_path = os.path.join(os.getcwd(), "saved_models/")
print("available_gpus:", available_gpus)

# Model Setup
train_config = get_simclr_config()
train_config.batch_size = 4096
train_config.epochs = 40

# Run setup
filename = f"SimCLR_BEN_ResNet18_adam_bs{train_config.batch_size}"
save_name = filename + ".ckpt"
resume_from_checkpoint = False
reproducibility(train_config)

model = SimCLRTraining(
    config=train_config,
    feat_dim=512,
)

# Setup data loading and augments
transform = Augment(train_config.img_size, normalizer=T.RandomCrop(64))
bigearthnet_dataloader = get_bigearthnet_dataloader(
    data_dir=Path("/content/BigEarthNet-v1.0"),
    batch_size=train_config.batch_size,
    dataset_transform=transform,
)

# Needed to get simulate a large batch size
accumulator = GradientAccumulationScheduler(scheduling={0: 1})

checkpoint_callback = ModelCheckpoint(
    filename=filename,
    dirpath=save_model_path,
    every_n_epochs=5,
    save_last=True,
    save_top_k=2,
    monitor="train/NTXentLoss",
    mode="min",
)

# Setup WandB logging
wandb_logger = WandbLogger(
    project="ssl-remote-sensing-simclr", config=train_config.__dict__
)
shared_trainer_kwargs = {
    "callbacks": [accumulator, checkpoint_callback],
    "max_epochs": train_config.epochs,
    "logger": wandb_logger,
    "log_every_n_steps": 1,
    "accelerator": "gpu",
}
if resume_from_checkpoint:
    trainer = Trainer(
        **shared_trainer_kwargs,
        resume_from_checkpoint=train_config.checkpoint_path,
    )
else:
    trainer = Trainer(
        **shared_trainer_kwargs,
    )

trainer.fit(model, bigearthnet_dataloader)
trainer.save_checkpoint(save_name)
wandb.save(checkpoint_callback.best_model_path)
wandb.finish()
print(f"Best model is stored under {checkpoint_callback.best_model_path}")
