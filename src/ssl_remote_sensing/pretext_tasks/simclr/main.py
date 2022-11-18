import os

import torch
import torchvision.models as models
from torchvision.datasets import EuroSAT
from torch.utils.data import DataLoader
from torch.multiprocessing import cpu_count
from pytorch_lightning.callbacks import GradientAccumulationScheduler, ModelCheckpoint
from pytorch_lightning import Trainer

from ssl_remote_sensing.data.eurosat_dataset import means, stds
from ssl_remote_sensing.pretext_tasks.simclr.utils import reproducibility
from ssl_remote_sensing.pretext_tasks.simclr.training import SimCLRTraining
from ssl_remote_sensing.pretext_tasks.simclr.augmentation import Augment

# Machine setup
available_gpus = torch.cuda.device_count()
save_model_path = os.path.join(os.getcwd(), "saved_models/")
print("available_gpus:", available_gpus)

# Run setup
filename = "SimCLR_ResNet18_adam"
save_name = filename + ".ckpt"
resume_from_checkpoint = False


# Model Setup
class Hparams:
    def __init__(self):
        self.epochs = 1  # number of training epochs
        self.seed = 1234  # randomness seed
        self.cuda = False  # use nvidia gpu
        self.img_size = 64  # image shape
        self.save = "./saved_models/"  # save checkpoint
        self.gradient_accumulation_steps = 1  # gradient accumulation steps
        self.batch_size = 64
        self.lr = 1e-3
        self.embedding_size = 128  # papers value is 128
        self.temperature = 0.5  # 0.1 or 0.5
        self.weight_decay = 1e-6
        self.checkpoint_path = None


train_config = Hparams()
reproducibility(train_config)

model = SimCLRTraining(
    config=train_config,
    model=models.resnet18(weights=None),
    feat_dim=512,
    norm_means=means,
    norm_stds=stds,
)

transform = Augment(train_config.img_size, )

dataset = EuroSAT(
    "/Users/alexanderlontke/datasets/", transform=transform, download=True
)
data_loader = DataLoader(
    dataset=torch.utils.data.Subset(dataset, range(200)),
    batch_size=train_config.batch_size,
    num_workers=cpu_count(),
)

# Needed to get simulate a large batch size
accumulator = GradientAccumulationScheduler(scheduling={0: 1})

checkpoint_callback = ModelCheckpoint(
    filename=filename,
    dirpath=save_model_path,
    every_n_epochs=2,
    save_last=True,
    save_top_k=2,
    monitor="Contrastive loss_epoch",
    mode="min",
)

if resume_from_checkpoint:
    trainer = Trainer(
        callbacks=[accumulator, checkpoint_callback],
        gpus=available_gpus,
        max_epochs=train_config.epochs,
        resume_from_checkpoint=train_config.checkpoint_path,
    )
else:
    trainer = Trainer(
        callbacks=[accumulator, checkpoint_callback],
        gpus=available_gpus,
        max_epochs=train_config.epochs,
    )

trainer.fit(model, data_loader)
trainer.save_checkpoint(save_name)
