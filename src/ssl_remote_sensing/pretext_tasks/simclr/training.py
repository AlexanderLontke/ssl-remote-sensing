import torch
import wandb
import pytorch_lightning as pl
# try:
#     from pytorch_lightning.loggers import Logger
# except ImportError:
#     from pytorch_lightning.loggers import LightningLoggerBase
#     Logger = LightningLoggerBase
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR # TODO: the feature is marked with issue in pytorch-lightning-bolts
from torch.optim import SGD, Adam

from pretext_tasks.simclr.resnet_18_backbone import AddProjection
from pretext_tasks.simclr.utils import define_parameter_groups
from pretext_tasks.simclr.loss import ContrastiveLoss


class SimCLRTraining(pl.LightningModule):
    def __init__(self, config, feat_dim=512):
        super().__init__()
        self.config = config
        self.model = AddProjection(config, mlp_dim=feat_dim)
        self.loss = ContrastiveLoss(
            batch_size=config.batch_size, temperature=self.config.temperature
        )

    def forward(self, batch, *args, **kwargs) -> torch.Tensor:
        return self.model(batch)

    def training_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        (x1, x2) = batch
        z1 = self.model(x1)
        z2 = self.model(x2)
        loss = self.loss(z1, z2)

        self.log(
            "train/NTXentLoss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        wd = self.config.weight_decay
        lr = self.config.lr
        max_epochs = int(self.config.epochs)
        param_groups = define_parameter_groups(
            self.model, weight_decay=wd, optimizer_name="adam"
        )
        optimizer = Adam(param_groups, lr=lr, weight_decay=wd)

        print(
            f"Optimizer Adam, "
            f"Learning Rate {lr}, "
            f"Effective batch size {self.config.batch_size * self.config.gradient_accumulation_steps}"
        )

        # # TODO: the feature is marked with issue in pytorch-lightning-bolts
        # scheduler_warmup = LinearWarmupCosineAnnealingLR(
        #     optimizer, warmup_epochs=10, max_epochs=max_epochs, warmup_start_lr=0.0
        # )


        # return [optimizer], [scheduler_warmup]
        return optimizer
