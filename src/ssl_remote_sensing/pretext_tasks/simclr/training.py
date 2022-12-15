import torch
import wandb
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim import SGD, Adam

from ssl_remote_sensing.pretext_tasks.simclr.resnet_18_backbone import AddProjection
from ssl_remote_sensing.pretext_tasks.simclr.utils import define_parameter_groups
from ssl_remote_sensing.pretext_tasks.simclr.loss import nt_xent_loss


class SimCLRTraining(pl.LightningModule):
    def __init__(self, config, feat_dim=512):
        super().__init__()
        self.config = config
        self.model = AddProjection(config, mlp_dim=feat_dim)

    def forward(self, batch, *args, **kwargs) -> torch.Tensor:
        return self.model(batch)

    def training_step(self, batch, batch_idx, *args, **kwargs) -> torch.Tensor:
        (x1, x2) = batch
        z1 = self.model(x1)
        z2 = self.model(x2)
        loss = nt_xent_loss(z1, z2, self.config.temperature)

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

        scheduler_warmup = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=10, max_epochs=max_epochs, warmup_start_lr=0.0
        )
        return [optimizer], [scheduler_warmup]
