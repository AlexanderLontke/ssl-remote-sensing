from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from pytorch_lightning import LightningModule
from itertools import chain
from torch.autograd import Variable
import torchvision

from ssl_remote_sensing.pretext_tasks.gan.bigan_discriminator import BiganDiscriminator
from ssl_remote_sensing.pretext_tasks.gan.bigan_generator import BiganResnetGenerator
from ssl_remote_sensing.pretext_tasks.gan.bigan_encoder import BiganResnetEncoder
from ssl_remote_sensing.models.ResNet18 import resnet18_basenet


class BIGAN(LightningModule):
    def __init__(
        self,
        beta1: float = 0.5,
        feature_maps_gen: int = 64,
        feature_maps_disc: int = 64,
        feature_maps_enc: int = 64,
        image_channels: int = 3,
        latent_dim: int = 100,
        learning_rate: float = 0.0002,
        batch_size: int = 1000,
        **kwargs: Any,
    ) -> None:

        super().__init__()
        self.beta1 = beta1
        self.feature_maps_gen = feature_maps_gen
        self.feature_maps_disc = feature_maps_disc
        self.feature_maps_enc = feature_maps_enc
        self.image_channels = image_channels
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.pretrained_model = resnet18_basenet(pretrained=False, random_init=True)

        self.generator = self._get_generator()
        self.discriminator = self._get_discriminator()
        self.encoder = self._get_encoder()

        self.criterion = nn.BCELoss()

    def _get_generator(self) -> nn.Module:
        generator = BiganResnetGenerator(
            self.latent_dim,
            self.feature_maps_gen,
            self.image_channels,
            pretrained_model=self.pretrained_model,
        )
        return generator

    def _get_discriminator(self) -> nn.Module:
        discriminator = BiganDiscriminator(
            self.latent_dim, self.feature_maps_disc, self.image_channels
        )
        discriminator.apply(self._weights_init)
        return discriminator

    def _get_encoder(self) -> nn.Module:
        encoder = BiganResnetEncoder(
            self.latent_dim,
            self.feature_maps_enc,
            self.image_channels,
            pretrained_model=self.pretrained_model,
        )
        return encoder

    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)

    def configure_optimizers(self):
        lr = self.learning_rate
        betas = (self.beta1, 0.999)
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        opt_gen_enc = torch.optim.Adam(
            chain(self.generator.parameters(), self.encoder.parameters()),
            lr=lr,
            betas=betas,
        )
        return [opt_disc, opt_gen_enc], []

    def forward(self, noise: Tensor, x: Tensor):

        z_fake = Variable(
            torch.randn((self.batch_size, self.latent_dim, 1, 1)), requires_grad=False
        ).to(self.device)
        x_fake = self.generator(z_fake)

        x_true = x.float()
        z_true = self.encoder(x_true)

        return (
            x_true,
            z_true,
            x_fake,
            z_fake,
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch

        z_fake = Variable(
            torch.randn((self.batch_size, self.latent_dim, 1, 1)), requires_grad=False
        ).to(self.device)
        x_fake = self.generator(z_fake)

        x_true = x.float()
        z_true = self.encoder(x_true)

        # Train discriminator
        result = None
        if optimizer_idx == 0:
            result = self._disc_step(x_true, z_true, x_fake, z_fake)

        # Train generator
        if optimizer_idx == 1:
            result = self._gen_enc_step(x_true, z_true, x_fake, z_fake)

        return result

    def _disc_step(
        self, x_true: Tensor, z_true: Tensor, x_fake: Tensor, z_fake: Tensor
    ) -> Tensor:
        disc_loss = self._get_disc_loss(x_true, z_true, x_fake, z_fake)
        self.log("loss/disc", disc_loss, on_epoch=True)
        return disc_loss

    def _gen_enc_step(
        self, x_true: Tensor, z_true: Tensor, x_fake: Tensor, z_fake: Tensor
    ) -> Tensor:
        gen_enc_loss = self._get_gen_enc_loss(x_true, z_true, x_fake, z_fake)
        self.log("loss/gen_enc", gen_enc_loss, on_epoch=True)
        return gen_enc_loss

    def _get_disc_loss(
        self, x_true: Tensor, z_true: Tensor, x_fake: Tensor, z_fake: Tensor
    ) -> Tensor:
        # Train with real
        out_true = self.discriminator(x_true, z_true)
        y_true = torch.ones_like(out_true)
        real_loss = self.criterion(out_true, y_true)

        # Train with fake
        out_fake = self.discriminator(x_fake, z_fake)
        y_fake = torch.zeros_like(out_fake)
        fake_loss = self.criterion(out_fake, y_fake)

        disc_loss = real_loss + fake_loss

        return disc_loss

    def _get_gen_enc_loss(
        self, x_true: Tensor, z_true: Tensor, x_fake: Tensor, z_fake: Tensor
    ) -> Tensor:
        # Train with real
        out_true = self.discriminator(x_true, z_true)
        y_fake = torch.zeros_like(out_true)
        real_loss = self.criterion(out_true, y_fake)

        # Train with fake
        out_fake = self.discriminator(x_fake, z_fake)
        y_true = torch.ones_like(out_fake)
        fake_loss = self.criterion(out_fake, y_true)

        gen_enc_loss = real_loss + fake_loss

        return gen_enc_loss

    def _get_noise(self, n_samples: int, latent_dim: int) -> Tensor:
        return torch.randn(n_samples, latent_dim, device=self.device)

    def on_validation_epoch_end(self):
        z = Variable(torch.randn((5, self.latent_dim, 1, 1)), requires_grad=False)

        # log sampled images
        sample_imgs = self.generator(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
