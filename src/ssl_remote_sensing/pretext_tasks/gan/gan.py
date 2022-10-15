from pytorch_lightning import LightningModule
from ssl_remote_sensing.pretext_tasks.gan.generator import Generator
from ssl_remote_sensing.pretext_tasks.gan.discriminator import Discriminator
import torch
import torch.nn.functional as F
import torchvision
from ssl_remote_sensing.pretext_tasks.gan.utils import weights_init
from collections import OrderedDict

class GAN(LightningModule):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.latent_dim = config.size_latent
        
        # networks
        img_shape = (self.config.num_channels, self.config.image_width, self.config.image_height)
        self.generator = Generator(config=config)
        self.generator.apply(weights_init)
        self.discriminator = Discriminator(config=config)
        self.discriminator.apply(weights_init)

        self.validation_z = torch.randn(64, self.config.size_latent, 1, 1)

        self.example_input_array = torch.zeros(4, self.config.size_latent, 1, 1)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.latent_dim, 1, 1)
        z = z.type_as(imgs)

        # train generator
        if optimizer_idx == 0:

            # generate images
            self.generated_imgs = self(z)

            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image('generated_images', grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs).view(-1)

            # adversarial loss is binary cross-entropy
            self.test = self.discriminator(self(z)).view(-1)
            g_loss = self.adversarial_loss(self.discriminator(self(z)).view(-1), valid)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs).view(-1)

            real_loss = self.adversarial_loss(self.discriminator(imgs).view(-1), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs).view(-1)

            fake_loss = self.adversarial_loss(
                self.discriminator(self(z).detach()).view(-1), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def configure_optimizers(self):
        lr = self.config.lr
        b1 = self.config.b1
        b2 = self.config.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.SGD(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []

    def on_validation_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)