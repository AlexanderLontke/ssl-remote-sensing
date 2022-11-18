import torch.nn as nn

# implement the Generator network architecture
class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.ngpu = config.num_gpus
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(
                config.size_latent, config.g_featuremaps * 8, 4, 1, 0, bias=False
            ),
            nn.BatchNorm2d(config.g_featuremaps * 8),
            nn.ReLU(True),
            # state size. (config.g_featuremaps*8) x 4 x 4
            nn.ConvTranspose2d(
                config.g_featuremaps * 8, config.g_featuremaps * 4, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(config.g_featuremaps * 4),
            nn.ReLU(True),
            # state size. (config.g_featuremaps*4) x 8 x 8
            nn.ConvTranspose2d(
                config.g_featuremaps * 4, config.g_featuremaps * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(config.g_featuremaps * 2),
            nn.ReLU(True),
            # state size. (config.g_featuremaps*2) x 16 x 16
            nn.ConvTranspose2d(
                config.g_featuremaps * 2, config.g_featuremaps, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(config.g_featuremaps),
            nn.ReLU(True),
            # state size. (config.g_featuremaps) x 32 x 32
            nn.ConvTranspose2d(
                config.g_featuremaps, config.num_channels, 4, 2, 1, bias=False
            ),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

    def print_architecture(self):
        print("[LOG] Generator architecture:\n\n{}\n".format(self.main))
