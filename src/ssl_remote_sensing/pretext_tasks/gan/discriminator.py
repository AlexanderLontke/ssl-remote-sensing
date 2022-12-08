import torch.nn as nn

# implement the Discriminator network architecture
class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.ngpu = config.num_gpus
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(config.num_channels, config.d_featuremaps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (config.d_featuremaps) x 32 x 32
            nn.Conv2d(
                config.d_featuremaps, config.d_featuremaps * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(config.d_featuremaps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (config.d_featuremaps*2) x 16 x 16
            nn.Conv2d(
                config.d_featuremaps * 2, config.d_featuremaps * 4, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(config.d_featuremaps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (config.d_featuremaps*4) x 8 x 8
            nn.Conv2d(
                config.d_featuremaps * 4, config.d_featuremaps * 8, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(config.d_featuremaps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (config.d_featuremaps*8) x 4 x 4
            nn.Conv2d(config.d_featuremaps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)

    def print_architecture(self):
        print("[LOG] Discriminator architecture:\n\n{}\n".format(self.main))
