import torch.nn as nn
from torch import Tensor


class BiganResnetEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        feature_maps: int,
        image_channels: int,
        pretrained_model: nn.Module,
    ) -> None:
        super(BiganResnetEncoder, self).__init__()
        # input dim: 3 x 64 x 64
        self.pretrained_model = pretrained_model
        self.enc = nn.Sequential(nn.Linear(512, 100))

        self.enc.apply(self._weights_init)

    @staticmethod
    def _make_enc_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        bias: bool = False,
        last_block: bool = False,
    ) -> nn.Sequential:
        if not last_block:
            enc_block = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=bias
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            enc_block = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=bias
                ),
                nn.Tanh(),
            )

        return enc_block

    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pretrained_model(x)
        x = x.view(x.size(0), -1)
        x = self.enc(x)
        x = x.unsqueeze(2).unsqueeze(3)
        return x
