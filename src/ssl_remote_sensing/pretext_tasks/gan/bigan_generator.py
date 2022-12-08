import torch.nn as nn
from torch import Tensor

class BiganResnetGenerator(nn.Module):
    def __init__(self, latent_dim: int, feature_maps: int, image_channels: int, pretrained_model: nn.Module) -> None:
        super(BiganResnetGenerator, self).__init__()
        # input dim: 100 x 1 x 1
        self.noise_init = nn.Sequential(
            self._make_gen_block(latent_dim, feature_maps * 8, kernel_size = 4, stride = 1, padding = 0),
            self._make_gen_block(feature_maps * 8, feature_maps * 4),
            self._make_gen_block(feature_maps * 4, feature_maps * 2),
            self._make_gen_block(feature_maps * 2, feature_maps),
            self._make_gen_block(feature_maps, image_channels, last_block=True),
        )
        # input dim: 3 x 64 x 64
        self.pretrained_model = pretrained_model
        # input dim: 512 x 1 x 1
        self.gen = nn.Sequential(
            self._make_gen_block(feature_maps * 8, feature_maps * 8, kernel_size = 4, stride = 1, padding = 0),
            self._make_gen_block(feature_maps * 8, feature_maps * 4),
            self._make_gen_block(feature_maps * 4, feature_maps * 2),
            self._make_gen_block(feature_maps * 2, feature_maps),
            self._make_gen_block(feature_maps, image_channels, last_block=True),
        )
        # output dim: 3 x 64 x 64

        self.noise_init.apply(self._weights_init)
        self.gen.apply(self._weights_init)
    

    @staticmethod
    def _make_gen_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        bias: bool = False,
        last_block: bool = False,
    ) -> nn.Sequential:
        if not last_block:
            gen_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            gen_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.Tanh(),
            )
        return gen_block
    
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
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            enc_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
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
        x = self.noise_init(x)
        x = self.pretrained_model(x)
        x = self.gen(x)
        return x