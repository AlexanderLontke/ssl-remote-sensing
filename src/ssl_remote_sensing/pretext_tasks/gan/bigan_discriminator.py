import torch.nn as nn
import torch
from torch import Tensor

class BiganDiscriminator(nn.Module):
  def __init__(self, latent_dim: int, feature_maps: int, image_channels: int) -> None:
    super().__init__()

    self.disc_x = nn.Sequential(
        self._make_disc_block(image_channels, feature_maps, batch_norm=False),
        self._make_disc_block(feature_maps, feature_maps * 2),
        self._make_disc_block(feature_maps * 2, feature_maps * 4),
        self._make_disc_block(feature_maps * 4, feature_maps * 8),
        self._make_disc_block(feature_maps * 8, feature_maps * 8, kernel_size = 4, stride = 1, padding = 0, last_block = True),
    )

    self.disc_z = nn.Sequential(
        nn.Conv2d(latent_dim, feature_maps * 8, kernel_size = 1, stride = 1, bias=False),
        nn.LeakyReLU(negative_slope=0.1, inplace=True), 
        nn.Dropout2d(0.2),
        nn.Conv2d(feature_maps * 8, feature_maps * 8, kernel_size = 1, stride = 1, bias=False),
        nn.LeakyReLU(negative_slope=0.1, inplace=True), 
        nn.Dropout2d(0.2),
    )

    self.disc_joint = nn.Sequential(
        nn.Conv2d(feature_maps * 16, feature_maps * 16, kernel_size = 1, stride = 1, bias=False),
        nn.LeakyReLU(negative_slope=0.1, inplace=True), 
        nn.Dropout(0.2),
        nn.Conv2d(feature_maps * 16, feature_maps * 16, kernel_size = 1, stride = 1, bias=False),
        nn.LeakyReLU(negative_slope=0.1, inplace=True), 
        nn.Dropout2d(0.2),
        nn.Conv2d(feature_maps * 16, 1, kernel_size = 1, stride = 1, bias=False)
    )

  @staticmethod
  def _make_disc_block(
      in_channels: int,
      out_channels: int, 
      kernel_size: int = 4,
      stride: int = 2,
      padding: int = 1,
      bias: bool = False,
      batch_norm: bool = True,
      last_block: bool = False,
  ) -> nn.Sequential:
    if not last_block:
      disc_block = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
          nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
          nn.LeakyReLU(0.2, inplace=True),
      )
    else:
      disc_block = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
      )
    
    return disc_block
    
  def forward(self, x: Tensor, z: Tensor) -> Tensor:
    x = self.disc_x(x)
    z = self.disc_z(z)
    joint = torch.cat((x,z), dim=1)
    out = self.disc_joint(joint)
    return torch.sigmoid(out)