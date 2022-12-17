import torch
from torch import nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    """
    Model used for ML-Challenge
    """

    def __init__(self):
        """
        Model definition
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, 5)  # in: 3x64x64 out: 60x60x24
        self.pool = nn.MaxPool2d(2, 2)  # out: 30x30x24
        self.conv2 = nn.Conv2d(24, 72, 5)  # out: 26x26x72

    def forward(self, x):
        """
        Model forward pass
        :param x: List of image samples
        :return:
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        return x


class FullyConnectedBlock(nn.Module):
    """
    Model used for ML-Challenge
    """

    def __init__(self, input_dim: int, output_dim):
        """
        Model definition
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)  # 72 * 13 * 13
        self.fc2 = nn.Linear(512, 124)
        self.fc3 = nn.Linear(124, output_dim)

    def forward(self, x):
        """
        Model forward pass
        :param x: List of image samples
        :return:
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LinearBlock(nn.Module):
    """
    Model used for ML-Challenge
    """

    def __init__(self, input_dim: int, output_dim):
        """
        Model definition
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = nn.BatchNorm1d(num_features=output_dim)

    def forward(self, x):
        """
        Model forward pass
        :param x: List of image samples
        :return:
        """
        return self.norm(self.fc1(x))


class DownstreamClassificationNet(nn.Module):
    """
    Model used for ML-Challenge
    """

    def __init__(
        self,
        input_dim: int,
        encoder: nn.Module = None,
        output_dim: int = 10,
        gan_encoder: bool = False,
        freeze_encoder: bool = True,
        use_linear_head: bool = True,
    ):
        """
        Model definition
        """
        super().__init__()
        self.gan_encoder = gan_encoder
        self.encoder = encoder if encoder else EncoderBlock()
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if use_linear_head:
            self.head = LinearBlock(input_dim=input_dim, output_dim=output_dim)
        else:
            self.head = FullyConnectedBlock(input_dim=input_dim, output_dim=output_dim)

    def forward(self, x):
        """
        Model forward pass
        :param x: List of image samples
        :return:
        """
        x = self.encoder(x)
        if self.gan_encoder:
            x = torch.flatten(x, 1)
        x = self.head(x)
        return x
