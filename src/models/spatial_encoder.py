import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import resnet18



class SpatialEncoder(nn.Module):
    """
    Spatial feature encoder phi from TimeCycle.

    Input:
        [B, 1, H, W]

    Output:
        [B, C, H/8, W/8]

    With the initial configuration:
        240x240 -> [B, 256, 30, 30]
         80x80  -> [B, 256, 10, 10]
    """

    def __init__(self) -> None:
        super().__init__()

        backbone = resnet18(weights=None) #TODO: make optional

        # Mammograms are grayscale.
        backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # We keep:
        #
        # conv1:   /2
        # maxpool: /2
        # layer1:  /1
        # layer2:  /2
        #
        # Total: /8
        #
        # layer3 is included for representation capacity, but we prevent
        # its usual stride-2 spatial downsampling.

        first_block = backbone.layer3[0]

        first_block.conv1.stride = (1, 1)
        first_block.downsample[0].stride = (1, 1)

        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )

        self.out_channels = 256

    def forward(self, x: Tensor) -> Tensor:
        features = self.encoder(x)

        # TimeCycle compares features using dot products.
        # L2 normalization makes these effectively cosine similarities.
        features = F.normalize(
            features,
            p=2,
            dim=1,
        )

        return features
