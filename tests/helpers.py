"""Stand-ins shared by the tests: random features, a 'pretrained' network and textured frames."""
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from src.models.encoder import ResNetEncoder


STRIDE = 8


def random_features(batch=2, channels=32, height=32, width=16, seed=0):
    """A non-square feature map (2:1 like the frames) with unrelated unit vectors in every cell."""
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(batch, channels, height, width, generator=generator), dim=1)


def pretrained_like():
    """A ResNet-18 with random weights and random BatchNorm statistics, standing in for a pretrained one."""
    torch.manual_seed(0)
    network = torchvision.models.resnet18()
    for module in network.modules():
        if isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.running_mean, std=0.1)
            nn.init.uniform_(module.running_var, 0.5, 1.5)
            nn.init.uniform_(module.weight, 0.5, 1.5)
    return network.eval()


def textured_frames(batch, height=320, width=192, seed=0):
    """Smooth random texture on the left 80% of the frame ('tissue'), zero elsewhere ('background')."""
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.rand(batch, 1, height // 8, width // 8, generator=generator)
    fine = torch.rand(batch, 1, height // 2, width // 2, generator=generator)
    upsample = lambda x: F.interpolate(x, size=(height, width), mode='bilinear', align_corners=False)
    frames = (0.15 + 0.5 * upsample(coarse) + 0.3 * upsample(fine)).clamp(0, 1)
    frames[..., int(0.8 * width):] = 0
    return frames


def calibrated_encoder(height=320, width=192):
    torch.manual_seed(0)
    encoder = ResNetEncoder(pretrained_like().state_dict())
    encoder.calibrate([textured_frames(4, height, width, seed=100)])  # without it the affinity is flat
    return encoder
