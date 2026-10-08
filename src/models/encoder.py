"""The encoder: a ResNet-18 trunk at stride 8 for one-channel images, initialized from Mirai."""
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn

from src.config import EncoderConfig


class ResNetEncoder(nn.Module):
    """Maps images [N, 1, H, W] in [0, 1] to features [N, 256, H / 8, W / 8], not L2-normalized.

    The trunk is ResNet-18 up to its third stage, with that stage's stride removed. With dilate=True the stage's later
    3x3 filters are dilated, so at every second cell the trunk computes exactly what the pretrained stride-16 network
    computed, and pretrained filters and stored BatchNorm statistics stay valid. The receptive field is then 211 px
    (163 px without dilation).

    The trunk ends in a ReLU, so its features are non-negative and any two cells have a high cosine similarity,
    whatever they show. output_norm, a BatchNorm with no ReLU after it, centres them. Its statistics cannot come from
    the checkpoint: calibrate() sets them once from data before training.

    state_dict         pretrained weights in torchvision's key format, or None for random weights
    mean, std          the intensity normalization the pretrained weights expect, on the [0, 1] scale
    freeze_batch_norm  keep every BatchNorm in evaluation mode, so a cell's feature depends only on the pixels in its
                       receptive field; their scale and shift stay trainable. False only for random weights.
    """

    stride = 8
    out_channels = 256

    def __init__(self, state_dict: dict[str, Tensor] | None = None, mean: float = 0.5, std: float = 0.25,
                 freeze_batch_norm: bool = True, dilate: bool = True):
        super().__init__()
        network = torchvision.models.resnet18()
        if state_dict is not None:
            # Every trunk key must be there (a missing one raises a KeyError); layer4 and fc are not used.
            trunk_keys = [key for key in network.state_dict() if not key.startswith(('layer4.', 'fc.'))]
            network.load_state_dict({key: state_dict[key] for key in trunk_keys}, strict=False)

        # Summing the first filters over the three input channels gives exactly the original output on a grey image.
        conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        conv1.weight.data = network.conv1.weight.data.sum(dim=1, keepdim=True)

        first, second = network.layer3
        first.conv1.stride = first.downsample[0].stride = (1, 1)
        if dilate:
            for conv in (first.conv2, second.conv1, second.conv2):
                conv.dilation = conv.padding = (2, 2)

        self.trunk = nn.Sequential(conv1, network.bn1, network.relu, network.maxpool,
                                   network.layer1, network.layer2, network.layer3)
        self.output_norm = nn.BatchNorm2d(self.out_channels)
        self.freeze_batch_norm = freeze_batch_norm
        self.register_buffer('mean', torch.tensor(float(mean)))
        self.register_buffer('std', torch.tensor(float(std)))
        self.register_buffer('calibrated', torch.tensor(False))
        self.train()

    def train(self, mode: bool = True):
        # Lightning calls .train() on the whole model at the start of every epoch,
        # which would silently put the BatchNorm layers back into training mode.
        super().train(mode)
        if self.freeze_batch_norm:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

        return self

    def forward(self, images: Tensor) -> Tensor:
        return self.output_norm(self.trunk((images - self.mean) / self.std))

    @torch.no_grad()
    def calibrate(self, batches: Iterable[Tensor], tissue_threshold: float = 0.01):
        """Set output_norm's statistics from image batches [N, 1, H, W], over tissue cells only.

        Background cells are many and identical, and would dominate the mean. Run it once on a few hundred training
        frames and save the encoder: under multi-GPU training every process must load the same statistics.
        """
        was_training = self.training
        self.eval()
        count = total = square_total = 0
        for images in batches:
            features = self.trunk((images - self.mean) / self.std).double()
            weight = (F.avg_pool2d((images > tissue_threshold).double(), self.stride) > 0.5).double()
            count = count + weight.sum()
            total = total + (features * weight).sum(dim=(0, 2, 3))
            square_total = square_total + (features.square() * weight).sum(dim=(0, 2, 3))
        if not count > 0:
            raise ValueError('calibrate() found no tissue cells in the given batches')

        mean = total / count
        self.output_norm.running_mean.copy_(mean)
        self.output_norm.running_var.copy_((square_total / count - mean.square()).clamp_min(1e-6))
        self.calibrated.fill_(True)
        self.train(was_training)


def load_mirai_trunk(path: Path) -> dict[str, Tensor]:
    """The trunk weights of a mirai-shira checkpoint, in torchvision's key format."""
    state_dict = torch.load(path, map_location='cpu', weights_only=True)['model']

    return {key.removeprefix('trunk.'): value for key, value in state_dict.items() if key.startswith('trunk.')}


def build_encoder(config: EncoderConfig) -> ResNetEncoder:
    """The encoder with Mirai weights (random ones without a checkpoint), then the calibrated state if there is one."""
    state_dict = None if config.checkpoint is None else load_mirai_trunk(config.checkpoint)
    encoder = ResNetEncoder(state_dict, config.mean, config.std, config.freeze_batch_norm, config.dilate)
    if config.calibrated is not None:
        encoder.load_state_dict(torch.load(config.calibrated, map_location='cpu', weights_only=True))

    return encoder
