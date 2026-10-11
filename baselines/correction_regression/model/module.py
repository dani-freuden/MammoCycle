"""Correction-regression baseline: regress the anatomical-coordinates baseline's error.

The dataset crops both images, a 448 px window resized to 224x224: the current one around the current box centre, the
prior one around the anatomical centre, the centre of the anatomical baseline's predicted box (pred_* in its
predictions.csv). The target is (dx, dy) = anatomical centre - true prior box centre, as a fraction of the window, so the
corrected centre is the anatomical centre - window * output.

A batch is a dict of 'current' and 'prior', (B, 1, 224, 224) crops in [0, 1], and 'target', (B, 2) as (dx, dy).
Validation also logs the distance from the corrected centre to the true one in PNG pixels, WINDOW * |target - output|,
beside the anatomical baseline's on the same pairs, WINDOW * |target|.
"""
from pathlib import Path

import lightning as L
import torch
import torchvision
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.encoder import load_mirai_trunk

from ..data.prepare import WINDOW


class CorrectionRegressionModel(L.LightningModule):
    def __init__(self, checkpoint: Path | None = None, mean: float = 0.11866, std: float = 0.18850,
                 huber_delta: float = 0.05, learning_rate: float = 1e-4, weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        # ResNet-18 with Mirai weights (random ones without a checkpoint), shared by both crops.
        network = torchvision.models.resnet18()
        network.fc = nn.Identity()
        if checkpoint is not None:
            network.load_state_dict(load_mirai_trunk(checkpoint))
        # Summing the first filters over the three input channels gives exactly the original output on a grey image.
        conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        conv1.weight.data = network.conv1.weight.data.sum(dim=1, keepdim=True)
        self.backbone = nn.Sequential(conv1, network.bn1, network.relu, network.maxpool,
                                      network.layer1, network.layer2, network.layer3, network.layer4)

        # The prior keeps its 7x7 layout, which is where the offset is read from.
        self.prior_reduce = nn.Conv2d(512, 64, kernel_size=1)
        self.head = nn.Sequential(nn.Linear(512 + 64 * 7 * 7, 256), nn.ReLU(),
                                  nn.Linear(256, 64), nn.ReLU(),
                                  nn.Linear(64, 2))
        self._val_errors: list[Tensor] = []  # (B, 2) per batch: the corrected and the anatomical centre's error

    def forward(self, current: Tensor, prior: Tensor) -> Tensor:
        """The predicted (dx, dy), (B, 2), as a fraction of the window."""
        current_feat = self.backbone(self._normalize(current)).mean(dim=(2, 3))  # (B, 512)
        prior_feat = self.prior_reduce(self.backbone(self._normalize(prior))).flatten(1)  # (B, 64 * 7 * 7)

        return self.head(torch.cat([current_feat, prior_feat], dim=1))

    def training_step(self, batch, batch_idx):
        loss = self._loss(self(batch['current'], batch['prior']), batch['target'])
        self.log('train/loss', loss, batch_size=len(batch['target']), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        predicted, target = self(batch['current'], batch['prior']), batch['target']
        self.log('val/loss', self._loss(predicted, target), batch_size=len(target), prog_bar=True)
        self._val_errors.append(WINDOW * torch.stack([(target - predicted).norm(dim=1), target.norm(dim=1)], dim=1))

    def on_validation_epoch_end(self):
        errors = torch.cat(self._val_errors).float()
        self._val_errors = []
        self.log_dict({'val/error_px/mean': errors[:, 0].mean(), 'val/error_px/median': errors[:, 0].median(),
                       'val/anatomical_error_px/mean': errors[:, 1].mean(),
                       'val/anatomical_error_px/median': errors[:, 1].median()})

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate,
                                 weight_decay=self.hparams.weight_decay)

    def _loss(self, predicted: Tensor, target: Tensor) -> Tensor:
        """Huber(dx_gt, predicted_dx) + Huber(dy_gt, predicted_dy)."""
        delta = self.hparams.huber_delta

        return (F.huber_loss(predicted[:, 0], target[:, 0], delta=delta)
                + F.huber_loss(predicted[:, 1], target[:, 1], delta=delta))

    def _normalize(self, images: Tensor) -> Tensor:
        return (images - self.hparams.mean) / self.hparams.std
