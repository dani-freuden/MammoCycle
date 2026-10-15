from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

import lightning.pytorch as pl
from torchvision.models import resnet18, ResNet18_Weights

# b5_prior_fusion imports geometry as a top-level module
sys.path.insert(0, str(Path(__file__).parents[1] / "mixturemodel" / "lesion_matching"))
from b5_prior_fusion import AnatomicalPrior, ScoreMap, choose_tau, print_table, sweep_tau


# B5 temperatures swept on dev. The logits are cos / cosine_temperature,
# so tau = 1 fuses softmax(logits) with the prior as it is.
FUSION_TAUS = np.logspace(-1, 1, 9)


class LocalAppearanceMatcher(pl.LightningModule):
    """
    Minimal learned local matcher.

    Expected batch:
        {
            "current": Tensor[B, 1, Hc, Wc],
            "prior": Tensor[B, 1, Hp, Wp],

            # Ground-truth corresponding center INSIDE prior crop,
            # in prior-crop pixel coordinates: (x, y)
            "target_xy": Tensor[B, 2],

            # optional
            "prior_mask": Tensor[B, 1, Hp, Wp],
            "pixel_spacing": Tensor[B, 2],  # (sy_mm, sx_mm)

            # validation, for the fusion with anatomical_prior
            "M_prior": Tensor[B, 2, 3],  # prior-crop px -> prior image px
            "p_anat": Tensor[B, 2],      # (x, y), prior image px
            "centroid": Tensor[B, 2],
            "p_gt": Tensor[B, 2],
        }

    Idea:
        current crop -> one feature vector q
        prior crop   -> spatial feature map F

        score[y, x] = cosine(q, F[:, y, x]) / cosine_temperature

    The score map is upsampled back to prior-crop resolution,
    so target_xy can stay in ordinary crop pixel coordinates.

    With an anatomical_prior, every validation epoch fuses the dev score
    maps with it (b5_prior_fusion.sweep_tau over FUSION_TAUS) and logs
    val/fused_tmean30 at the tau chosen on dev, and that tau as val/fused_tau.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        target_sigma_px: float = 8.0,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        cosine_temperature: float = 0.07,
        spacing_mm: float = 0.12,
        anatomical_prior: AnatomicalPrior | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["anatomical_prior"])
        self.anatomical_prior = anatomical_prior
        self.val_cases = []

        # --------------------------------------------------
        # Shared ResNet-18 encoder, stopped at layer2.
        #
        # Output:
        #   [B, 128, H/8, W/8]
        # --------------------------------------------------

        weights = (
            ResNet18_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )

        resnet = resnet18(weights=weights)

        # Convert RGB conv1 -> grayscale conv1.
        old_conv = resnet.conv1

        new_conv = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        if pretrained:
            # Equivalent to feeding the same grayscale image
            # into all three RGB channels.
            with torch.no_grad():
                new_conv.weight.copy_(
                    old_conv.weight.sum(dim=1, keepdim=True)
                )

        self.encoder = nn.Sequential(
            new_conv,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        )

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

    # ======================================================
    # Encoder / matcher
    # ======================================================

    def encode(self, x: Tensor) -> Tensor:
        """
        x: [B, 1, H, W]

        returns:
            [B, 128, H/8, W/8] approximately
        """
        return self.encoder(x)

    def forward(
        self,
        current: Tensor,
        prior: Tensor,
        prior_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Returns appearance logits at PRIOR IMAGE resolution.

        output:
            [B, Hp, Wp]
        """

        # Shared feature space
        current_features = self.encode(current)
        prior_features = self.encode(prior)

        # --------------------------------------------------
        # Current = query.
        #
        # For version 1, simply summarize the entire centered
        # current crop into one descriptor.
        # --------------------------------------------------

        query = F.adaptive_avg_pool2d(
            current_features,
            output_size=1,
        ).flatten(1)  # [B, C]

        # L2 normalization -> dot product becomes cosine similarity
        query = F.normalize(query, dim=1)
        prior_features = F.normalize(prior_features, dim=1)

        # --------------------------------------------------
        # Cosine similarity with every prior spatial location
        #
        # query: [B, C]
        # prior: [B, C, Hf, Wf]
        #
        # -> [B, Hf, Wf]
        # --------------------------------------------------

        logits = torch.einsum(
            "bc,bchw->bhw",
            query,
            prior_features,
        ) / self.hparams.cosine_temperature

        # --------------------------------------------------
        # Upsample back into PRIOR CROP pixel coordinates.
        # This keeps all GT bookkeeping simple.
        # --------------------------------------------------

        logits = F.interpolate(
            logits[:, None],
            size=prior.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0]

        # Optional breast mask
        if prior_mask is not None:
            mask = prior_mask[:, 0].bool()
            logits = logits.masked_fill(~mask, -1e4)

        return logits

    # ======================================================
    # Target
    # ======================================================

    def make_target(
        self,
        target_xy: Tensor,
        height: int,
        width: int,
    ) -> Tensor:
        """
        Gaussian spatial target centered on GT (x, y).

        returns:
            [B, H, W], sums to 1 per sample.
        """

        device = target_xy.device
        dtype = target_xy.dtype

        ys = torch.arange(
            height,
            device=device,
            dtype=dtype,
        )
        xs = torch.arange(
            width,
            device=device,
            dtype=dtype,
        )

        yy, xx = torch.meshgrid(
            ys,
            xs,
            indexing="ij",
        )

        xx = xx[None]
        yy = yy[None]

        x = target_xy[:, 0, None, None]
        y = target_xy[:, 1, None, None]

        dist2 = (xx - x).square() + (yy - y).square()

        sigma = self.hparams.target_sigma_px

        target = torch.exp(
            -0.5 * dist2 / (sigma ** 2)
        )

        target = target / (
            target.sum(dim=(1, 2), keepdim=True) + 1e-8
        )

        return target

    # ======================================================
    # Loss
    # ======================================================

    def spatial_ce(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Cross entropy between two spatial distributions.
        """

        B = logits.shape[0]

        log_probs = F.log_softmax(
            logits.flatten(1),
            dim=1,
        ).view_as(logits)

        return -(
            target * log_probs
        ).sum(dim=(1, 2)).mean()

    # ======================================================
    # Decode
    # ======================================================

    @staticmethod
    def argmax_xy(logits: Tensor) -> Tensor:
        """
        logits: [B, H, W]

        returns:
            [B, 2] as (x, y)
        """

        B, H, W = logits.shape

        idx = logits.flatten(1).argmax(dim=1)

        y = idx // W
        x = idx % W

        return torch.stack([x, y], dim=1).float()

    # ======================================================
    # Lightning steps
    # ======================================================

    def _step(
        self,
        batch: dict,
        stage: str,
    ) -> tuple[Tensor, Tensor]:
        """Logs the stage's metrics; returns the loss and the logits."""

        current = batch["current"]
        prior = batch["prior"]
        target_xy = batch["target_xy"].float()

        prior_mask = batch.get("prior_mask")

        logits = self(
            current,
            prior,
            prior_mask=prior_mask,
        )

        target = self.make_target(
            target_xy,
            height=logits.shape[-2],
            width=logits.shape[-1],
        )

        loss = self.spatial_ce(
            logits,
            target,
        )

        # ----------------------------------------------
        # Localization error
        # ----------------------------------------------

        pred_xy = self.argmax_xy(logits)

        delta_px = pred_xy - target_xy

        error_px = delta_px.norm(dim=1)

        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
            batch_size=current.shape[0],
        )

        self.log(
            f"{stage}/error_px",
            error_px.mean(),
            prog_bar=(stage != "train"),
            on_step=False,
            on_epoch=True,
            batch_size=current.shape[0],
        )

        # Optional physical error
        if "pixel_spacing" in batch:
            # pixel_spacing = (sy, sx)
            spacing = batch["pixel_spacing"]

            dx_mm = delta_px[:, 0] * spacing[:, 1]
            dy_mm = delta_px[:, 1] * spacing[:, 0]

            error_mm = torch.sqrt(
                dx_mm.square() + dy_mm.square()
            )

            self.log(
                f"{stage}/error_mm",
                error_mm.mean(),
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=current.shape[0],
            )

        return loss, logits

    def training_step(
        self,
        batch: dict,
        batch_idx: int,
    ):
        return self._step(batch, "train")[0]

    def validation_step(
        self,
        batch: dict,
        batch_idx: int,
    ):
        loss, logits = self._step(batch, "val")

        # Keep the score maps for the fusion at the end of the epoch
        if self.anatomical_prior is not None:
            logits = logits.float().cpu().numpy()
            for k in range(len(logits)):
                self.val_cases.append({
                    "smap": ScoreMap(score=logits[k], M_grid=batch["M_prior"][k].cpu().numpy()),
                    **{key: batch[key][k].cpu().numpy() for key in ("p_anat", "centroid", "p_gt")},
                })

        return loss

    def on_validation_epoch_end(self):
        if not self.val_cases:
            return

        rows = sweep_tau(
            self.val_cases,
            self.anatomical_prior,
            FUSION_TAUS,
            self.hparams.spacing_mm,
        )
        self.val_cases = []

        tau = choose_tau(rows)
        print(f"\nepoch {self.current_epoch}, dev, mm at a nominal {self.hparams.spacing_mm} mm/px")
        print_table(rows)

        self.log("val/fused_tmean30", float(dict(rows)[f"fused tau={tau:.4g}"]["tmean30"]), prog_bar=True)
        self.log("val/fused_tau", tau)

    def test_step(
        self,
        batch: dict,
        batch_idx: int,
    ):
        return self._step(batch, "test")[0]

    # ======================================================
    # Inference
    # ======================================================

    @torch.no_grad()
    def predict_match(
        self,
        current: Tensor,
        prior: Tensor,
        prior_mask: Tensor | None = None,
    ):
        logits = self(
            current,
            prior,
            prior_mask=prior_mask,
        )

        pred_xy = self.argmax_xy(logits)

        probs = F.softmax(
            logits.flatten(1),
            dim=1,
        ).view_as(logits)

        return {
            "logits": logits,
            "probs": probs,
            "pred_xy": pred_xy,
        }

    # ======================================================
    # B5-style anatomy fusion
    # ======================================================

    @staticmethod
    def fuse_with_anatomical_prior(
        appearance_logits: Tensor,
        anatomical_log_prior: Tensor,
        temperature: float = 1.0,
    ) -> Tensor:
        """
        Both tensors: [B, H, W]

        Returns unnormalized posterior log-score.

        This is deliberately kept OUTSIDE the learned matcher.
        """

        B = appearance_logits.shape[0]

        appearance_log_prob = F.log_softmax(
            (
                appearance_logits / temperature
            ).flatten(1),
            dim=1,
        ).view_as(appearance_logits)

        return (
            appearance_log_prob
            + anatomical_log_prior
        )

    # ======================================================
    # Optimizer
    # ======================================================

    def configure_optimizers(self):
        return torch.optim.AdamW(
            filter(
                lambda p: p.requires_grad,
                self.parameters(),
            ),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )