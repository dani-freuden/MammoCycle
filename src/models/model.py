from dataclasses import dataclass, asdict

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import resnet18

from src.models.patch_sampler import PatchSampler

from .spatial_encoder import SpatialEncoder
from .patch_localizer import PatchLocalizer

@dataclass
class MammoCycleConfig:
    image_size: int = 240
    patch_size: int = 80

    learning_rate: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999

    max_cycle_steps: int = 4
    lambda_skip: float = 0.1
    lambda_long: float = 0.1

@dataclass
class TrackerOutput:
    patch_features: Tensor
    params: Tensor
    grid: Tensor
    affinity: Tensor


class MammoCycleModule(L.LightningModule):
    """
    Self-supervised temporal correspondence model for longitudinal
    mammography.

    One video corresponds to one patient + one canonical mammographic view.
    """

    def __init__(
        self,
        config: MammoCycleConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = config or MammoCycleConfig()
        self.save_hyperparameters(asdict(self.config))

        image_feature_size = self.config.image_size // 8
        patch_feature_size = self.config.patch_size // 8
        
        self.encoder = SpatialEncoder()
        self.localizer = PatchLocalizer(
            image_feature_size=image_feature_size,
            patch_feature_size=patch_feature_size,
        )
        self.sampler = PatchSampler(
            image_feature_size=image_feature_size,
            patch_feature_size=patch_feature_size,
        )
    def encode(self, x: Tensor) -> Tensor:
        """
        Encode a batch of mammograms or patches into normalized
        spatial feature maps.

        Args:
            x:
                Tensor shaped [B, 1, H, W].

        Returns:
            Tensor shaped [B, 256, H/8, W/8].
        """
        return self.encoder(x)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward is intentionally simple.

        At inference time, the learned representation is the important
        artifact from TimeCycle.
        """
        return self.encode(x)

    def training_step(
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
    ) -> Tensor:
        raise NotImplementedError(
            "We will implement the TimeCycle losses next."
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate,
            betas=(
                self.config.beta1,
                self.config.beta2,
            ),
        )
        
        
            
    def compute_affinity(
        self,
        image_features: Tensor,
        patch_features: Tensor,
    ) -> Tensor:
        """
        Compute patch-to-image spatial correspondence probabilities.

        Args:
            image_features:
                Normalized spatial features with shape [B, C, H, W].

            patch_features:
                Normalized patch features with shape [B, C, Hp, Wp].

        Returns:
            Affinity tensor with shape:

                [B, H * W, Hp * Wp]

            For every patch location, the probabilities over all image
            locations sum to 1.
        """
        if image_features.ndim != 4:
            raise ValueError(
                f"Expected image_features to have 4 dimensions, "
                f"got {image_features.shape}"
            )

        if patch_features.ndim != 4:
            raise ValueError(
                f"Expected patch_features to have 4 dimensions, "
                f"got {patch_features.shape}"
            )

        if image_features.shape[:2] != patch_features.shape[:2]:
            raise ValueError(
                "Image and patch features must have matching batch and "
                f"channel dimensions. Got {image_features.shape} and "
                f"{patch_features.shape}."
            )
            
            
        image_features = F.normalize(
            image_features,
            p=2,
            dim=1,
        )

        patch_features = F.normalize(
            patch_features,
            p=2,
            dim=1,
        )        

        # [B, C, H, W] -> [B, C, H*W]
        image_flat = image_features.flatten(start_dim=2)

        # [B, C, Hp, Wp] -> [B, C, Hp*Wp]
        patch_flat = patch_features.flatten(start_dim=2)

        # We want:
        #
        #   image location: [B, N, C]
        #   patch location: [B, C, M]
        #
        # resulting in:
        #
        #   [B, N, M]
        #
        # where:
        #   N = H * W
        #   M = Hp * Wp
        similarity = torch.bmm(
            image_flat.transpose(1, 2),
            patch_flat,
        )

        # For each patch location, produce a probability distribution
        # over all possible locations in the full image.
        affinity = F.softmax(similarity, dim=1)

        return affinity
    

    def track(
        self,
        image_features: Tensor,
        query_patch_features: Tensor,
    ) -> TrackerOutput:
        """
        Track a query patch inside a target image feature map.

        This implements the TimeCycle tracker:

            T(x^I, x^p) = h(x^I, g(f(x^I, x^p)))

        Args:
            image_features:
                Target image features of shape [B, C, H, W].

                Default:
                    [B, 256, 30, 30]

            query_patch_features:
                Query patch features of shape [B, C, Hp, Wp].

                Default:
                    [B, 256, 10, 10]

        Returns:
            TrackerOutput containing:

                patch_features:
                    Localized patch from the target image.
                    [B, C, Hp, Wp]

                params:
                    Predicted [tx, ty, angle].
                    [B, 3]

                grid:
                    Sampling coordinates used by grid_sample.
                    [B, Hp, Wp, 2]

                affinity:
                    Patch-to-image correspondence probabilities.
                    [B, H*W, Hp*Wp]
        """
        affinity = self.compute_affinity(
            image_features=image_features,
            patch_features=query_patch_features,
        )

        params = self.localizer(affinity)

        patch_features, grid = self.sampler(
            image_features=image_features,
            params=params,
        )

        return TrackerOutput(
            patch_features=patch_features,
            params=params,
            grid=grid,
            affinity=affinity,
        )  
        
    def alignment_loss(
        self,
        reference_params: Tensor,
        predicted_params: Tensor,
    ) -> Tensor:
        reference_grid = self.sampler.make_grid(
            reference_params,
        )

        predicted_grid = self.sampler.make_grid(
            predicted_params,
        )

        squared_distance = (
            reference_grid - predicted_grid
        ).square().sum(dim=-1)

        return squared_distance.mean()