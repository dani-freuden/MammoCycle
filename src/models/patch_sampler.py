
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class PatchSampler(nn.Module):
    """
    Bilinear sampler h from TimeCycle.

    Given a full image feature map and localization parameters
    [tx, ty, angle], extracts a fixed-size feature patch.

    Default geometry:

        image features: [B, C, 30, 30]
        output patch:   [B, C, 10, 10]

    Translation parameters are expressed in PyTorch's normalized
    sampling coordinates.

    Angle is expressed in radians.
    """

    def __init__(
        self,
        image_feature_size: int = 30,
        patch_feature_size: int = 10,
        align_corners: bool = False,
    ) -> None:
        super().__init__()

        self.image_feature_size = image_feature_size
        self.patch_feature_size = patch_feature_size
        self.align_corners = align_corners

        self.scale = (
            patch_feature_size / image_feature_size
        )

    def params_to_matrix(self, params: Tensor) -> Tensor:
        """
        Convert [tx, ty, angle] into affine matrices.

        Args:
            params:
                Tensor of shape [B, 3].

        Returns:
            Affine matrices of shape [B, 2, 3].
        """
        if params.ndim != 2 or params.shape[1] != 3:
            raise ValueError(
                "Expected params with shape [B, 3], "
                f"got {params.shape}."
            )

        tx = params[:, 0]
        ty = params[:, 1]
        angle = params[:, 2]

        cos = torch.cos(angle)
        sin = torch.sin(angle)

        scale = self.scale

        matrix = torch.zeros(
            params.shape[0],
            2,
            3,
            dtype=params.dtype,
            device=params.device,
        )

        matrix[:, 0, 0] = scale * cos
        matrix[:, 0, 1] = -scale * sin
        matrix[:, 1, 0] = scale * sin
        matrix[:, 1, 1] = scale * cos

        matrix[:, 0, 2] = tx
        matrix[:, 1, 2] = ty

        return matrix

    def make_grid(
        self,
        params: Tensor,
    ) -> Tensor:
        matrix = self.params_to_matrix(params)

        output_size = (
            params.shape[0],
            1,
            self.patch_feature_size,
            self.patch_feature_size,
        )

        return F.affine_grid(
            matrix,
            size=output_size,
            align_corners=self.align_corners,
        )

    def forward(
        self,
        image_features: Tensor,
        params: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Sample a feature patch from image_features.

        Args:
            image_features:
                [B, C, H, W]

            params:
                [B, 3] containing [tx, ty, angle].

        Returns:
            patch_features:
                [B, C, patch_feature_size, patch_feature_size]

            grid:
                [B, patch_feature_size, patch_feature_size, 2]
        """
        if image_features.ndim != 4:
            raise ValueError(
                "Expected image_features with shape [B, C, H, W], "
                f"got {image_features.shape}."
            )

        batch_size, channels, height, width = image_features.shape

        if height != self.image_feature_size:
            raise ValueError(
                f"Expected feature height {self.image_feature_size}, "
                f"got {height}."
            )

        if width != self.image_feature_size:
            raise ValueError(
                f"Expected feature width {self.image_feature_size}, "
                f"got {width}."
            )

        if params.shape[0] != batch_size:
            raise ValueError(
                "Batch size mismatch between features and params: "
                f"{batch_size} vs {params.shape[0]}."
            )

        grid = self.make_grid(params)

        patch_features = F.grid_sample(
            image_features,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=self.align_corners,
        )

        return patch_features, grid
        
            
    def crop_to_params(
        self,
        crop_xy: Tensor,
        image_size: int,
        patch_size: int,
    ) -> Tensor:
        """
        Convert top-left crop coordinates to affine sampler parameters.

        crop_xy[b] = [x_left, y_top]

        Returns:
            [B, 3] containing [tx, ty, angle].
        """
        if crop_xy.ndim != 2 or crop_xy.shape[1] != 2:
            raise ValueError(
                "Expected crop_xy with shape [B, 2], "
                f"got {crop_xy.shape}."
            )

        if not crop_xy.is_floating_point():
            crop_xy = crop_xy.float()

        x_left = crop_xy[:, 0]
        y_top = crop_xy[:, 1]

        center_x = x_left + patch_size / 2
        center_y = y_top + patch_size / 2

        tx = 2.0 * center_x / image_size - 1.0
        ty = 2.0 * center_y / image_size - 1.0

        angle = torch.zeros_like(tx)

        return torch.stack(
            (tx, ty, angle),
            dim=1,
        )