import torch.nn as nn
from torch import Tensor

class PatchLocalizer(nn.Module):
    """
    Localizer g from TimeCycle.

    Takes the patch-to-image affinity tensor and predicts a rigid
    localization transform:

        [tx, ty, rotation]

    For the default TimeCycle geometry:

        affinity: [B, 900, 100]

    which is interpreted as:

        [B, 900, 10, 10]

    before being processed by the ConvNet.
    """

    def __init__(
        self,
        image_feature_size: int = 30,
        patch_feature_size: int = 10,
    ) -> None:
        super().__init__()

        self.image_feature_size = image_feature_size
        self.patch_feature_size = patch_feature_size

        image_positions = image_feature_size**2

        self.conv = nn.Sequential(
            nn.Conv2d(
                image_positions,
                512,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                512,
                512,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(
            512 * patch_feature_size * patch_feature_size,
            3,
        )

        # Our implementation choice:
        # start close to a centered, unrotated patch.
        nn.init.normal_(self.fc.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.fc.bias)

    def forward(self, affinity: Tensor) -> Tensor:
        """
        Args:
            affinity:
                [B, image_positions, patch_positions]

                Default:
                [B, 900, 100]

        Returns:
            Tensor [B, 3]:

                [:, 0] -> tx
                [:, 1] -> ty
                [:, 2] -> rotation
        """
        batch_size, image_positions, patch_positions = affinity.shape

        expected_image_positions = self.image_feature_size**2
        expected_patch_positions = self.patch_feature_size**2

        if image_positions != expected_image_positions:
            raise ValueError(
                f"Expected {expected_image_positions} image positions, "
                f"got {image_positions}."
            )

        if patch_positions != expected_patch_positions:
            raise ValueError(
                f"Expected {expected_patch_positions} patch positions, "
                f"got {patch_positions}."
            )

        # [B, 900, 100]
        #       ->
        # [B, 900, 10, 10]
        x = affinity.reshape(
            batch_size,
            image_positions,
            self.patch_feature_size,
            self.patch_feature_size,
        )

        # [B, 900, 10, 10]
        #       ->
        # [B, 512, 10, 10]
        x = self.conv(x)

        # [B, 512, 10, 10]
        #       ->
        # [B, 51200]
        x = x.flatten(start_dim=1)

        # [B, 51200]
        #       ->
        # [B, 3]
        theta = self.fc(x)

        return theta
    
    