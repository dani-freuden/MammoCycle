from src.models.patch_sampler import PatchSampler
import torch

def test_center_crop_params():
    sampler = PatchSampler(
        image_feature_size=30,
        patch_feature_size=10,
    )

    crop_xy = torch.tensor([
        [80.0, 80.0],
    ])

    params = sampler.crop_to_params(
        crop_xy=crop_xy,
        image_size=240,
        patch_size=80,
    )

    expected = torch.tensor([
        [0.0, 0.0, 0.0],
    ])

    torch.testing.assert_close(
        params,
        expected,
    )
def test_bottom_right_crop_params():
    sampler = PatchSampler(
        image_feature_size=30,
        patch_feature_size=10,
    )

    crop_xy = torch.tensor([
        [160.0, 160.0],
    ])

    params = sampler.crop_to_params(
        crop_xy=crop_xy,
        image_size=240,
        patch_size=80,
    )

    expected = torch.tensor([
        [2 / 3, 2 / 3, 0.0],
    ])

    torch.testing.assert_close(
        params,
        expected,
    )
def test_bottom_right_crop_params():
    sampler = PatchSampler(
        image_feature_size=30,
        patch_feature_size=10,
    )

    crop_xy = torch.tensor([
        [160.0, 160.0],
    ])

    params = sampler.crop_to_params(
        crop_xy=crop_xy,
        image_size=240,
        patch_size=80,
    )

    expected = torch.tensor([
        [2 / 3, 2 / 3, 0.0],
    ])

    torch.testing.assert_close(
        params,
        expected,
    )
def test_top_left_crop_params():
    sampler = PatchSampler(
        image_feature_size=30,
        patch_feature_size=10,
    )

    crop_xy = torch.tensor([
        [0.0, 0.0],
    ])

    params = sampler.crop_to_params(
        crop_xy=crop_xy,
        image_size=240,
        patch_size=80,
    )

    expected = torch.tensor([
        [-2 / 3, -2 / 3, 0.0],
    ])

    torch.testing.assert_close(
        params,
        expected,
    )