import torch
from src.models.patch_sampler import PatchSampler
from src.models.model import MammoCycleModule


def test_encoder_full_image_shape():
    model = MammoCycleModule()

    x = torch.randn(2, 1, 240, 240)

    features = model.encode(x)

    assert features.shape == (2, 256, 30, 30)


def test_encoder_patch_shape():
    model = MammoCycleModule()

    x = torch.randn(2, 1, 80, 80)

    features = model.encode(x)

    assert features.shape == (2, 256, 10, 10)


def test_encoder_features_are_normalized():
    model = MammoCycleModule()

    x = torch.randn(2, 1, 240, 240)

    features = model.encode(x)

    norms = torch.linalg.vector_norm(
        features,
        ord=2,
        dim=1,
    )

    torch.testing.assert_close(
        norms,
        torch.ones_like(norms),
        atol=1e-5,
        rtol=1e-5,
    )


def test_affinity_shape():
    model = MammoCycleModule()

    image = torch.randn(2, 1, 240, 240)
    patch = torch.randn(2, 1, 80, 80)

    image_features = model.encode(image)
    patch_features = model.encode(patch)

    affinity = model.compute_affinity(
        image_features,
        patch_features,
    )

    assert affinity.shape == (2, 900, 100)


def test_affinity_is_probability_over_image_locations():
    model = MammoCycleModule()

    image = torch.randn(2, 1, 240, 240)
    patch = torch.randn(2, 1, 80, 80)

    image_features = model.encode(image)
    patch_features = model.encode(patch)

    affinity = model.compute_affinity(
        image_features,
        patch_features,
    )

    probabilities = affinity.sum(dim=1)

    torch.testing.assert_close(
        probabilities,
        torch.ones_like(probabilities),
        atol=1e-5,
        rtol=1e-5,
    )
    
def test_localizer_output_shape():
    model = MammoCycleModule()

    image = torch.randn(2, 1, 240, 240)
    patch = torch.randn(2, 1, 80, 80)

    image_features = model.encode(image)
    patch_features = model.encode(patch)

    affinity = model.compute_affinity(
        image_features,
        patch_features,
    )

    theta = model.localizer(affinity)

    assert theta.shape == (2, 3)
    
def test_localizer_is_differentiable():
    model = MammoCycleModule()

    image = torch.randn(2, 1, 240, 240)
    patch = torch.randn(2, 1, 80, 80)

    image_features = model.encode(image)
    patch_features = model.encode(patch)

    affinity = model.compute_affinity(
        image_features,
        patch_features,
    )

    theta = model.localizer(affinity)

    loss = theta.square().mean()
    loss.backward()

    assert model.localizer.fc.weight.grad is not None
    assert model.localizer.conv[0].weight.grad is not None
def test_sampler_zero_transform_returns_center_crop():
    sampler = PatchSampler(
        image_feature_size=30,
        patch_feature_size=10,
    )

    features = torch.arange(
        30 * 30,
        dtype=torch.float32,
    ).reshape(1, 1, 30, 30)

    params = torch.zeros(1, 3)

    sampled, grid = sampler(
        features,
        params,
    )

    expected = features[:, :, 10:20, 10:20]

    assert sampled.shape == (1, 1, 10, 10)
    assert grid.shape == (1, 10, 10, 2)

    torch.testing.assert_close(
        sampled,
        expected,
    )
    
      
def test_sampler_translation_changes_patch():
    sampler = PatchSampler(
        image_feature_size=30,
        patch_feature_size=10,
    )

    features = torch.randn(1, 4, 30, 30)

    center_params = torch.zeros(1, 3)

    shifted_params = torch.tensor([
        [0.25, 0.0, 0.0]
    ])

    center_patch, _ = sampler(
        features,
        center_params,
    )

    shifted_patch, _ = sampler(
        features,
        shifted_params,
    )

    assert not torch.allclose(
        center_patch,
        shifted_patch,
    )
def test_sampler_is_differentiable():
    sampler = PatchSampler(
        image_feature_size=30,
        patch_feature_size=10,
    )

    features = torch.randn(
        2,
        256,
        30,
        30,
        requires_grad=True,
    )

    params = torch.zeros(
        2,
        3,
        requires_grad=True,
    )

    patch, _ = sampler(
        features,
        params,
    )

    loss = patch.square().mean()
    loss.backward()

    assert features.grad is not None
    assert params.grad is not None



def test_tracker_output_shapes():
    model = MammoCycleModule()

    image = torch.randn(2, 1, 240, 240)
    patch = torch.randn(2, 1, 80, 80)

    image_features = model.encode(image)
    patch_features = model.encode(patch)

    output = model.track(
        image_features=image_features,
        query_patch_features=patch_features,
    )

    assert output.patch_features.shape == (
        2,
        256,
        10,
        10,
    )

    assert output.params.shape == (2, 3)

    assert output.grid.shape == (
        2,
        10,
        10,
        2,
    )

    assert output.affinity.shape == (
        2,
        900,
        100,
    )
    
    
def test_tracker_is_end_to_end_differentiable():
    model = MammoCycleModule()

    image = torch.randn(2, 1, 240, 240)
    patch = torch.randn(2, 1, 80, 80)

    image_features = model.encode(image)
    patch_features = model.encode(patch)

    output = model.track(
        image_features=image_features,
        query_patch_features=patch_features,
    )

    loss = output.patch_features.square().mean()

    loss.backward()

    assert model.localizer.fc.weight.grad is not None
    assert model.localizer.conv[0].weight.grad is not None

    assert model.encoder.encoder[0].weight.grad is not None     
if __name__ == '__main__':
    import pytest
    pytest.main()