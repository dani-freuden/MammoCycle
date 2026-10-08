import pytest
import torch
import torchvision
from torch import nn

from src.config import Config
from src.models.encoder import ResNetEncoder, load_mirai_trunk
from tests.helpers import pretrained_like


MIRAI_CHECKPOINT = Config.from_name('overfit').encoder.checkpoint


def original_trunk(network, images):
    x = images.expand(-1, 3, -1, -1)
    x = network.maxpool(network.relu(network.bn1(network.conv1(x))))
    return network.layer3(network.layer2(network.layer1(x)))


def test_dilated_trunk_reproduces_the_pretrained_network_at_every_second_cell():
    network = pretrained_like()
    encoder = ResNetEncoder(network.state_dict(), mean=0.0, std=1.0).eval()
    images = torch.rand(2, 1, 256, 128)

    with torch.no_grad():
        ours, original = encoder.trunk(images), original_trunk(network, images)

    assert ours.shape == (2, 256, 32, 16)  # stride 8
    assert original.shape == (2, 256, 16, 8)  # stride 16
    assert torch.allclose(ours[:, :, ::2, ::2], original, rtol=1e-4, atol=1e-3)  # equal up to float error


@pytest.mark.skipif(MIRAI_CHECKPOINT is None or not MIRAI_CHECKPOINT.exists(), reason='needs the Mirai checkpoint')
def test_mirai_checkpoint_loads_completely_and_survives_the_surgery():
    state_dict = load_mirai_trunk(MIRAI_CHECKPOINT)
    network = torchvision.models.resnet18()
    network.load_state_dict(state_dict, strict=False)  # Mirai has no fc
    encoder = ResNetEncoder(state_dict, mean=0.0, std=1.0).eval()  # raises if a trunk key is missing
    images = torch.rand(1, 1, 512, 320)

    with torch.no_grad():
        ours, original = encoder.trunk(images), original_trunk(network.eval(), images)

    assert torch.allclose(ours[:, :, ::2, ::2], original, rtol=1e-4, atol=1e-3)


def test_undilated_trunk_differs_from_the_pretrained_network():
    network = pretrained_like()
    encoder = ResNetEncoder(network.state_dict(), mean=0.0, std=1.0, dilate=False).eval()
    images = torch.rand(2, 1, 256, 128)

    with torch.no_grad():
        assert not torch.allclose(encoder.trunk(images)[:, :, ::2, ::2], original_trunk(network, images), atol=1e-2)


def test_batch_norm_stays_frozen_in_training_mode():
    encoder = ResNetEncoder(pretrained_like().state_dict())
    encoder.train()
    norms = [module for module in encoder.modules() if isinstance(module, nn.BatchNorm2d)]
    before = [norm.running_mean.clone() for norm in norms]
    encoder(torch.rand(2, 1, 64, 64))

    assert encoder.training and not any(norm.training for norm in norms)
    assert all(torch.equal(norm.running_mean, old) for norm, old in zip(norms, before))
    assert all(norm.weight.requires_grad and norm.bias.requires_grad for norm in norms)


def test_batch_norm_trains_when_not_frozen():
    encoder = ResNetEncoder(freeze_batch_norm=False).train()
    assert all(module.training for module in encoder.modules() if isinstance(module, nn.BatchNorm2d))


def test_missing_pretrained_keys_are_reported():
    state_dict = pretrained_like().state_dict()
    del state_dict['layer2.0.conv1.weight']
    with pytest.raises(KeyError, match='layer2.0.conv1.weight'):
        ResNetEncoder(state_dict)


@pytest.mark.parametrize('dilate, receptive_field', [(True, 211), (False, 163)])
def test_a_cell_depends_only_on_its_receptive_field(dilate, receptive_field):
    """With frozen BatchNorm a feature is a function of nearby pixels only, so it cannot encode frame position."""
    encoder = ResNetEncoder(pretrained_like().state_dict(), dilate=dilate).eval()
    images = torch.rand(1, 1, 512, 512, requires_grad=True)
    encoder(images)[0, :, 32, 32].sum().backward()

    touched = images.grad[0, 0].abs() > 0
    rows = touched.any(dim=1).nonzero().flatten()
    assert rows.max() - rows.min() + 1 <= receptive_field
    assert rows.max() - rows.min() + 1 >= receptive_field - 16  # the bound is tight, up to dead ReLU paths


def half_tissue_images(batch=4, seed=0):
    images = torch.rand(batch, 1, 128, 128, generator=torch.Generator().manual_seed(seed)) * 0.9 + 0.1
    images[..., 64:] = 0
    return images


def mean_cosine_between_cells(features):
    cells = torch.nn.functional.normalize(features[..., :8].flatten(2), dim=1)  # tissue cells only
    similarity = torch.einsum('bcm,bcn->bmn', cells, cells)
    off_diagonal = similarity.sum() - similarity.diagonal(dim1=1, dim2=2).sum()
    return off_diagonal / (similarity.numel() - cells.shape[0] * cells.shape[2])


def test_calibration_centres_the_features():
    encoder = ResNetEncoder(pretrained_like().state_dict()).eval()
    images = half_tissue_images()
    with torch.no_grad():
        before = mean_cosine_between_cells(encoder(images))
    assert not encoder.calibrated

    encoder.calibrate([images[:2], images[2:]])
    with torch.no_grad():
        features = encoder(images)
    tissue = features[..., :8]

    assert encoder.calibrated and not encoder.training
    assert before > 0.5  # non-negative features: every cell resembles every other
    assert mean_cosine_between_cells(features).abs() < 0.05  # after centring: unrelated cells are near orthogonal
    assert torch.allclose(tissue.mean(dim=(0, 2, 3)), torch.zeros(256), atol=1e-3)
    live = encoder.output_norm.running_var > 1e-2  # skip near-dead channels, where BatchNorm's eps matters
    assert torch.allclose(tissue.var(dim=(0, 2, 3), unbiased=False)[live], torch.ones(int(live.sum())), atol=1e-2)


def test_calibration_is_saved_with_the_weights_and_survives_training_mode():
    encoder = ResNetEncoder(pretrained_like().state_dict())
    encoder.calibrate([half_tissue_images()])
    mean = encoder.output_norm.running_mean.clone()

    encoder.train()
    encoder(half_tissue_images(seed=1))
    assert torch.equal(encoder.output_norm.running_mean, mean)  # frozen like the other BatchNorm layers

    copy = ResNetEncoder()
    copy.load_state_dict(encoder.state_dict())
    assert copy.calibrated and torch.equal(copy.output_norm.running_mean, mean)


def test_calibration_without_tissue_is_an_error():
    with pytest.raises(ValueError):
        ResNetEncoder().calibrate([torch.zeros(1, 1, 64, 64)])
