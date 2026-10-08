import lightning as L
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.augmentations import FrameAugmentation, rigid_matrix, transform_boxes, warp
from src.models.encoder import ResNetEncoder
from src.models.module import MammoCycleModule, background_landing, box_metrics
from tests.helpers import calibrated_encoder, pretrained_like, textured_frames


HEIGHT, WIDTH = 256, 192


def make_module(encoder=None, **kwargs):
    return MammoCycleModule(encoder or calibrated_encoder(), **{'patch_size': 96, 'num_patches': 3, **kwargs})


def records(batch, num_frames, seed=0):
    """Records made of copies of one textured frame."""
    return textured_frames(batch, seed=seed)[:, None].expand(-1, num_frames, -1, -1, -1).contiguous()


# ---------------------------------------------------------------- the step


@pytest.mark.parametrize('num_frames', [2, 3, 4])
def test_step_shapes_and_keys(num_frames):
    module = make_module()
    output = module.cycle_step(records(2, num_frames))

    priors = num_frames - 1
    assert set(output.cycle_losses) == ({f'skip_{i}' for i in range(1, priors + 1)}
                                        | {f'long_{i}' for i in range(2, priors + 1)})
    assert set(output.similarity_losses) == {f'prior_{i}' for i in range(1, priors + 1)}
    assert all(value.shape == (6,) for value in output.cycle_losses.values())
    assert output.query.shape == (6, 256, 12, 12) and output.start_grid.shape == (6, 144, 2)
    assert len(output.frame_features) == num_frames and output.frame_features[0].shape == (6, 256, 40, 24)
    assert output.loss.dim() == 0 and torch.isfinite(output.loss)

    output.loss.backward()
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients) > 0


def test_identical_frames_give_small_cycle_error_even_before_training():
    with torch.no_grad():
        output = make_module(num_patches=4).cycle_step(records(2, 2), torch.Generator().manual_seed(0))

    assert output.cycle_errors['skip_1'].mean() < 16  # two cells; what remains is the pull towards the centre
    assert output.similarity_losses['prior_1'].mean() < 0.2


def test_same_patches_are_sampled_with_the_same_generator():
    module, frames = make_module(), records(2, 2)
    with torch.no_grad():
        first, second = (module.cycle_step(frames, torch.Generator().manual_seed(5)) for _ in range(2))

    assert torch.equal(first.start_grid, second.start_grid) and torch.equal(first.loss, second.loss)


def test_different_patient_error_is_larger_than_same_patient_error():
    module = make_module()
    with torch.no_grad():
        output = module.cycle_step(records(4, 2), torch.Generator().manual_seed(0))
        other = module.different_patient_error(output)

    assert other.shape == (12,)
    assert other.mean() > 2 * output.cycle_errors['skip_1'].mean()


def test_background_landing_counts_points_outside_tissue():
    tissue = torch.zeros(2, 1, 64, 32)
    tissue[..., :16] = 1  # tissue is the left half
    # Two inside, one on background, one off the frame; then four inside.
    points = torch.tensor([[[4.0, 10.0], [12.0, 50.0], [20.0, 10.0], [40.0, 10.0]],
                           [[4.0, 10.0], [5.0, 20.0], [6.0, 30.0], [7.0, 40.0]]])
    assert torch.allclose(background_landing(points, tissue), torch.tensor([0.5, 0.0]))


def test_training_on_one_batch_reduces_the_loss():
    module = make_module(num_patches=4)
    augment = FrameAugmentation(max_shift=12, max_rotation_degrees=3, reverse_probability=0)
    frames = augment(records(2, 3, seed=1), generator=torch.Generator().manual_seed(0))
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-4)

    losses = []
    for _ in range(12):
        output = module.cycle_step(frames, torch.Generator().manual_seed(0))
        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()
        losses.append(output.loss.item())

    assert all(map(torch.isfinite, torch.tensor(losses)))
    assert min(losses[-3:]) < losses[0]


# ---------------------------------------------------------------- inference


@pytest.mark.parametrize('method', ['window', 'propagation'])
def test_both_inference_paths_follow_a_known_shift(method):
    """The frame is large enough for the boxes to sit more than half a receptive field from every border.

    In a small frame every cell sees a frame border, a shift changes what it sees, and the match degrades.
    """
    height, width = 640, 448
    module = make_module(calibrated_encoder(height, width), temperature=0.05).eval()  # the stand-in's temperature
    current = textured_frames(2, height, width, seed=2)
    matrix = rigid_matrix(torch.tensor([[16.0, -24.0]] * 2), torch.zeros(2), torch.ones(2),
                          center=(width / 2, height / 2))
    prior = warp(current, matrix)  # content moves 16 right and 24 up
    boxes = torch.tensor([[164.0, 290.0, 196.0, 330.0], [184.0, 340.0, 224.0, 372.0]])
    expected = transform_boxes(boxes, matrix)

    if method == 'window':
        result = module.locate_by_window(current, prior, boxes)
        assert set(result) == {'center', 'box', 'round_trip_error', 'similarity'}
        assert torch.allclose(result['box'], expected, atol=2.0)
        assert (result['round_trip_error'] < 2).all() and (result['similarity'] > 0.9).all()
    else:
        result = module.locate_by_propagation(current, prior, boxes)
        assert set(result) == {'center', 'box', 'score', 'found', 'map'}
        assert result['found'].all() and result['map'].shape == (2, height // 8, width // 8)

    assert (result['center'] - (expected[:, :2] + expected[:, 2:]) / 2).norm(dim=1).max() < 8  # within one cell


def test_box_metrics():
    target = torch.tensor([[10.0, 10.0, 30.0, 50.0], [10.0, 10.0, 30.0, 50.0]])
    box = torch.tensor([[10.0, 10.0, 30.0, 50.0], [30.0, 10.0, 50.0, 50.0]])
    center = (box[:, :2] + box[:, 2:]) / 2
    metrics = box_metrics(center, box, target)

    assert torch.equal(metrics['hit_rate'], torch.tensor([1.0, 0.0]))
    assert torch.allclose(metrics['center_distance_px'], torch.tensor([0.0, 20.0]))
    assert torch.allclose(metrics['iou'], torch.tensor([1.0, 0.0]))


# ---------------------------------------------------------------- Lightning


class Records(Dataset):
    """Each item is one record: num_frames augmented views of one textured image."""

    def __init__(self, count, num_frames):
        frames = textured_frames(count, HEIGHT, WIDTH, seed=num_frames)[:, None].expand(-1, num_frames, -1, -1, -1)
        augment = FrameAugmentation(max_shift=8, max_rotation_degrees=2, reverse_probability=0)
        self.frames = augment(frames.contiguous(), generator=torch.Generator().manual_seed(0))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return {'frames': self.frames[index]}


def make_trainer(**kwargs):
    return L.Trainer(accelerator='cpu', max_steps=3, logger=False, enable_checkpointing=False,
                     enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0, **kwargs)


@pytest.mark.parametrize('num_frames, precision', [(2, '32-true'), (3, '32-true'), (3, 'bf16-mixed')])
def test_fit_and_validate(num_frames, precision):
    module = make_module(calibrated_encoder(HEIGHT, WIDTH), num_patches=2, warmup_steps=2)
    before = module.encoder.trunk[0].weight.clone()
    running_mean = module.encoder.output_norm.running_mean.clone()

    trainer = make_trainer(precision=precision, limit_val_batches=2)
    trainer.fit(module, DataLoader(Records(6, num_frames), batch_size=2), DataLoader(Records(4, 2), batch_size=2))
    metrics = trainer.callback_metrics

    priors = num_frames - 1
    expected = {'train/loss', 'train/cycle_loss', 'train/similarity_loss', 'train/localizer/spread_px',
                'train/localizer/peak_to_mean_gap_px', 'train/localizer/peak_probability',
                *(f'train/cycle_loss/skip_{i}' for i in range(1, priors + 1)),
                *(f'train/cycle_loss/long_{i}' for i in range(2, priors + 1)),
                *(f'train/similarity_loss/prior_{i}' for i in range(1, priors + 1)),
                'val/loss', 'val/cycle_error_px/skip_1', 'val/cycle_error_px/median', 'val/cycle_error_px/p90',
                'val/cycle_closed_share', 'val/different_patient_error_px', 'val/different_patient_ratio',
                'val/background_landing', 'val/localizer/fit_residual_px',
                *(f'val/box/{method}/{name}' for method in ('propagate', 'window', 'identity')
                  for name in ('hit_rate', 'center_distance_px', 'iou'))}
    assert expected <= set(metrics), expected - set(metrics)
    assert all(torch.isfinite(value) for value in metrics.values())

    assert not torch.equal(module.encoder.trunk[0].weight, before)  # the encoder was trained
    assert torch.equal(module.encoder.output_norm.running_mean, running_mean)  # its statistics were not
    assert not any(m.training for m in module.encoder.modules() if isinstance(m, nn.BatchNorm2d))


def test_fit_refuses_an_uncalibrated_encoder():
    module = make_module(ResNetEncoder(pretrained_like().state_dict()))
    with pytest.raises(RuntimeError, match='calibrate'):
        make_trainer().fit(module, DataLoader(Records(2, 2), batch_size=2))


def test_only_the_encoder_has_parameters_and_it_stays_out_of_hparams():
    module = make_module()
    assert all(name.startswith('encoder.') for name, _ in module.named_parameters())
    assert 'encoder' not in module.hparams and module.hparams.patch_size == 96


def test_patch_size_off_the_stride_grid_is_rejected():
    with pytest.raises(ValueError):
        MammoCycleModule(ResNetEncoder(), patch_size=100)
