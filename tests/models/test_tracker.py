"""The tracker on random features, where every cell has a unique match."""
import math

import pytest
import torch
import torch.nn.functional as F

from src.models.tracker import (Tracker, cell_centers, cycle_loss, fit_rigid, patch_grid, patch_layout, place_layout,
                                read_anatomy, run_cycles, sample_features, similarity_loss)
from tests.helpers import STRIDE, random_features


def crop(features, row, column, size=6):
    return features[:, :, row:row + size, column:column + size]


def true_grid(batch, row, column, size=6):
    top_left = torch.tensor([[column * STRIDE, row * STRIDE]]).expand(batch, -1)
    return patch_grid(top_left, size, size, STRIDE)


def sharp_tracker():
    return Tracker(stride=STRIDE, temperature=0.01)


def test_sampling_at_cell_centres_returns_the_features():
    frame = random_features()
    centers = cell_centers(32, 16, STRIDE).expand(2, -1, -1)
    assert torch.allclose(sample_features(frame, centers, STRIDE, (32, 16)), frame, atol=1e-5)


def test_patch_is_found_where_it_came_from():
    frame = random_features()
    hop = sharp_tracker()(crop(frame, 10, 5), frame)

    assert torch.allclose(hop.grid, true_grid(2, 10, 5), atol=0.05)
    assert torch.allclose(hop.angle, torch.zeros(2), atol=1e-3)
    assert torch.allclose(hop.features, crop(frame, 10, 5), atol=1e-3)


def test_patch_follows_a_known_shift():
    frame = random_features()
    shifted = torch.roll(frame, shifts=(3, -2), dims=(2, 3))  # content moves 3 cells down and 2 cells left
    hop = sharp_tracker()(crop(frame, 10, 5), shifted)

    assert torch.allclose(hop.grid, true_grid(2, 13, 3), atol=0.05)


@pytest.mark.parametrize('degrees', [-15.0, 5.0, 10.0])
def test_fit_recovers_a_known_rotation(degrees):
    layout = patch_layout(20, 20, STRIDE).expand(3, -1, -1)
    center = torch.tensor([[100.0, 300.0], [250.0, 500.0], [40.0, 900.0]])
    angle = torch.full((3,), math.radians(degrees))
    points = place_layout(layout, center, angle)

    fitted_center, fitted_angle = fit_rigid(layout, points, max_rotation=math.radians(180), ridge=0.0)
    assert torch.allclose(fitted_center, center, atol=1e-3)
    assert torch.allclose(fitted_angle, angle, atol=1e-2)  # tanh(x / pi) * pi is close to x for small x

    # With the default bound (20 degrees) and ridge, the angle is shrunk but keeps its sign and stays in range.
    _, bounded = fit_rigid(layout, points, max_rotation=math.radians(20))
    assert torch.all(bounded.sign() == angle.sign())
    assert torch.all(bounded.abs() < angle.abs())


def test_rotation_is_bounded():
    layout = patch_layout(20, 20, STRIDE).expand(1, -1, -1)
    points = place_layout(layout, torch.zeros(1, 2), torch.tensor([math.radians(80)]))
    _, angle = fit_rigid(layout, points, max_rotation=math.radians(20))

    assert 0 < angle.item() < math.radians(20)


def test_collapsed_points_give_zero_rotation_and_finite_gradients():
    layout = patch_layout(20, 20, STRIDE).expand(1, -1, -1)
    points = torch.full((1, 400, 2), 256.0, requires_grad=True)  # what a flat affinity produces
    center, angle = fit_rigid(layout, points, max_rotation=math.radians(20))
    (angle.sum() + center.sum()).backward()

    assert angle.item() == 0
    assert torch.isfinite(points.grad).all()


def test_flat_affinity_sends_the_patch_to_the_frame_centre():
    frame = random_features()
    constant = F.normalize(torch.ones(2, 32, 6, 6), dim=1)
    hop = Tracker(stride=STRIDE, temperature=1e6)(constant, frame)

    assert torch.allclose(hop.center, torch.tensor([64.0, 128.0]).expand(2, -1), atol=0.5)


def test_diagnostics_expose_averaging_over_two_peaks():
    frame = random_features(batch=1)
    frame[:, :, 28, 12] = frame[:, :, 4, 2]  # the same feature at two far-apart cells
    query = frame[:, :, 4:5, 2:3]  # a one-cell patch
    hop = sharp_tracker()(query, frame, diagnostics=True)

    first, second = torch.tensor([2.5, 4.5]) * STRIDE, torch.tensor([12.5, 28.5]) * STRIDE
    assert torch.allclose(hop.points[0, 0], (first + second) / 2, atol=0.5)  # lands between the peaks
    half_distance = (second - first).norm() / 2
    assert torch.allclose(hop.diagnostics['peak_to_mean_gap_px'], half_distance.expand(1, 1), atol=0.5)
    assert torch.allclose(hop.diagnostics['spread_px'], half_distance.expand(1, 1), atol=0.5)
    assert torch.allclose(hop.diagnostics['peak_probability'], torch.full((1, 1), 0.5), atol=0.01)


def test_diagnostics_of_a_clean_match():
    frame = random_features()
    hop = sharp_tracker()(crop(frame, 10, 5), frame, diagnostics=True)

    assert hop.diagnostics['peak_probability'].min() > 0.99
    assert hop.diagnostics['peak_to_mean_gap_px'].max() < 0.5
    assert hop.diagnostics['fit_residual_px'].max() < 0.5
    assert hop.diagnostics['entropy'].max() < 0.05


@pytest.mark.parametrize('num_frames, num_calls', [(2, 2), (3, 7), (4, 13)])
def test_cycles_close_on_identical_frames(num_frames, num_calls):
    calls = []

    class CountingTracker(Tracker):
        def __call__(self, *args, **kwargs):
            calls.append(1)
            return super().__call__(*args, **kwargs)

    frame = random_features()
    cycles, first_hops = run_cycles(CountingTracker(stride=STRIDE, temperature=0.01), crop(frame, 10, 5),
                                    [frame] * num_frames)
    priors = num_frames - 1

    assert len(calls) == num_calls
    assert len(first_hops) == priors and first_hops[0].diagnostics is not None
    assert set(cycles) == {f'skip_{i}' for i in range(1, priors + 1)} | {f'long_{i}' for i in range(2, priors + 1)}
    for end_grid in cycles.values():
        assert torch.allclose(end_grid, true_grid(2, 10, 5), atol=0.1)


def test_cycles_close_when_every_frame_is_shifted_differently():
    frame = random_features(height=40, width=24)
    frames = [torch.roll(frame, shifts=shift, dims=(2, 3)) for shift in [(4, 2), (-3, 1), (2, -2)]] + [frame]
    cycles, first_hops = run_cycles(sharp_tracker(), crop(frame, 15, 9), frames)

    for end_grid in cycles.values():
        assert torch.allclose(end_grid, true_grid(2, 15, 9), atol=0.1)
    # first_hops[0] goes into the most recent prior, which was shifted by (2, -2) cells.
    assert torch.allclose(first_hops[0].grid, true_grid(2, 17, 7), atol=0.1)
    assert torch.allclose(first_hops[2].grid, true_grid(2, 19, 11), atol=0.1)


def linear_anatomy(batch, height, width):
    """A stand-in anatomical map: every cell's pixel position divided by 100."""
    return cell_centers(height, width, STRIDE).T.reshape(1, 2, height, width).expand(batch, -1, -1, -1) / 100


def test_reading_the_anatomy_between_cells():
    anatomy = linear_anatomy(2, 32, 16)
    points = torch.tensor([[40.0, 96.0], [3.0, 300.0]])  # a cell corner; off the frame's bottom-left corner

    assert torch.allclose(read_anatomy(anatomy, points, STRIDE), torch.tensor([[0.4, 0.96], [0.04, 2.52]]))
    assert read_anatomy(None, points, STRIDE) is None


def test_zero_anatomical_weight_changes_nothing():
    frame = random_features()
    query = F.normalize(crop(frame, 10, 5) + 0.5 * crop(frame, 20, 9), dim=1)  # ambiguous, so not one-hot
    plain = Tracker(stride=STRIDE, temperature=0.05)(query, frame)
    zero = Tracker(stride=STRIDE, temperature=0.05)(query, frame, torch.zeros(2, 2), random_features(channels=2))

    assert torch.equal(plain.grid, zero.grid) and torch.equal(plain.features, zero.features)


def test_anatomical_prior_resolves_a_repeated_patch():
    """The patch appears twice in both priors; the prior picks the copy at the patch's own anatomical position."""
    frame = random_features(height=40, width=24)
    prior = frame.clone()
    prior[:, :, 30:36, 14:20] = crop(frame, 10, 5)
    frames, start = [prior, prior, frame], true_grid(2, 10, 5)

    without, _ = run_cycles(sharp_tracker(), crop(frame, 10, 5), frames)
    assert (without['skip_1'] - start).norm(dim=-1).mean() > 16  # the hop lands between the copies

    tracker = Tracker(stride=STRIDE, temperature=0.01, anatomical_weight=0.1)
    query_anatomy = start.mean(dim=1) / 100
    cycles, first_hops = run_cycles(tracker, crop(frame, 10, 5), frames, query_anatomy, [linear_anatomy(2, 40, 24)] * 3)

    for end_grid in cycles.values():
        assert torch.allclose(end_grid, start, atol=0.1)
    assert torch.allclose(first_hops[1].anatomy, query_anatomy, atol=1e-4)


def test_losses():
    start = true_grid(2, 10, 5)
    assert torch.allclose(cycle_loss(start, start, patch_size=48), torch.zeros(2))
    # Every cell off by 48 pixels in x is one patch size: squared distance 1.
    moved = start + torch.tensor([48.0, 0.0])
    assert torch.allclose(cycle_loss(start, moved, patch_size=48), torch.ones(2))
    # The loss is isotropic: the same offset in y costs the same.
    assert torch.allclose(cycle_loss(start, start + torch.tensor([0.0, 48.0]), patch_size=48), torch.ones(2))
    # Huber is quadratic below delta and linear above: delta * (distance - delta / 2).
    assert torch.allclose(cycle_loss(start, moved, patch_size=48, huber_delta=0.25),
                          torch.full((2,), 0.25 * (1 - 0.125)))

    features = random_features()
    assert torch.allclose(similarity_loss(features, features), torch.zeros(2), atol=1e-5)
    assert torch.allclose(similarity_loss(features, -features), torch.full((2,), 2.0), atol=1e-5)


def test_gradient_pulls_the_patch_back_to_its_start():
    """One gradient step on the cycle loss must move the returned patch closer to where it started."""
    frame = random_features(seed=1)
    prior = torch.roll(frame, shifts=(2, 1), dims=(2, 3))
    raw = (crop(frame, 10, 5) + 0.8 * crop(frame, 14, 8)).clone().requires_grad_(True)  # an ambiguous query
    tracker = Tracker(stride=STRIDE, temperature=0.07)
    start = true_grid(2, 10, 5)

    def loss_of(raw_query):
        cycles, _ = run_cycles(tracker, F.normalize(raw_query, dim=1), [prior, frame])
        return cycle_loss(start, cycles['skip_1'], patch_size=48).mean()

    loss = loss_of(raw)
    loss.backward()
    assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0
    assert loss_of(raw.detach() - 0.5 * raw.grad / raw.grad.norm()) < loss
