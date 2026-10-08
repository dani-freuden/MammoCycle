import torch

from src.models.propagation import box_coverage, locate, propagate_labels
from tests.helpers import STRIDE, random_features


BOXES = torch.tensor([[40.0, 80.0, 72.0, 120.0]] * 2)  # cells 10-14 x 5-8


def test_box_coverage_measures_cell_coverage():
    coverage = box_coverage(torch.tensor([[8.0, 16.0, 20.0, 32.0]]), height=6, width=4, stride=8)

    expected = torch.zeros(1, 6, 4)
    expected[0, 2:4, 1] = 1.0  # x from 8 to 16 is fully covered
    expected[0, 2:4, 2] = 0.5  # x from 16 to 20 covers half of the cell from 16 to 24
    assert torch.allclose(coverage, expected)


def test_labels_follow_a_known_shift():
    source = random_features()
    target = torch.roll(source, shifts=(3, -2), dims=(2, 3))
    labels = box_coverage(BOXES, 32, 16, STRIDE)[:, None]

    propagated = propagate_labels(source, target, labels, temperature=0.01, stride=STRIDE)
    assert torch.allclose(propagated, torch.roll(labels, shifts=(3, -2), dims=(2, 3)), atol=1e-4)

    result = locate(propagated[:, 0], STRIDE)
    assert result['found'].all()
    assert torch.allclose(result['center'], torch.tensor([[56.0 - 16, 100.0 + 24]] * 2), atol=0.01)
    assert torch.allclose(result['box'], torch.tensor([[40.0 - 16, 80.0 + 24, 72.0 - 16, 120.0 + 24]] * 2))


def test_radius_removes_a_far_away_false_match():
    source = random_features(batch=1)
    target = source.clone()
    target[:, :, 28, 12] = source[:, :, 11, 6]  # a far cell that looks like a lesion cell
    labels = box_coverage(BOXES[:1], 32, 16, STRIDE)[:, None]

    everywhere = propagate_labels(source, target, labels, temperature=0.01, stride=STRIDE)
    nearby = propagate_labels(source, target, labels, temperature=0.01, stride=STRIDE, radius=64.0)
    assert everywhere[0, 0, 28, 12] > 0.99
    assert nearby[0, 0, 28, 12] < 0.01
    assert torch.allclose(nearby[0, 0, 10:15, 5:9], torch.ones(5, 4), atol=1e-4)


def test_chunking_does_not_change_the_result():
    source, target = random_features(seed=0), random_features(seed=1)
    labels = box_coverage(BOXES, 32, 16, STRIDE)[:, None]
    kwargs = dict(temperature=0.07, stride=STRIDE, radius=100.0)

    assert torch.allclose(propagate_labels(source, target, labels, chunk=7, **kwargs),
                          propagate_labels(source, target, labels, chunk=4096, **kwargs), atol=1e-5)


def test_locate_falls_back_to_the_peak():
    lesion = torch.zeros(1, 32, 16)
    lesion[0, 20, 7] = 0.3
    result = locate(lesion, STRIDE)

    assert not result['found'].item()
    assert torch.allclose(result['score'], torch.tensor([0.3]))
    assert torch.allclose(result['center'], torch.tensor([[7.5 * 8, 20.5 * 8]]))
    assert torch.allclose(result['box'], torch.tensor([[56.0, 160.0, 64.0, 168.0]]))
