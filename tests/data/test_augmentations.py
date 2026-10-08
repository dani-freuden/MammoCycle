import math

import torch

from src.data.augmentations import (FrameAugmentation, augment_intensity, random_rigid_matrix, rigid_matrix,
                                    synthetic_prior, transform_boxes, transform_points, warp)


def blob_image(center, height=128, width=64, sigma=3.0):
    ys = torch.arange(height).view(-1, 1) + 0.5
    xs = torch.arange(width).view(1, -1) + 0.5
    return torch.exp(-0.5 * (((xs - center[0]) / sigma) ** 2 + ((ys - center[1]) / sigma) ** 2))[None, None]


def blob_center(image):
    """Intensity-weighted centre (x, y) of a single-blob image."""
    height, width = image.shape[-2:]
    ys = torch.arange(height).view(-1, 1) + 0.5
    xs = torch.arange(width).view(1, -1) + 0.5
    weight = image[0, 0] / image[0, 0].sum()
    return torch.stack([(weight * xs).sum(), (weight * ys).sum()])


def test_whole_pixel_shift_is_exact():
    image = torch.rand(1, 1, 128, 64)  # not square, like the frames
    matrix = rigid_matrix(torch.tensor([[5.0, -7.0]]), torch.zeros(1), torch.ones(1), center=(32, 64))
    moved = warp(image, matrix)

    assert torch.allclose(moved[..., :121, 5:], image[..., 7:, :59], atol=1e-5)  # content went 5 right and 7 up
    assert moved[..., 121:, :].abs().sum() == 0 and moved[..., :, :5].abs().sum() == 0


def test_points_move_with_the_image_content():
    start = torch.tensor([20.0, 50.0])
    for degrees, scale, shift in [(10.0, 1.0, (4.0, -6.0)), (-7.0, 1.1, (-3.0, 9.0)), (15.0, 0.9, (0.0, 0.0))]:
        matrix = rigid_matrix(torch.tensor([shift]), torch.tensor([math.radians(degrees)]), torch.tensor([scale]),
                              center=(32, 64))
        expected = transform_points(start.view(1, 1, 2), matrix)[0, 0]
        assert torch.allclose(blob_center(warp(blob_image(start), matrix)), expected, atol=0.1)


def test_boxes_bound_the_transformed_corners():
    matrix = rigid_matrix(torch.tensor([[10.0, 0.0]]), torch.tensor([math.pi / 2]), torch.ones(1), center=(0, 0))
    box = transform_boxes(torch.tensor([[10.0, 20.0, 30.0, 60.0]]), matrix)  # (x, y) -> (-y + 10, x)

    assert torch.allclose(box, torch.tensor([[-50.0, 10.0, -10.0, 30.0]]), atol=1e-4)


def test_intensity_keeps_background_at_zero_and_changes_tissue():
    images = torch.zeros(4, 1, 64, 64)
    images[..., 16:48, 16:48] = torch.rand(4, 1, 32, 32) * 0.8 + 0.1
    out = augment_intensity(images, blur_probability=1.0, generator=torch.Generator().manual_seed(0))

    assert (out[images == 0] == 0).all()
    assert out.min() >= 0 and out.max() <= 1
    assert not torch.allclose(out, images)
    assert not torch.allclose(out[0] - images[0], out[1] - images[1])  # each image gets its own change


def test_frame_augmentation_treats_every_frame_independently():
    frame = torch.zeros(1, 64, 64)
    frame[..., 16:48, 16:48] = 0.5
    frames = frame.expand(3, 4, 1, 64, 64).clone()
    out = FrameAugmentation(max_shift=8, max_rotation_degrees=5)(frames, generator=torch.Generator().manual_seed(0))

    assert out.shape == (3, 4, 1, 64, 64)
    assert not torch.allclose(out[0, 0], out[0, 1])  # different geometry per frame
    assert torch.equal(frames, frame.expand(3, 4, 1, 64, 64))  # the input is not modified


def test_time_reversal_flips_the_frames():
    frames = torch.arange(4.0).view(1, 4, 1, 1, 1).expand(2, 4, 1, 8, 8) / 10
    always = FrameAugmentation(max_shift=0, max_rotation_degrees=0, reverse_probability=1.0, intensity=False)
    never = FrameAugmentation(max_shift=0, max_rotation_degrees=0, reverse_probability=0.0, intensity=False)

    assert torch.allclose(always(frames)[:, :, 0, 4, 4], torch.tensor([0.3, 0.2, 0.1, 0.0]).expand(2, 4), atol=1e-5)
    assert torch.allclose(never(frames), frames, atol=1e-5)


def test_synthetic_prior_knows_where_the_box_went():
    images = blob_image(torch.tensor([30.0, 70.0]), sigma=4.0).clamp_min(1e-3)  # a blob on dim "tissue"
    boxes = torch.tensor([[22.0, 62.0, 38.0, 78.0]])
    prior, target = synthetic_prior(images, boxes, max_shift=10, generator=torch.Generator().manual_seed(3))

    assert prior.shape == (1, 1, 128, 64)
    found = blob_center((prior > 0.5 * prior.max()).float())  # centre of the blob's bright core
    assert torch.allclose(found, (target[0, :2] + target[0, 2:]) / 2, atol=1.0)


def test_random_matrices_respect_their_limits():
    matrix = random_rigid_matrix(500, (128, 64), max_shift=10, max_rotation_degrees=5, scale_range=(0.9, 1.1),
                                 generator=torch.Generator().manual_seed(0))
    scale = matrix[:, :, :2].det().sqrt()
    angle = torch.atan2(matrix[:, 1, 0], matrix[:, 0, 0]).rad2deg()
    center = torch.tensor([32.0, 64.0])
    shift = transform_points(center.expand(500, 1, 2), matrix)[:, 0] - center

    assert scale.min() >= 0.9 and scale.max() <= 1.1 and scale.std() > 0.01
    assert angle.abs().max() <= 5 and angle.std() > 1
    assert shift.abs().max() <= 10 + 1e-3 and shift.std() > 1
