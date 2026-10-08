"""Augmentations on batched tensors, on any device, applied outside the model.

Geometry is a pixel-space matrix [N, 2, 3] that says where content moves to: p' = A p + t with p = (x, y), in the
pixel convention of the tracker. The same matrix moves an image (warp) and points or boxes on it (transform_points,
transform_boxes), which is what gives synthetic pairs their ground truth.

Intensity changes keep the background at exactly zero, because tissue is found by thresholding.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


def rigid_matrix(shift: Tensor, angle: Tensor, scale: Tensor, center: tuple[float, float]) -> Tensor:
    """Rotate by angle (radians) and scale about center, then shift. shift [N, 2], angle [N], scale [N] -> [N, 2, 3]."""
    cos, sin = scale * angle.cos(), scale * angle.sin()
    linear = torch.stack([torch.stack([cos, -sin], dim=-1), torch.stack([sin, cos], dim=-1)], dim=-2)
    center = shift.new_tensor(center)
    translation = center - linear @ center + shift

    return torch.cat([linear, translation[..., None]], dim=-1)


def random_rigid_matrix(count: int, size: tuple[int, int], max_shift: float, max_rotation_degrees: float,
                        scale_range: tuple[float, float] = (1.0, 1.0), generator: torch.Generator | None = None,
                        device=None) -> Tensor:
    """Independent random transforms about the image centre. size is (height, width)."""
    uniform = lambda low, high, *shape: (torch.rand(count, *shape, generator=generator) * (high - low) + low).to(device)

    return rigid_matrix(shift=uniform(-max_shift, max_shift, 2),
                        angle=uniform(-max_rotation_degrees, max_rotation_degrees) * math.pi / 180,
                        scale=uniform(*scale_range),
                        center=(size[1] / 2, size[0] / 2))


def warp(images: Tensor, matrix: Tensor) -> Tensor:
    """Move the content of images [N, C, H, W] by matrix [N, 2, 3]. Areas that receive no content are zero."""
    height, width = images.shape[-2:]
    bottom = matrix.new_tensor([0.0, 0.0, 1.0]).expand(matrix.shape[0], 1, 3)
    # grid_sample pulls: for every output pixel it needs the input position, which is the inverse transform,
    # and it wants that in normalized coordinates (x_n = 2 x / width - 1).
    inverse = torch.linalg.inv(torch.cat([matrix, bottom], dim=1))
    normalize = matrix.new_tensor([[2 / width, 0, -1], [0, 2 / height, -1], [0, 0, 1]])
    theta = normalize @ inverse @ torch.linalg.inv(normalize)
    grid = F.affine_grid(theta[:, :2], list(images.shape), align_corners=False)

    return F.grid_sample(images, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def transform_points(points: Tensor, matrix: Tensor) -> Tensor:
    """points [N, K, 2] as (x, y) -> where the same content is after warp(images, matrix)."""
    return points @ matrix[:, :, :2].transpose(1, 2) + matrix[:, None, :, 2]


def transform_boxes(boxes: Tensor, matrix: Tensor) -> Tensor:
    """boxes [N, 4] as (x1, y1, x2, y2) -> the axis-aligned box around the four transformed corners."""
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    corners = torch.stack([torch.stack([x1, y1], -1), torch.stack([x2, y1], -1),
                           torch.stack([x2, y2], -1), torch.stack([x1, y2], -1)], dim=1)
    moved = transform_points(corners, matrix)

    return torch.cat([moved.amin(dim=1), moved.amax(dim=1)], dim=-1)


def augment_intensity(images: Tensor, gamma_range: tuple[float, float] = (0.8, 1.25),
                      gain_range: tuple[float, float] = (0.9, 1.1), max_noise: float = 0.02,
                      blur_probability: float = 0.3, generator: torch.Generator | None = None) -> Tensor:
    """Independent gamma, gain, blur and noise for each image in [N, 1, H, W]. Background (zero) stays zero."""
    count, device = images.shape[0], images.device
    uniform = lambda low, high: (torch.rand(count, 1, 1, 1, generator=generator) * (high - low) + low).to(device)
    tissue = images > 0

    out = images.clamp_min(0) ** uniform(*gamma_range) * uniform(*gain_range)

    kernel = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device) / 16
    blurred = F.conv2d(F.conv2d(out, kernel.view(1, 1, 1, 5), padding=(0, 2)), kernel.view(1, 1, 5, 1), padding=(2, 0))
    out = torch.where(uniform(0, 1) < blur_probability, blurred, out)

    noise = torch.randn(images.shape, generator=generator).to(device) * uniform(0, max_noise)

    return ((out + noise).clamp(0, 1) * tissue).to(images.dtype)


@dataclass
class FrameAugmentation:
    """Training augmentation of frames [B, T, 1, H, W]: every frame gets its own geometry and intensity.

    Independent geometry per frame removes the "same coordinates on the other exam" shortcut and imitates positioning
    differences. Reversing a record in time, with reverse_probability, lets the query patch come from the oldest exam.
    """
    max_shift: float = 48.0
    max_rotation_degrees: float = 5.0
    scale_range: tuple[float, float] = (1.0, 1.0)
    reverse_probability: float = 0.5
    intensity: bool = True

    def __call__(self, frames: Tensor, generator: torch.Generator | None = None) -> Tensor:
        batch, length = frames.shape[:2]
        matrix = random_rigid_matrix(batch * length, frames.shape[-2:], self.max_shift, self.max_rotation_degrees,
                                     self.scale_range, generator, frames.device)
        out = warp(frames.flatten(0, 1), matrix)
        if self.intensity:
            out = augment_intensity(out, generator=generator)
        out = out.unflatten(0, (batch, length))

        reverse = (torch.rand(batch, generator=generator) < self.reverse_probability).to(frames.device)

        return torch.where(reverse.view(-1, 1, 1, 1, 1), out.flip(1), out)


def synthetic_prior(images: Tensor, boxes: Tensor, max_shift: float = 64.0, max_rotation_degrees: float = 8.0,
                    scale_range: tuple[float, float] = (0.9, 1.1),
                    generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
    """A fake prior of each image from a known warp and intensity change, so the true position of every box is known.

    images [N, 1, H, W], boxes [N, 4] on them. Returns the priors [N, 1, H, W] and the boxes [N, 4] on the priors.
    It measures robustness to our own warps, not to real change over years.

    TODO(elastic): add a smooth random displacement field to imitate compression; its ground truth is the field.
    """
    matrix = random_rigid_matrix(images.shape[0], images.shape[-2:], max_shift, max_rotation_degrees, scale_range,
                                 generator, images.device)

    return augment_intensity(warp(images, matrix), generator=generator), transform_boxes(boxes, matrix)
