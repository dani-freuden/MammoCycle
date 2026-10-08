"""The parameter-free tracker, the cycles it runs and their losses.

Conventions:
- Positions are (x, y) in frame pixels, x along the width. Pixel i covers [i, i + 1), so the centre of feature cell j
  at stride s is at (j + 0.5) * s. grid_sample's normalized coordinate is then 2 * x / size - 1 (align_corners=False),
  for an image and for its feature map alike.
- The fit and the losses work in pixels: the frame is not square, so normalized units differ in x and y.
- Features reaching this code are L2-normalized over channels, and it must run in float32.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


def cell_centers(height: int, width: int, stride: int, device=None) -> Tensor:
    """Pixel coordinates (x, y) of every feature cell centre, in row-major order. Shape [height * width, 2]."""
    ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * stride
    xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * stride
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

    return torch.stack([grid_x, grid_y], dim=-1).flatten(0, 1)


def patch_layout(height: int, width: int, stride: int, device=None) -> Tensor:
    """Cell centres of a patch relative to the patch centre. Shape [height * width, 2], mean zero."""
    centers = cell_centers(height, width, stride, device)

    return centers - centers.mean(dim=0)


def patch_grid(top_left: Tensor, height: int, width: int, stride: int) -> Tensor:
    """Cell centres of patches whose top-left pixel corners are top_left [B, 2]. Shape [B, height * width, 2]."""
    return top_left[:, None].float() + cell_centers(height, width, stride, top_left.device)


def affinity(query: Tensor, frame: Tensor, temperature: float, penalty: Tensor | None = None) -> Tensor:
    """For each query cell, a probability distribution over the frame cells.

    query [B, C, hq, wq], frame [B, C, hf, wf] -> [B, hq * wq, hf * wf], each row sums to one. penalty, broadcastable to
    the output, is subtracted from the cosine similarity before the temperature.
    """
    similarity = torch.einsum('bcq,bcf->bqf', query.flatten(2), frame.flatten(2))
    if penalty is not None:
        similarity = similarity - penalty

    return (similarity / temperature).softmax(dim=-1)


def read_anatomy(anatomy: Tensor | None, points: Tensor, stride: int) -> Tensor | None:
    """Bilinearly read anatomical maps [B, 2, hf, wf] at pixel positions points [B, 2]. Returns [B, 2].

    No maps give None. Positions off the frame read the nearest border cell.
    """
    if anatomy is None:
        return None

    size = points.new_tensor([anatomy.shape[-1] * stride, anatomy.shape[-2] * stride])  # (width, height) in pixels
    normalized = (2 * points / size - 1)[:, None, None]  # [B, 1, 1, 2]

    return F.grid_sample(anatomy, normalized, mode='bilinear', padding_mode='border', align_corners=False)[..., 0, 0]


def fit_rigid(layout: Tensor, points: Tensor, max_rotation: float, ridge: float = 0.01) -> tuple[Tensor, Tensor]:
    """Least-squares shift and rotation that map the patch layout onto the predicted points.

    layout [B, N, 2] with mean zero, points [B, N, 2]. Returns the centre [B, 2] and the angle [B] in radians.

    The ridge term shrinks the angle slightly towards zero (about 1% for the default). It keeps the angle and its
    gradient well defined when the points collapse onto one location, which happens when the affinity is flat.
    The angle is then squashed into (-max_rotation, max_rotation).
    """
    center = points.mean(dim=1)
    centred = points - center[:, None]

    dot = (layout * centred).sum(dim=(1, 2))
    cross = (layout[..., 0] * centred[..., 1] - layout[..., 1] * centred[..., 0]).sum(dim=1)
    energy = layout.square().sum(dim=(1, 2))
    angle = torch.atan2(cross, dot + ridge * energy)

    return center, max_rotation * torch.tanh(angle / max_rotation)


def place_layout(layout: Tensor, center: Tensor, angle: Tensor) -> Tensor:
    """Rotate the layout by angle and move it to center. [B, N, 2] -> [B, N, 2] pixel coordinates."""
    cos, sin = angle.cos()[:, None], angle.sin()[:, None]
    x = cos * layout[..., 0] - sin * layout[..., 1]
    y = sin * layout[..., 0] + cos * layout[..., 1]

    return torch.stack([x, y], dim=-1) + center[:, None]


def sample_features(frame: Tensor, grid: Tensor, stride: int, shape: tuple[int, int]) -> Tensor:
    """Bilinearly sample frame features at pixel coordinates and re-normalize them.

    frame [B, C, hf, wf], grid [B, N, 2] with N = shape[0] * shape[1] -> [B, C, *shape].
    Positions outside the frame give zero vectors.
    """
    size = grid.new_tensor([frame.shape[-1] * stride, frame.shape[-2] * stride])  # (width, height) in pixels
    normalized = (2 * grid / size - 1).unflatten(1, shape)
    sampled = F.grid_sample(frame, normalized, mode='bilinear', padding_mode='zeros', align_corners=False)

    return F.normalize(sampled, dim=1)


def localizer_diagnostics(probability: Tensor, frame_centers: Tensor, points: Tensor,
                          grid: Tensor) -> dict[str, Tensor]:
    """Per-cell statistics, each [B, Nq], that tell a feature failure from an averaging failure.

    peak_probability     highest probability in the cell's distribution; rises as features sharpen
    entropy              entropy of the distribution divided by its maximum: 0 is one-hot, 1 is flat
    spread_px            spatial standard deviation of the distribution around its expected position
    peak_to_mean_gap_px  distance between the most likely and the expected position; large means the expectation
                         averages separate peaks
    fit_residual_px      distance between the cell's expected position and where the fit puts it; large means the
                         cells disagree with each other
    """
    peak_probability, peak_index = probability.max(dim=-1)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1) / math.log(probability.shape[-1])
    second_moment = probability @ frame_centers.square().sum(dim=-1)

    return {
        'peak_probability': peak_probability,
        'entropy': entropy,
        'spread_px': (second_moment - points.square().sum(dim=-1)).clamp_min(0).sqrt(),
        'peak_to_mean_gap_px': (frame_centers[peak_index] - points).norm(dim=-1),
        'fit_residual_px': (points - grid).norm(dim=-1),
    }


@dataclass
class Hop:
    features: Tensor  # [B, C, hq, wq] the patch's features as found in the frame, L2-normalized
    grid: Tensor  # [B, hq * wq, 2] pixel position of each patch cell after the fit
    points: Tensor  # [B, hq * wq, 2] each cell's expected position before the fit
    center: Tensor  # [B, 2]
    angle: Tensor  # [B] radians
    anatomy: Tensor | None = None  # [B, 2] anatomical coordinate at center, when the frame has a map
    diagnostics: dict[str, Tensor] | None = None  # see localizer_diagnostics


@dataclass
class Tracker:
    """One hop: find the query patch in a frame and read its features there. Has no parameters.

    Each query cell gets a probability map over the frame cells, which is reduced to its expected position. A shift
    and a rotation are fitted to those points in closed form, and the frame's features are read at the fitted grid.

    Anatomical prior: given the anatomical coordinate a_q of the query patch centre and a map of a_j over the frame
    cells (src/utils/anatomy.py), every cell's similarity is lowered by anatomical_weight * |a_q - a_j|^2 before the
    temperature. Both are fixed metadata, so gradients reach the features only through the softmax. It multiplies
    the probabilities by a Gaussian in anatomical units of sigma = sqrt(temperature / (2 * anatomical_weight)): 0.39 for
    0.1 and 0.12 for 1 at temperature 0.03, 1 being the distance from the centroid to the skin.

    TODO(coarse-to-fine): match globally on this grid, then refine at stride 4 in a small window around the match.
    TODO(confidence weighting): weight each cell in fit_rigid by its peak probability.
    """
    stride: int = 8
    temperature: float = 0.03
    max_rotation_degrees: float = 20.0
    anatomical_weight: float = 0.0

    def __call__(self, query: Tensor, frame: Tensor, query_anatomy: Tensor | None = None,
                 frame_anatomy: Tensor | None = None, diagnostics: bool = False) -> Hop:
        """query [B, C, hq, wq] and frame [B, C, hf, wf], both L2-normalized.

        For the anatomical prior, query_anatomy [B, 2] is a_q at the query patch centre and frame_anatomy
        [B, 2, hf, wf] is a_j at every frame cell. Without them the prior is off.
        """
        batch, _, height, width = query.shape
        layout = patch_layout(height, width, self.stride, query.device).expand(batch, -1, -1)
        frame_centers = cell_centers(frame.shape[-2], frame.shape[-1], self.stride, frame.device)

        penalty = None
        if frame_anatomy is not None:
            distance = (frame_anatomy.flatten(2) - query_anatomy[:, :, None]).square().sum(dim=1)  # [B, hf * wf]
            penalty = self.anatomical_weight * distance[:, None]  # the same for every query cell

        probability = affinity(query, frame, self.temperature, penalty)
        points = probability @ frame_centers  # each query cell's expected position
        center, angle = fit_rigid(layout, points, math.radians(self.max_rotation_degrees))
        grid = place_layout(layout, center, angle)
        hop = Hop(sample_features(frame, grid, self.stride, (height, width)), grid, points, center, angle,
                  read_anatomy(frame_anatomy, center.detach(), self.stride))

        if diagnostics:
            hop.diagnostics = localizer_diagnostics(probability.detach(), frame_centers, points.detach(), grid.detach())

        return hop


def run_cycles(tracker: Tracker, query: Tensor, frames: list[Tensor], query_anatomy: Tensor | None = None,
               frame_anatomy: list[Tensor] | None = None) -> tuple[dict[str, Tensor], list[Hop]]:
    """Run every cycle that starts from the query patch in the last frame.

    query [B, C, hq, wq] is the patch's features in the last frame; frames are T feature maps [B, C, hf, wf], oldest
    first, the last one being the query frame. With k = T - 1 priors there are k skip cycles, 'skip_i' straight to the
    i-th prior and back, and k - 1 long cycles, 'long_i' back through every frame to the i-th prior and forward through
    every frame again (the long cycle of length 1 is the skip cycle of length 1).

    Returns where each cycle ends in the query frame, as a grid [B, hq * wq, 2], and the k first hops, first_hops[i - 1]
    into the i-th prior, for the similarity loss. first_hops[0] carries diagnostics. It makes 3k - 2 + k(k + 1) / 2
    tracker calls: 2, 7 and 13 for T = 2, 3 and 4.

    For the anatomical prior, query_anatomy [B, 2] is a_q at the query patch centre and frame_anatomy the T maps
    [B, 2, hf, wf]. Every later hop takes a_q where the previous hop landed.
    """
    last = len(frames) - 1
    maps = frame_anatomy or [None] * len(frames)

    # Backward chain: query -> last - 1 -> last - 2 -> ... -> 0. chain[j] is the hop into frame last - 1 - j.
    chain, features, anatomy = [], query, query_anatomy
    for target in range(last - 1, -1, -1):
        chain.append(tracker(features, frames[target], anatomy, maps[target], diagnostics=not chain))
        features, anatomy = chain[-1].features, chain[-1].anatomy

    cycles, first_hops = {}, []
    for length in range(1, last + 1):
        prior = last - length
        first = chain[0] if length == 1 else tracker(query, frames[prior], query_anatomy, maps[prior])
        first_hops.append(first)
        cycles[f'skip_{length}'] = tracker(first.features, frames[last], first.anatomy, maps[last]).grid

        if length > 1:
            # Back along the chain to the prior, then forward through every frame in between.
            features, anatomy = chain[length - 1].features, chain[length - 1].anatomy
            for target in range(prior + 1, last + 1):
                hop = tracker(features, frames[target], anatomy, maps[target])
                features, anatomy = hop.features, hop.anatomy
            cycles[f'long_{length}'] = hop.grid

    return cycles, first_hops


def cycle_loss(start_grid: Tensor, end_grid: Tensor, patch_size: int, huber_delta: float | None = None) -> Tensor:
    """How far the patch cells returned from where they started, in units of the patch size. Returns [B].

    Comparing whole grids penalizes position and rotation together. Dividing by the patch size (not by the frame
    size) keeps the loss isotropic on a non-square frame. Without huber_delta the distance is squared; with it, Huber
    keeps a constant gradient above huber_delta (0.25 is a quarter patch), which bounds patches with no true match.
    """
    distance = (end_grid - start_grid).norm(dim=-1) / patch_size
    if huber_delta is None:
        return distance.square().mean(dim=1)

    return F.huber_loss(distance, torch.zeros_like(distance), delta=huber_delta, reduction='none').mean(dim=1)


def similarity_loss(query: Tensor, matched: Tensor) -> Tensor:
    """One minus the mean cosine similarity between aligned cells. [B, C, h, w] x [B, C, h, w] -> [B]."""
    return 1 - (query * matched).sum(dim=1).mean(dim=(1, 2))
