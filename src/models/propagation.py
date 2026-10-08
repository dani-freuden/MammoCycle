"""Inference by label propagation, the paper's test-time method, on L2-normalized feature maps of two frames."""
import torch
from torch import Tensor

from .tracker import cell_centers


def box_coverage(boxes: Tensor, height: int, width: int, stride: int) -> Tensor:
    """The fraction of each feature cell inside boxes [B, 4] (x1, y1, x2, y2 pixels). Returns [B, height, width]."""
    def coverage(low: Tensor, high: Tensor, cells: int) -> Tensor:
        edges = torch.arange(cells + 1, device=boxes.device, dtype=torch.float32) * stride
        overlap = torch.minimum(high[:, None], edges[1:]) - torch.maximum(low[:, None], edges[:-1])
        return overlap.clamp(0, stride) / stride

    rows, columns = coverage(boxes[:, 1], boxes[:, 3], height), coverage(boxes[:, 0], boxes[:, 2], width)

    return rows[:, :, None] * columns[:, None, :]


def propagate_labels(source: Tensor, target: Tensor, labels: Tensor, temperature: float, stride: int,
                     top_k: int = 5, radius: float | None = None, chunk: int = 2048) -> Tensor:
    """Give every target cell the labels of its top_k most similar source cells.

    source, target [B, C, h, w]; labels [B, K, h, w] on the source grid -> [B, K, h, w] on the target grid. The top_k
    similarities become weights through a softmax. radius (pixels) limits each target cell to source cells within that
    distance of its own position; None searches the whole frame, as the paper does. chunk bounds memory: the full
    affinity between two 128 x 80 grids has 100 million entries per pair.
    """
    height, width = target.shape[-2:]
    source, target, labels = source.flatten(2), target.flatten(2), labels.flatten(2)
    centers = cell_centers(height, width, stride, target.device)

    outputs = []
    for start in range(0, target.shape[-1], chunk):
        logits = torch.einsum('bct,bcs->bts', target[:, :, start:start + chunk], source) / temperature
        if radius is not None:
            too_far = torch.cdist(centers[start:start + chunk], centers) > radius
            logits = logits.masked_fill(too_far, float('-inf'))

        values, index = logits.topk(top_k, dim=-1)  # [B, n, k]
        picked = torch.gather(labels[:, :, None].expand(-1, -1, index.shape[1], -1), 3,
                              index[:, None].expand(-1, labels.shape[1], -1, -1))  # [B, K, n, k]
        outputs.append((picked * values.softmax(dim=-1)[:, None]).sum(dim=-1))

    return torch.cat(outputs, dim=-1).unflatten(-1, (height, width))


def locate(lesion: Tensor, stride: int, threshold: float = 0.5) -> dict[str, Tensor]:
    """Turn a soft lesion map [B, h, w] into a location.

    center  [B, 2] weighted centre of the cells above the threshold, or the peak cell when none is
    box     [B, 4] bounding box of the cells above the threshold, or the peak cell when none is
    score   [B]    peak value of the map
    found   [B]    whether any cell is above the threshold

    TODO(connected region): keep only the region around the peak, so one far false positive does not stretch the box.
    """
    height, width = lesion.shape[-2:]
    centers = cell_centers(height, width, stride, lesion.device)
    flat = lesion.flatten(1)
    score, peak = flat.max(dim=1)

    above = (flat > threshold) | (torch.arange(flat.shape[1], device=lesion.device) == peak[:, None])
    weights = flat * above
    center = (weights @ centers) / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

    low = torch.where(above[..., None], centers[None] - stride / 2, torch.inf).amin(dim=1)
    high = torch.where(above[..., None], centers[None] + stride / 2, -torch.inf).amax(dim=1)

    return {'center': center, 'box': torch.cat([low, high], dim=1), 'score': score, 'found': score > threshold}
