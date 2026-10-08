import torch
import torch.nn.functional as F
from torch import Tensor


def sample_patches(frames: Tensor, num_patches: int, patch_size: int, stride: int, min_tissue_fraction: float,
                   threshold: float, generator: torch.Generator | None = None) -> Tensor:
    """Pick patch positions uniformly, with replacement, among those whose window is mostly tissue.

    frames [B, 1, H, W] in [0, 1]; tissue is where they exceed threshold. Returns the top-left pixel (x, y) of each
    patch, [B, num_patches, 2], on the stride grid, so a patch's cells line up with the frame's cells. A frame with no
    position at min_tissue_fraction (a very small breast) uses its most tissue-filled positions. generator is a CPU
    generator wherever the frames live; validation passes a seeded one so every epoch scores the same patches.

    TODO(smarter sampling): weight positions by texture, keep the patches of one frame apart, avoid a band along the
    chest wall, where tissue is most often missing from the other exam.
    """
    fraction = F.avg_pool2d((frames > threshold).float(), patch_size, stride=stride)  # one value per position
    columns = fraction.shape[-1]
    fraction = fraction.flatten(1)

    valid = fraction >= min_tissue_fraction
    best = fraction == fraction.amax(dim=1, keepdim=True)
    weights = torch.where(valid.any(dim=1, keepdim=True), valid, best).float()
    index = torch.multinomial(weights.cpu(), num_patches, replacement=True, generator=generator).to(frames.device)

    return torch.stack([index % columns, index // columns], dim=-1) * stride
