import torch

from src.models.patch_sampler import sample_patches


def half_breast(batch=3, height=512, width=256):
    """Tissue on the left 60% of the frame."""
    frames = torch.zeros(batch, 1, height, width)
    frames[..., :int(0.6 * width)] = 0.5
    return frames


def tissue_fraction(frames, top_left, size):
    """The tissue fraction of every sampled window, [B * P]."""
    return torch.stack([(frames[b, 0, y:y + size, x:x + size] > 0).float().mean()
                        for b in range(len(frames)) for x, y in top_left[b].tolist()])


def test_sampled_patches_are_mostly_tissue_and_on_the_stride_grid():
    frames = half_breast()
    top_left = sample_patches(frames, 50, patch_size=64, stride=8, min_tissue_fraction=0.75, threshold=0.01,
                              generator=torch.Generator().manual_seed(0))

    assert top_left.shape == (3, 50, 2)
    assert (top_left % 8 == 0).all()
    assert top_left[..., 0].min() >= 0 and top_left[..., 0].max() <= 256 - 64
    assert top_left[..., 1].min() >= 0 and top_left[..., 1].max() <= 512 - 64
    assert (tissue_fraction(frames, top_left, 64) >= 0.75).all()
    assert top_left[..., 0].float().std() > 0 and top_left[..., 1].float().std() > 0  # positions vary


def test_same_generator_gives_the_same_patches():
    draw = lambda: sample_patches(half_breast(), 5, 64, 8, 0.75, 0.01, generator=torch.Generator().manual_seed(3))
    assert torch.equal(draw(), draw())


def test_frame_without_valid_position_falls_back_to_its_best_one():
    frames = torch.zeros(1, 1, 256, 256)
    frames[..., 100:130, 100:130] = 0.5  # far too small for 75% of a 64 pixel patch
    top_left = sample_patches(frames, 4, patch_size=64, stride=8, min_tissue_fraction=0.75, threshold=0.01)

    assert torch.allclose(tissue_fraction(frames, top_left, 64), torch.full((4,), 900 / 4096))
