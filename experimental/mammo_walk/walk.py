"""Contrastive random walk (Jabri, Owens and Efros, arXiv:2006.14613) between the current and the prior CC exam of a
breast.

Nodes: a GRID x GRID grid of PATCH px patches, NODE_STRIDE px apart, in a WINDOW px window of each exam at SPACING_MM.
Each patch is embedded on its own (PatchEmbedding), so no embedding knows where it sits in its window. With feature
maps of whole windows, position alone could close a two-exam palindrome. The current window is centred on a point p_cur,
the prior one on its anatomical match p_anat.

Walk (walk_losses): A_ab = softmax(f_a f_b^T / T) over the prior nodes, A_ba the same back. The cycle loss is the
cross-entropy of A_ab A_ba against the identity, over the current nodes inside the breast. Nodes outside the breast are
never stepped on, and in training a random EDGE_DROPOUT share of the edges is dropped, as in the paper. The anatomical
loss is each step's expected |a(j) - a(i)|^2 in anatomical coordinates (src.utils.anatomy, mirrored boundary), divided
by 2 ANAT_SIGMA^2.

Geometry follows lesion_matching/geometry.py: points are (x, y) with pixel centres at integers, and a crop is a 2x3
matrix crop px -> image px. Grid cell (col, row) of patches stride px apart has its patch centred at crop px
stride * (col, row) + (PATCH - 1) / 2 (grid_affine).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset

from src.data.dataset import breast_mask
from src.models.encoder import ResNetEncoder, load_mirai_trunk
from src.utils.anatomy import anatomical_coords, boundary_radii, get_centroid, get_nipple_jas_cc, image_coords

sys.path.insert(0, str(Path(__file__).parents[1] / 'mixturemodel' / 'lesion_matching'))  # top-level sibling imports
from b5_prior_fusion import ScoreMap
from geometry import apply_affine, compose, crop_affine, invert_affine, resample_to_spacing, warp_crop


MIRAI = Path('/data/rbg/users/dani_fr/projects/mirai-shira/runs/shifa-8bit-2-steps4000/best.ckpt')
MIRAI_MEAN, MIRAI_STD = 0.11866, 0.18850  # src.config.EncoderConfig: Mirai's normalization on the [0, 1] scale

PNG_SPACING_MM = 0.12  # lesion_matching.predict.SPACING_MM, the nominal mm per PNG pixel
SPACING_MM = 0.2
PATCH = 256  # 51.2 mm of context per node
NODE_STRIDE = 64  # 12.8 mm between nodes
GRID = 7
WINDOW = PATCH + (GRID - 1) * NODE_STRIDE  # 640 px, 128 mm
EVAL_STRIDE = 8  # 1.6 mm between the candidate positions of score_map
EVAL_GRID = 57  # +-44.8 mm around p_anat, as Step D

ANAT_SIGMA = 0.09  # std of the mirrored anatomical residual of the annotated pairs
EDGE_DROPOUT = 0.1
TARGET_SIGMA_NODES = 0.5  # width of the lesion's soft target in the fine-tuning CE
ROTATION_DEGREES, SCALE_RANGE, JITTER_MM = 10.0, 0.1, 2.0  # window augmentation, as Step D


def load_image(path: str) -> tuple:
    """A PNG at SPACING_MM (uint8), and its breast mask, centroid and jas_cc nipple (x, y)."""
    image, _ = resample_to_spacing(np.asarray(Image.open(path)), PNG_SPACING_MM, SPACING_MM)
    mask = breast_mask(image)
    centroid = np.array(get_centroid(mask), dtype=float)

    return image, mask, centroid, np.array(get_nipple_jas_cc(mask, centroid), dtype=float)


def anatomical_match(point, source: tuple, target: tuple) -> np.ndarray:
    """The point (x, y) of the target image at the anatomical coordinates of point in the source. Both are load_image
    tuples."""
    _, mask, centroid, nipple = source
    a = anatomical_coords(np.array([point]), mask, centroid, nipple)
    _, mask, centroid, nipple = target

    return image_coords(a, centroid, nipple, boundary_radii(mask, centroid, 720))[0]


def grid_affine(stride: int) -> np.ndarray:
    """2x3: grid cell (col, row) of patches stride px apart -> crop pixel of its patch's centre."""
    centre = (PATCH - 1) / 2
    return np.array([[stride, 0, centre], [0, stride, centre]], dtype=float)


def grid_cells(count: int) -> np.ndarray:
    """Every cell (col, row) of a count x count grid, row-major as the embeddings. [count^2, 2]"""
    cols, rows = np.meshgrid(np.arange(count), np.arange(count))
    return np.stack([cols.ravel(), rows.ravel()], axis=1)


class PatchEmbedding(nn.Module):
    """Patches [N, 1, PATCH, PATCH] in [0, 1] -> L2-normalized embeddings [N, dim]: the Mirai trunk
    (src.models.encoder, BatchNorm frozen) averaged over the patch, then a linear head."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.encoder = ResNetEncoder(load_mirai_trunk(MIRAI), MIRAI_MEAN, MIRAI_STD)
        self.head = nn.Linear(ResNetEncoder.out_channels, dim)

    def forward(self, patches: Tensor) -> Tensor:
        return F.normalize(self.head(self.encoder(patches).mean(dim=(2, 3))), dim=-1)


def embed_patches(model, windows: Tensor, stride: int, chunk: int = 512) -> Tensor:
    """Every PATCH x PATCH patch, stride px apart, of windows [B, 1, H, W], embedded on its own: [B, cells, dim] with
    the cells row-major."""
    patches = windows.unfold(2, PATCH, stride).unfold(3, PATCH, stride)  # [B, 1, rows, cols, PATCH, PATCH]
    batch, cells = patches.shape[0], patches.shape[2] * patches.shape[3]
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(batch * cells, 1, PATCH, PATCH)

    return torch.cat([model(part) for part in patches.split(chunk)]).view(batch, cells, -1)


def window(image: tuple, M: np.ndarray) -> tuple:
    """The WINDOW x WINDOW crop M of a load_image tuple: pixels [1, WINDOW, WINDOW] in [0, 1], whether each node's
    centre is in the breast [GRID^2] and the nodes' anatomical coordinates [GRID^2, 2]."""
    pixels, mask, centroid, nipple = image
    nodes = apply_affine(compose(M, grid_affine(NODE_STRIDE)), grid_cells(GRID))
    x, y = np.round(nodes).astype(int).T
    inside = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    valid = np.zeros(len(nodes), dtype=bool)
    valid[inside] = mask[y[inside], x[inside]]

    return (torch.from_numpy(warp_crop(pixels, M, (WINDOW, WINDOW)).astype(np.float32) / 255)[None],
            torch.from_numpy(valid),
            torch.from_numpy(anatomical_coords(nodes, mask, centroid, nipple)).float())


class WalkPairs(Dataset):
    """Training windows of exam pairs.

    pairs: dicts with the 'current' and 'prior' image paths and, for annotated pairs, 'p_cur', 'p_anat' and 'p_gt' in
    working pixels. load(path) gives load_image's tuple. An unannotated pair gets a random p_cur in the current breast,
    and its anatomical match as p_anat. The current window is centred on p_cur and the prior one on p_anat plus a
    Gaussian JITTER_MM, each with its own rotation and scale.

    Items: 'current' and 'prior' windows [1, WINDOW, WINDOW]; '*_valid' [GRID^2] and '*_anatomy' [GRID^2, 2] (see
    window); for annotated pairs 'target', p_gt as a cell (col, row) of the prior's node grid.
    """

    def __init__(self, pairs: list, load):
        self.pairs, self.load = pairs, load

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng()
        pair = self.pairs[index]
        current, prior = self.load(pair['current']), self.load(pair['prior'])
        if 'p_cur' in pair:
            p_cur, p_anat = pair['p_cur'], pair['p_anat']
        else:
            ys, xs = np.nonzero(current[1])
            i = rng.integers(len(xs))
            p_cur = np.array([xs[i], ys[i]], dtype=float)
            p_anat = anatomical_match(p_cur, current, prior)

        item, matrices = {}, {}
        for name, image, centre in (('current', current, p_cur),
                                    ('prior', prior, p_anat + rng.normal(0, JITTER_MM / SPACING_MM, 2))):
            matrices[name] = crop_affine(centre, (WINDOW, WINDOW), 1 + rng.uniform(-SCALE_RANGE, SCALE_RANGE),
                                         rng.uniform(-ROTATION_DEGREES, ROTATION_DEGREES))
            item[name], item[f'{name}_valid'], item[f'{name}_anatomy'] = window(image, matrices[name])
        if 'p_gt' in pair:
            to_grid = invert_affine(compose(matrices['prior'], grid_affine(NODE_STRIDE)))
            item['target'] = torch.from_numpy(apply_affine(to_grid, pair['p_gt'])[0]).float()

        return item


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean over the last dimension of the entries where mask is true. [B, N] -> [B]"""
    return (values * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)


def step_logits(similarity: Tensor, valid_columns: Tensor, edge_dropout: float) -> Tensor:
    """Logits of one step [B, N, M] with the columns outside the breast and a random edge_dropout share of the edges at
    -1e20, as CRW's stoch_mat. Finite, so a fully dropped row is uniform rather than NaN."""
    drop = ~valid_columns[:, None, :]
    if edge_dropout > 0:
        drop = drop | (torch.rand_like(similarity) < edge_dropout)

    return similarity.masked_fill(drop, -1e20)


def walk_losses(current: Tensor, prior: Tensor, current_valid: Tensor, prior_valid: Tensor, current_anatomy: Tensor,
                prior_anatomy: Tensor, temperature: float, edge_dropout: float = 0.0) -> tuple:
    """The palindrome current -> prior -> current, in float32. Embeddings [B, N, D], valid [B, N] (node in the breast),
    anatomy [B, N, 2].

    Returns the cycle loss [B], the anatomical loss [B], and the first step's logits [B, N, N] (current rows, prior
    columns) without edge dropout, for a supervised loss on a row.
    """
    similarity = current @ prior.transpose(1, 2) / temperature
    forward = step_logits(similarity, prior_valid, edge_dropout).softmax(dim=-1)
    backward = step_logits(similarity.transpose(1, 2), current_valid, edge_dropout).softmax(dim=-1)
    returned = (forward @ backward).diagonal(dim1=1, dim2=2)

    distance = (current_anatomy[:, :, None] - prior_anatomy[:, None]).square().sum(dim=-1) / (2 * ANAT_SIGMA ** 2)
    anatomy = (masked_mean((forward * distance).sum(dim=-1), current_valid)
               + masked_mean((backward * distance.transpose(1, 2)).sum(dim=-1), prior_valid))

    return (masked_mean(-returned.clamp_min(1e-20).log(), current_valid), anatomy,
            step_logits(similarity, prior_valid, 0.0))


@torch.no_grad()
def score_map(model, case: dict, temperature: float, device) -> ScoreMap:
    """One step of the walk from the patch on p_cur to the EVAL_GRID^2 patches, EVAL_STRIDE px apart, of a window on
    p_anat in the prior: log p(lesion centre at each), as a ScoreMap for b5_prior_fusion. case is one of
    train_stepD.matcher_cases."""
    size = PATCH + (EVAL_GRID - 1) * EVAL_STRIDE
    M = crop_affine(case['p_anat'], (size, size))
    lesion = warp_crop(case['img_cur'], crop_affine(case['p_cur'], (PATCH, PATCH)), (PATCH, PATCH))
    search = warp_crop(case['img_prior'], M, (size, size))
    to_tensor = lambda pixels: torch.from_numpy(pixels.astype(np.float32) / 255)[None, None].to(device)

    with torch.autocast('cuda', enabled=str(device).startswith('cuda')):
        query = model(to_tensor(lesion)).float()[0]
        candidates = embed_patches(model, to_tensor(search), EVAL_STRIDE)[0].float()
    score = (candidates @ query / temperature).log_softmax(dim=0)

    return ScoreMap(score=score.view(EVAL_GRID, EVAL_GRID).double().cpu().numpy(),
                    M_grid=compose(M, grid_affine(EVAL_STRIDE)))
