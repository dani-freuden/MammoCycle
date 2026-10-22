"""Checks for walk.py: the walk losses, the training windows and the score map's geometry. From the repo root:
    python -m experimental.mammo_walk.test_walk
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.anatomy import anatomical_coords, get_centroid

from .walk import (ANAT_SIGMA, EVAL_STRIDE, GRID, WINDOW, WalkPairs, anatomical_match, score_map, walk_losses)

from b5_prior_fusion import decode  # on sys.path through walk


torch.manual_seed(0)
rng = np.random.default_rng(0)
n = GRID * GRID
valid = torch.ones(1, n, dtype=torch.bool)
anatomy = torch.rand(1, n, 2)

# ---- 1. losses ------------------------------------------------------------------------------------------------------
# Distinct embeddings close every cycle, whatever the order of the prior nodes, and each step lands on its own
# anatomical twin; equal embeddings walk uniformly and return with probability 1 / n.
eye, order = torch.eye(n)[None], torch.randperm(n)
cycle, anat, _ = walk_losses(eye, eye[:, order], valid, valid, anatomy, anatomy[:, order], temperature=0.01)
assert cycle.item() < 1e-3 and anat.item() < 1e-3, (cycle, anat)
flat = F.normalize(torch.ones(1, n, 8), dim=-1)
cycle, _, _ = walk_losses(flat, flat, valid, valid, anatomy, anatomy, temperature=0.07)
assert abs(cycle.item() - np.log(n)) < 1e-4, cycle
# An offset of ANAT_SIGMA in u and v costs 1 per step.
_, anat, _ = walk_losses(eye, eye[:, order], valid, valid, anatomy, anatomy[:, order] + ANAT_SIGMA, temperature=0.01)
assert abs(anat.item() - 2) < 1e-3, anat
# Nodes outside the breast get no probability, even with edge dropout.
prior_valid = valid.clone()
prior_valid[0, :5] = False
_, _, logits = walk_losses(eye, eye, valid, prior_valid, anatomy, anatomy, temperature=0.07, edge_dropout=0.5)
assert logits.softmax(dim=-1)[0, :, :5].max() == 0
print('losses: ok')

# ---- 2. training windows ----------------------------------------------------------------------------------------------
# A half-disc breast with its chest wall on the right, and smooth texture.
height, width, radius = 1400, 900, 800
y, x = np.mgrid[:height, :width]
mask = (x - width) ** 2 + (y - height / 2) ** 2 < radius ** 2
texture = cv2.GaussianBlur(rng.standard_normal((height, width)).astype(np.float32), (0, 0), 6)
pixels = (np.clip(128 + 400 * texture, 1, 255) * mask).astype(np.uint8)
centroid = np.array(get_centroid(mask), dtype=float)
image = (pixels, mask, centroid, np.array([width - radius, height / 2]))
images = {'current': image, 'prior': image}

p_cur = np.array([600.0, 650.0])
assert np.allclose(anatomical_match(p_cur, image, image), p_cur, atol=1.5)
item = WalkPairs([{'current': 'current', 'prior': 'prior', 'p_cur': p_cur, 'p_anat': p_cur, 'p_gt': p_cur}],
                 images.__getitem__)[0]
assert item['current'].shape == item['prior'].shape == (1, WINDOW, WINDOW)
assert item['current_valid'].shape == (n,) and item['current_anatomy'].shape == (n, 2)
# The centre node sits on p_cur whatever the rotation and scale; the prior window moves only by the jitter (2 mm).
assert np.allclose(item['current_anatomy'][n // 2], anatomical_coords(p_cur[None], *image[1:])[0], atol=1e-4)
assert np.abs(item['target'].numpy() - (GRID - 1) / 2).max() < 1, item['target']
print('windows: ok')

# ---- 3. score map geometry ----------------------------------------------------------------------------------------------
# The prior is the current image shifted; with the pixels themselves as embeddings, the score map's peak must map back
# to where p_cur moved.
shift = np.array([37.0, -21.0])
prior = cv2.warpAffine(pixels, np.array([[1, 0, shift[0]], [0, 1, shift[1]]]), (width, height))
case = {'img_cur': pixels, 'img_prior': prior, 'p_cur': p_cur, 'p_anat': p_cur + rng.uniform(-40, 40, 2)}
pixel_embedding = lambda patches: F.normalize(F.avg_pool2d(patches, 4).flatten(1), dim=-1)
smap = score_map(pixel_embedding, case, temperature=1e-3, device='cpu')
point, _ = decode(smap.score, smap.M_grid)
error = np.linalg.norm(point - (p_cur + shift))
print(f'score map: peak {error:.2f} px from the truth (candidates every {EVAL_STRIDE} px)')
assert error < EVAL_STRIDE / 2
