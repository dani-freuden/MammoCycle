"""How much harder real pairs are than the synthetic ones, for the phase-1 matcher and for plain NCC.

Both are scored by appearance only, on the crops the matcher sees, and decoded over its output grid (argmax, no prior,
no breast mask):
- real: the annotated train-split pairs, which phase 1 does not train on. The template is on the current box centre
  and the search window on the anatomical point, the anatomy box centre of outputs/b5_prior_fusion/predictions.csv,
  as in train_selfsup's evaluation.
- synthetic: train_selfsup's frozen synthetic dev set (same images, seed and size), from the current DeformConfig.

The NCC template is the central 1.5 x box (12 mm up to the template crop's 51 mm) of the template crop, and both crops
are DoG-filtered as in B5; its argmax is taken over the positions the matcher's grid covers. Errors are in mm at the
nominal spacing. Re-run after every DeformConfig change: NCC needs no training, so it shows how the synthetic task's
difficulty moves.

Usage (from the repo root):
    python -m experimental.mixturemodel.lesion_matching.measure_gap --checkpoint outputs/stepD_selfsup/best.pt
    python -m experimental.mixturemodel.lesion_matching.measure_gap --checkpoint outputs/stepD_selfsup/best.pt \
        --synthetic-only
"""
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import pandas as pd
import torch
import typer
from tqdm import tqdm
from typer import Option

from baselines.anatomical_coordinates.predict import BOX, DATA_DIR
from src.enums import Split

from .predict import centre
from .train_selfsup import ALL_IMAGES_DIR, box_sizes_mm, load_record, unannotated_images
from .train_stepD import CFG, SCALE, load_image

sys.path.insert(0, str(Path(__file__).parent))  # the siblings import each other as top-level modules
from b5_prior_fusion import dog_filter, print_table, summarize, template_side_px
from geometry import apply_affine, rescale_points
from selfsup import SingleImageSSLDataset, freeze, grid_error_mm
from stepD_matcher import PairDataset, SiameseMatcher, grid_affine


def real_cases(predictions: Path, workers: int) -> list:
    """The annotated train-split pairs as PairDataset cases, in working pixels."""
    pairs = pd.read_csv(predictions)
    pairs = pairs[pairs.split_group == str(Split.TRAIN)]
    paths = sorted(set(pairs.current_path) | set(pairs.prior_path))
    with Pool(workers) as pool:
        images = dict(zip(paths, tqdm(pool.imap(load_image, paths), total=len(paths), desc='images', unit='image')))
    point = lambda box: rescale_points(centre(box), SCALE)
    boxes = (pairs[[f'{exam}_{c}' for c in BOX]].values for exam in ('current', 'prior', 'anat'))

    return [{'img_cur': images[current_path][0], 'img_prior': images[prior_path][0], 'p_cur': point(current),
             'box_wh': SCALE * (current[[3, 2]] - current[[1, 0]]), 'p_gt': point(prior), 'p_anat': point(anat)}
            for current_path, prior_path, current, prior, anat in zip(pairs.current_path, pairs.prior_path, *boxes)]


def ncc_point(sample: dict, side_px: int | None = None) -> np.ndarray:
    """NCC's appearance-only point (x, y) in the search crop of one sample (template with its box mask, search). The
    template is the central side_px of the template crop, by default 1.5 x the box (12 mm up to the whole crop)."""
    template, box = sample['template'][0].numpy(), sample['template'][1].numpy()
    if side_px is None:
        ys, xs = np.nonzero(box)
        side_px = min(template_side_px((np.ptp(xs) + 1, np.ptp(ys) + 1), CFG.spacing_mm), CFG.template_px)
    side = side_px - side_px % 2  # same parity as the template crop, so the patch centre is the lesion centre
    start = (CFG.template_px - side) // 2
    patch = dog_filter(template, CFG.spacing_mm)[start:start + side, start:start + side]
    search = dog_filter(sample['search'][0].numpy(), CFG.spacing_mm)
    score = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)  # position (j, i) <-> centre + (side - 1) / 2

    # only the positions the matcher's grid covers
    low, high = apply_affine(grid_affine(CFG), [[0, 0], [CFG.out_cells - 1] * 2])[:, 0]
    centres = np.arange(score.shape[0]) + (side - 1) / 2
    covered = (centres >= low - 0.5) & (centres <= high + 0.5)
    score = np.where(covered[:, None] & covered[None, :], np.nan_to_num(score, nan=-1.0), -np.inf)
    i, j = np.unravel_index(np.argmax(score), score.shape)

    return np.array([j, i]) + (side - 1) / 2


def truth(sample: dict) -> np.ndarray:
    """The true lesion centre (x, y) in the search crop of one sample."""
    return apply_affine(grid_affine(CFG), sample['gt_grid'].numpy()[None])[0]


def ncc_error_mm(sample: dict, side_px: int | None = None) -> float:
    """The appearance-only NCC error of one sample, in mm, with ncc_point's template."""
    return float(np.linalg.norm(ncc_point(sample, side_px) - truth(sample)) * CFG.spacing_mm)


def measure_real(model: SiameseMatcher, cases: list, device: str):
    """Print the window centre's, the matcher's and NCC's errors on the real pairs."""
    real = PairDataset(cases, CFG, train=False)
    anatomy = [np.linalg.norm(c['p_gt'] - c['p_anat']) * CFG.spacing_mm for c in cases]
    ncc, outside = [], 0
    # by index: PairDataset wraps its index around, so iterating it directly never ends
    for sample in (real[i] for i in tqdm(range(len(real)), desc='NCC real')):
        ncc.append(ncc_error_mm(sample))
        outside += ((sample['gt_grid'] < 0) | (sample['gt_grid'] > CFG.out_cells - 1)).any().item()
    print(f'\nREAL: {len(real)} annotated train pairs, window on the anatomical point '
          f'({outside} with the true centre outside the grid)')
    print_table([('window centre (anatomy)', summarize(anatomy)),
                 ('matcher, appearance only', summarize(grid_error_mm(model, real, CFG, device))),
                 ('NCC, appearance only', summarize(ncc))])


def main(
    checkpoint: Annotated[Path, Option(help='best.pt of train_selfsup')],
    data_dir: Annotated[Path, Option(help='The annotated set: output folder of scripts/misc/prepare_embed_dataset.py')]
        = Path(DATA_DIR),
    all_images_dir: Annotated[Path, Option(help='Output folder of prepare_embed_dataset.py --all-images')]
        = ALL_IMAGES_DIR,
    predictions: Annotated[Path, Option(help='predictions.csv with every pair and its anatomy box (anat_*)')]
        = Path('./outputs/b5_prior_fusion/predictions.csv'),
    samples: Annotated[int, Option(help='Frozen synthetic dev pairs, as train_selfsup --val-samples')] = 500,
    synthetic_only: Annotated[bool, Option(help='Skip the real pairs, whose numbers DeformConfig does not change')]
        = False,
    gpu: Annotated[int, Option(help='CUDA device')] = 0,
    workers: Annotated[int, Option(help='Parallel image loading')] = 16,
):
    device = f'cuda:{gpu}'
    model = SiameseMatcher(CFG)
    model.load_state_dict(torch.load(checkpoint)['model'])
    paths = unannotated_images(all_images_dir, data_dir, Split.DEV)
    synthetic = freeze(SingleImageSSLDataset(len(paths), partial(load_record, paths), CFG,
                                             box_sizes_mm=box_sizes_mm(data_dir), fixed_seed=123), samples)
    if not synthetic_only:
        measure_real(model, real_cases(predictions, workers), device)
    print(f'\nSYNTHETIC: {len(synthetic)} frozen dev pairs, current DeformConfig')
    print_table([('matcher, appearance only', summarize(grid_error_mm(model, synthetic, CFG, device))),
                 ('NCC, appearance only', summarize([ncc_error_mm(s) for s in tqdm(synthetic, desc='NCC synth')]))])


if __name__ == '__main__':
    typer.run(main)
