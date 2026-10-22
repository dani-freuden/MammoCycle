"""Why the phase-1 matcher solves synthetic pairs but fails on real ones: two checks, no training.

1. Lesion appearance. Synthetic pairs made from the annotated current images of the train-split pairs, around a
   random breast point (as in training) or around the real lesion box, with the current DeformConfig and with knobs 2-4
   (ablate_deform). If the matcher drops on the lesion pairs, it is the lesions themselves it has not learned, not the
   changes between exams.
2. Real failures. On the real train pairs, scored as measure_gap.py scores them: how often the matcher and NCC hit
   (within HIT_MM), both, one or neither, by lesion kind, box size and vendor pair (current / prior; the GE images are
   processed differently, with a grey background, from the Hologic ones). A contact sheet shows pairs where NCC hits and
   the matcher misses by more than MISS_MM: the template with its box, and the search window with the truth (green),
   NCC's point (yellow) and the matcher's (red), once plain and once under the matcher's probability map.

Writes real_pairs.csv (each pair's errors, kind and box size) and ncc_hits_matcher_misses.png to output_dir.

Usage (from the repo root):
    python -m experimental.mixturemodel.lesion_matching.diagnose_gap --checkpoint outputs/stepD_selfsup/best.pt
"""
import sys
from functools import partial
from pathlib import Path
from typing import Annotated

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import typer
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader
from tqdm import tqdm
from typer import Option

from baselines.anatomical_coordinates.predict import BOX, DATA_DIR
from src.enums import Split

from .ablate_deform import SETTINGS
from .measure_gap import ncc_error_mm, ncc_point, real_cases, truth
from .predict import SPACING_MM, centre
from .train_selfsup import box_sizes_mm
from .train_stepD import CFG, SCALE, load_image

sys.path.insert(0, str(Path(__file__).parent))  # the siblings import each other as top-level modules
from b5_prior_fusion import print_table, summarize
from geometry import apply_affine, rescale_points
from selfsup import DeformConfig, SingleImageSSLDataset, grid_error_mm
from stepD_matcher import PairDataset, SiameseMatcher, grid_affine


HIT_MM = 5.0
MISS_MM = 15.0
SHEET_PAIRS = 12
SEED = 0


def train_pairs(predictions: Path) -> pd.DataFrame:
    """The train-split pairs of predictions.csv, in the order measure_gap.real_cases gives their cases."""
    pairs = pd.read_csv(predictions)
    return pairs[pairs.split_group == str(Split.TRAIN)]


def load_annotated(paths: list[str], boxes: dict, index: int) -> dict:
    """SingleImageSSLDataset's record of an annotated current image: image, breast mask and lesion boxes."""
    image, mask = load_image(paths[index])
    return {'img': image, 'mask': mask, 'boxes': boxes[paths[index]]}


def lesion_boxes(pairs: pd.DataFrame) -> tuple[list[str], dict]:
    """The pairs' current images and, per image, its distinct lesion boxes (cx, cy, w, h) in working pixels."""
    boxes = {}
    for path, box in zip(pairs.current_path, pairs[[f'current_{c}' for c in BOX]].values):
        boxes.setdefault(path, set()).add((*rescale_points(centre(box), SCALE), *(SCALE * (box[[3, 2]] - box[[1, 0]]))))
    return sorted(boxes), {path: sorted(b) for path, b in boxes.items()}


def lesion_check(model: SiameseMatcher, pairs: pd.DataFrame, data_dir: Path, samples: int, device: str,
                 workers: int):
    """Check 1: synthetic pairs from the annotated current images, around a random point or the real lesion."""
    paths, boxes = lesion_boxes(pairs)
    print(f'{len(paths)} annotated current images, {sum(map(len, boxes.values()))} distinct lesion boxes')

    sizes, rows = box_sizes_mm(data_dir), []
    for point, p_real_box in (('random point', 0.0), ('real lesion', 1.0)):
        for setting in ('current defaults (knob 1)', '+ 2, 3 and 4'):
            dataset = SingleImageSSLDataset(len(paths), partial(load_annotated, paths, boxes), CFG,
                                            DeformConfig(p_real_box=p_real_box, **SETTINGS[setting]), epoch_len=samples,
                                            box_sizes_mm=sizes, fixed_seed=123)
            synthetic = list(DataLoader(dataset, batch_size=None, num_workers=workers))
            name = f'{point}, {setting}'
            rows += [(f'{name}: matcher', summarize(grid_error_mm(model, synthetic, CFG, device))),
                     (f'{name}: NCC', summarize([ncc_error_mm(s) for s in synthetic]))]
            print(f'{name}: done', flush=True)
    print(f'\nCHECK 1: synthetic pairs from the annotated current images ({samples} each)')
    print_table(rows)


@torch.no_grad()
def matcher_maps(model: SiameseMatcher, dataset, device: str) -> np.ndarray:
    """The matcher's probability map over its grid for every sample, (N, H, W)."""
    model.eval().to(device)
    maps = []
    for batch in tqdm(DataLoader(dataset, 16, num_workers=4), desc='matcher', unit='batch'):
        logits = model(batch['template'].to(device), batch['search'].to(device)).float()
        maps.append(torch.softmax(logits.flatten(1), 1).view_as(logits).cpu().numpy())
    return np.concatenate(maps)


def show(ax, image, title):
    ax.imshow(image, cmap='gray', vmin=np.percentile(image, 1), vmax=np.percentile(image, 99))
    ax.set_title(title, fontsize=8)
    ax.axis('off')


def contact_sheet(real: PairDataset, rows: pd.DataFrame, maps: np.ndarray, path: Path):
    """One row per pair: the template with its box; the search window with the truth, NCC's and the matcher's
    points; the same under the matcher's probability map."""
    low, high = apply_affine(grid_affine(CFG), [[0, 0], [CFG.out_cells - 1] * 2])[:, 0]
    extent = (low - CFG.stride / 2, high + CFG.stride / 2, high + CFG.stride / 2, low - CFG.stride / 2)
    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 3.4 * len(rows)), layout='constrained', squeeze=False)
    for ax, (index, row) in zip(axes, rows.iterrows()):
        sample = real[index]
        ys, xs = np.nonzero(sample['template'][1].numpy())
        show(ax[0], sample['template'][0].numpy(), f'template, {row.kind}, box {row.box_mm:.0f} mm, {row.vendors}')
        ax[0].add_patch(Rectangle((xs.min(), ys.min()), np.ptp(xs), np.ptp(ys), fill=False, ec='#ffd60a', lw=1))
        for a, title in ((ax[1], f'NCC {row.ncc_mm:.1f} mm, matcher {row.matcher_mm:.1f} mm'),
                         (ax[2], 'matcher probability')):
            show(a, sample['search'][0].numpy(), title)
            for point, color in ((row[['true_x', 'true_y']], '#34c759'), (row[['ncc_x', 'ncc_y']], '#ffd60a'),
                                 (row[['matcher_x', 'matcher_y']], '#ff3b30')):
                a.scatter(*point, s=60, marker='+', color=color, linewidths=1.5, zorder=3)
        ax[2].imshow(maps[index], extent=extent, cmap='magma', alpha=0.55)
    fig.suptitle('NCC within 5 mm, matcher off by more than 15 mm. Green: truth, yellow: NCC, red: matcher',
                 fontsize=10)
    fig.savefig(path, dpi=70)
    plt.close(fig)


def vendors(pairs: pd.DataFrame, data_dir: Path) -> pd.Series:
    """Each pair's vendors, current / prior, e.g. 'HOLOGIC / GE'."""
    metadata = pd.read_csv(data_dir / 'metadata.csv')
    vendor = dict(zip((str(data_dir / file) for file in metadata.file_path),
                      metadata.manufacturer.str.split().str[0].str.rstrip(',')))
    return pairs.current_path.map(vendor) + ' / ' + pairs.prior_path.map(vendor)


def failure_check(model: SiameseMatcher, pairs: pd.DataFrame, data_dir: Path, predictions: Path, output_dir: Path,
                  device: str, workers: int):
    """Check 2: who hits on the real train pairs, by kind and box size, and the contact sheet."""
    real = PairDataset(real_cases(predictions, workers), CFG, train=False)
    maps = matcher_maps(model, real, device)
    cells = np.array([np.unravel_index(np.argmax(m), m.shape)[::-1] for m in maps])  # (j, i)
    # by index: PairDataset wraps its index around, so iterating it directly never ends
    samples = (real[i] for i in tqdm(range(len(real)), desc='NCC', unit='pair'))
    ncc, true = map(np.array, zip(*((ncc_point(s), truth(s)) for s in samples)))
    rows = pd.DataFrame({'kind': pairs.kind.values, 'vendors': vendors(pairs, data_dir).values,
                         'box_mm': (np.maximum(pairs.current_xmax - pairs.current_xmin,
                                               pairs.current_ymax - pairs.current_ymin) * SPACING_MM).values})
    rows[['matcher_x', 'matcher_y']] = apply_affine(grid_affine(CFG), cells)
    rows[['ncc_x', 'ncc_y']] = ncc
    rows[['true_x', 'true_y']] = true
    for method in ('matcher', 'ncc'):
        error = rows[[f'{method}_x', f'{method}_y']].values - true
        rows[f'{method}_mm'] = np.linalg.norm(error, axis=1) * CFG.spacing_mm
    rows.to_csv(output_dir / 'real_pairs.csv', index=False)

    matcher_hit, ncc_hit = rows.matcher_mm < HIT_MM, rows.ncc_mm < HIT_MM
    print(f'\nCHECK 2: {len(rows)} real train pairs, hit = within {HIT_MM:g} mm')
    print(pd.crosstab(matcher_hit.map({True: 'hits', False: 'misses'}), ncc_hit.map({True: 'hits', False: 'misses'}),
                      rownames=['matcher'], colnames=['NCC'], margins=True).to_string())
    rows['size'] = pd.qcut(rows.box_mm, 3, labels=['small', 'medium', 'large'])
    for by in ('kind', 'size', 'vendors'):
        table = rows.groupby(by, observed=True).agg(pairs=('box_mm', 'size'), box_mm=('box_mm', 'median'),
                                                    matcher_hit=('matcher_mm', lambda e: (e < HIT_MM).mean()),
                                                    ncc_hit=('ncc_mm', lambda e: (e < HIT_MM).mean()))
        print(f'\nby {by}\n' + table.round(2).to_string())

    shown = rows[ncc_hit & (rows.matcher_mm > MISS_MM)]
    print(f'\n{len(shown)} pairs where NCC hits and the matcher misses by more than {MISS_MM:g} mm')
    if len(shown):
        sheet = shown.sample(min(SHEET_PAIRS, len(shown)), random_state=SEED)
        contact_sheet(real, sheet, maps, output_dir / 'ncc_hits_matcher_misses.png')
        print(f'Wrote {output_dir / "ncc_hits_matcher_misses.png"} and {output_dir / "real_pairs.csv"}')


def main(
    checkpoint: Annotated[Path, Option(help='best.pt of train_selfsup')],
    data_dir: Annotated[Path, Option(help='The annotated set: output folder of scripts/misc/prepare_embed_dataset.py')]
        = Path(DATA_DIR),
    predictions: Annotated[Path, Option(help='predictions.csv with every pair and its anatomy box (anat_*)')]
        = Path('./outputs/b5_prior_fusion/predictions.csv'),
    output_dir: Annotated[Path, Option(help='Where real_pairs.csv and the contact sheet go')]
        = Path('./outputs/stepD_selfsup/diagnosis'),
    samples: Annotated[int, Option(help='Synthetic pairs per setting in check 1')] = 500,
    gpu: Annotated[int, Option(help='CUDA device')] = 0,
    workers: Annotated[int, Option(help='Data loader workers and parallel image loading')] = 24,
):
    device = f'cuda:{gpu}'
    output_dir.mkdir(parents=True, exist_ok=True)
    model = SiameseMatcher(CFG)
    model.load_state_dict(torch.load(checkpoint)['model'])
    pairs = train_pairs(predictions)
    lesion_check(model, pairs, data_dir, samples, device, workers)
    failure_check(model, pairs, data_dir, predictions, output_dir, device, workers)


if __name__ == '__main__':
    typer.run(main)
