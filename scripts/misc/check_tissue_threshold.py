"""Check how reliable `pixel > threshold` is as a tissue mask on the frames the model sees.

Images are taken from the records and loaded through CycleDataset, so they are cropped and fitted to the canvas
as in training, and patch positions are evaluated on the encoder's feature grid. For every sampled image the simple
threshold mask (the candidate) is compared against a reference mask (triangle threshold, then the largest
contour, filled). Three things are measured:

1. How dark the background is, away from the breast edge.
2. How often the two masks disagree, in each direction.
3. How many patch positions the sampler would accept with the threshold but the reference clearly rejects.

Usage:
    python scripts/misc/check_tissue_threshold.py   # prompts for every option
    python scripts/misc/check_tissue_threshold.py --split train --view CC --num-images 10000 --out-dir threshold_check
"""
import os
import random
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Literal

import cv2
import numpy as np
import pandas as pd
import typer
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from typer import Option

from src.data.dataset import CycleDataset
from src.data.records import Record, load_records
from src.enums import Split

load_dotenv()
DATA_DIR = os.getenv('DATA_DIR')

STRIDE = 8  # patch positions are evaluated every STRIDE pixels, like the encoder's feature grid
METRICS = ['background_exact_zero', 'background_p99', 'tissue_p01', 'reference_level',
           'tissue_share', 'false_tissue', 'missed_tissue', 'bad_positions']


def sample_images(data_dir: str, split: str, view: str, num_images: int, seed: int) -> list[Record]:
    """Sample exams from the records as one-exam records, so CycleDataset loads one frame per item.

    Each exam belongs to exactly one record, so no image is drawn twice.
    """
    splits = list(Split) if split == 'all' else [Split(split)]
    records = [record for split_records in load_records(Path(data_dir), splits).values() for record in split_records]
    images = [replace(record, exams=(exam,)) for record in records if view in ('all', str(record.view)) for exam in record.exams]

    return random.Random(seed).sample(images, min(num_images, len(images)))


def reference_mask(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Triangle threshold, then the largest contour, filled. Returns the mask and the threshold level in [0, 1].

    The triangle method places the threshold at the foot of the dominant histogram peak, which here is the
    background. Unlike Otsu it keeps dark fat near the skin, and it adapts when the background is not at zero.
    """
    image_8bit = np.round(image * 255).astype(np.uint8)
    level, binary = cv2.threshold(image_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(binary)
    if contours:
        cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)

    return mask > 0, level / 255


def tissue_fraction(mask: np.ndarray, patch_size: int) -> np.ndarray:
    """Fraction of tissue inside the patch window centred on each STRIDE-spaced position."""
    small = cv2.resize(mask.astype(np.float32), None, fx=1 / STRIDE, fy=1 / STRIDE, interpolation=cv2.INTER_AREA)
    window = max(patch_size // STRIDE, 1)

    return cv2.boxFilter(small, -1, (window, window), normalize=True, borderType=cv2.BORDER_CONSTANT)


def analyse(image: np.ndarray, threshold: float, patch_size: int, min_fraction: float, tolerance: float,
            edge_band: int) -> dict:
    candidate = image > threshold
    reference, reference_level = reference_mask(image)

    # The two masks always disagree a little at the skin line, so background statistics are taken
    # only outside a band around the reference mask.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * edge_band + 1, 2 * edge_band + 1))
    far_background = cv2.dilate(reference.astype(np.uint8), kernel) == 0
    background = image[far_background]

    accepted = tissue_fraction(candidate, patch_size) >= min_fraction
    clearly_rejected = tissue_fraction(reference, patch_size) < min_fraction - tolerance

    return {
        'reference_level': reference_level,
        'tissue_share': candidate.mean(),
        'background_exact_zero': (background == 0).mean() if background.size else np.nan,
        'background_p99': np.percentile(background, 99) if background.size else np.nan,
        'tissue_p01': np.percentile(image[reference], 1) if reference.any() else np.nan,
        'false_tissue': candidate[far_background].mean() if background.size else np.nan,
        'missed_tissue': (~candidate[reference]).mean() if reference.any() else np.nan,
        'bad_positions': (accepted & clearly_rejected).sum() / max(accepted.sum(), 1),
        'num_positions': int(accepted.sum()),
    }


def save_overlay(image: np.ndarray, threshold: float, out_path: Path):
    """Draw the candidate outline in red and the reference outline in green."""
    reference, _ = reference_mask(image)
    canvas = cv2.cvtColor(np.round(image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    for mask, colour in ((image > threshold, (0, 0, 255)), (reference, (0, 255, 0))):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, colour, 1)

    cv2.imwrite(str(out_path), canvas)


def summarise(results: pd.DataFrame, bad_positions_limit: float):
    table = results[METRICS].describe(percentiles=[0.5, 0.9, 0.99, 0.999]).T
    print(table.drop(columns=['count', 'std']).to_string(float_format='{:.4f}'.format))

    failing = results['bad_positions'] > bad_positions_limit
    print(f'\nImages with more than {bad_positions_limit:.0%} bad patch positions: '
          f'{failing.sum()} of {len(results)} ({failing.mean():.2%})')
    print(f'Images with no accepted patch position at all: {(results["num_positions"] == 0).sum()}')


def main(
    data_dir: Annotated[str, Option(prompt=True, help='Path to the prepared data directory')] = DATA_DIR,
    split: Annotated[Literal['all', 'train', 'dev', 'test'], Option(prompt=True, help='Split to sample from')] = 'train',
    view: Annotated[Literal['all', 'CC', 'MLO'], Option(prompt=True, help='View to sample from')] = 'all',
    num_images: Annotated[int, Option(prompt=True, help='Number of images to sample')] = 10_000,
    height: Annotated[int, Option(prompt=True, help='Frame height, as in the config')] = 1024,
    width: Annotated[int, Option(prompt=True, help='Frame width, as in the config')] = 640,
    patch_size: Annotated[int, Option(prompt=True, help='Patch size in frame pixels')] = 160,
    threshold: Annotated[float, Option(prompt=True, help='Candidate tissue threshold, in [0, 1]')] = 0.01,
    min_fraction: Annotated[float, Option(prompt=True, help='Tissue fraction the sampler requires')] = 0.75,
    tolerance: Annotated[float, Option(prompt=True, help='Slack before a position counts as bad')] = 0.15,
    edge_band: Annotated[int, Option(prompt=True, help='Frame pixels around the breast edge to ignore')] = 8,
    bad_positions_limit: Annotated[float, Option(prompt=True, help='Share of bad positions that fails an image')] = 0.01,
    save_worst: Annotated[int, Option(prompt=True, help='Number of overlay images to save')] = 30,
    out_dir: Annotated[Path, Option(prompt=True, help='Where to write results.csv and the overlays')] = Path('threshold_check'),
    num_workers: Annotated[int, Option(prompt=True, help='DataLoader workers')] = 8,
    seed: Annotated[int, Option(prompt=True, help='Seed for sampling the images')] = 0,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sample_images(data_dir, split, view, num_images, seed)
    dataset = CycleDataset(images, height, width)
    loader = DataLoader(dataset, batch_size=32, num_workers=num_workers)

    rows = []
    for batch in loader:
        for frames in batch['frames'].numpy():  # [1, 1, height, width] per one-exam record
            rows.append(analyse(frames[0, 0], threshold, patch_size, min_fraction, tolerance, edge_band))

    results = pd.DataFrame(rows)
    results.insert(0, 'path', [str(record.exams[0].path) for record in images])
    results.to_csv(out_dir / 'results.csv', index=False)
    if results.empty:
        return

    summarise(results, bad_positions_limit)

    worst = results.sort_values(['bad_positions', 'false_tissue'], ascending=False).head(save_worst)
    for rank, (index, path) in enumerate(zip(worst.index, worst['path'])):
        image = dataset[index]['frames'][0, 0].numpy()
        save_overlay(image, threshold, out_dir / f'{rank:03d}_{Path(path).stem}.png')
    print(f'\nSaved results.csv and {len(worst)} overlays (red = threshold, green = reference) to {out_dir}')


if __name__ == '__main__':
    typer.run(main)
