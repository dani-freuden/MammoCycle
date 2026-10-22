"""Anatomical-coordinates baseline: find each lesion of the current exam in a prior one. CC only.

As in MammoTracker (arXiv:2507.00328), the current box is known and the prior one is predicted, over every pair of CC
exams of a breast (to_pairs), a later (current) and an earlier (prior) one, not only consecutive exams. For every such pair and
every mammotracker lesion boxed in both, points along the current box's outline are mapped through anatomical
coordinates (breast mask, centroid and nipple of each image) into the prior image, and their bounding box is the
prediction. Runs on the prepared PNGs, where boxes.csv's png_* boxes are.

Writes predictions.csv to output_dir: one row per lesion pair, with the current box, the true prior box and the
predicted one, each as (ymin, xmin, ymax, xmax), and the prior image's mm per PNG pixel (empty without a
PixelSpacing). evaluate.py scores it.

Usage:
    python -m baselines.anatomical_coordinates.predict
    python -m baselines.anatomical_coordinates.predict --splits dev --splits test --workers 32
"""
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from PIL import Image
from tqdm import tqdm
from typer import Option

from src.constants import BOX, EMBED_DATA_DIR, Split
from src.data.lesion_pairs import DataDir, Reverse, lesion_pairs, pair_tasks
from src.utils.anatomy import BreastAnatomy
from src.utils.boxes import box_outline


def predict_single_pair(source_image, target_image, source_points):
    """Map points (x, y) on source_image into target_image through anatomical coordinates."""
    source, target = BreastAnatomy.of(source_image), BreastAnatomy.of(target_image)
    return [source.map_to(target, point) for point in source_points]


def predict_box(task: dict) -> tuple:
    """The bounding box (ymin, xmin, ymax, xmax) of the current box's outline mapped into the prior image."""
    current_image, prior_image = (np.asarray(Image.open(task[key])) for key in ('current_path', 'prior_path'))
    points = np.array(predict_single_pair(current_image, prior_image, box_outline(task['current_box'])))
    (xmin, ymin), (xmax, ymax) = points.min(axis=0), points.max(axis=0)

    return ymin, xmin, ymax, xmax


def main(
    data_dir: DataDir = EMBED_DATA_DIR,
    splits: Annotated[list[Split], Option(help='Splits to predict; repeat the option for several')]
        = [Split.TRAIN, Split.DEV, Split.TEST],
    output_dir: Annotated[Path, Option(help='Where predictions.csv goes')]
        = Path('./outputs/anatomical_coordinates'),
    workers: Annotated[int, Option(help='Parallel pairs')] = 16,
    reverse: Reverse = False,
):
    pairs = lesion_pairs(data_dir, splits, reverse)
    print(f'{len(pairs)} lesion pairs:', pairs.split_group.value_counts().to_dict())

    tasks = pair_tasks(pairs)
    with Pool(workers) as pool:
        predicted = list(tqdm(pool.imap(predict_box, tasks), total=len(tasks), desc='predict', unit='pair'))
    pairs[[f'pred_{c}' for c in BOX]] = np.round(predicted, 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output_dir / 'predictions.csv', index=False)
    print(f'Wrote {output_dir / "predictions.csv"}')


if __name__ == '__main__':
    typer.run(main)
