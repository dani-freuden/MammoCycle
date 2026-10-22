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
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
from typer import Option

from src.data.dataset import breast_mask
from src.enums import Split
from src.utils.anatomy import anatomical_frame_uv, from_anatomical_uv, get_centroid, get_nipple_jas_cc, map_prior_to_current, to_anatomical_uv

from ..anatomical_coordinates.predict import lesion_pairs


load_dotenv()
DATA_DIR = os.getenv('EMBED_DATA_DIR')

OUTLINE_STEP = 8  # PNG pixels between the points mapped along each edge of a box
BOX = ['ymin', 'xmin', 'ymax', 'xmax']


def predict_single_pair(source_image, target_image, source_points):
    """Map points (x, y) on source_image into target_image through anatomical coordinates."""
    source_mask = breast_mask(source_image)
    target_mask = breast_mask(target_image)
    source_centroid = get_centroid(source_mask)
    target_centroid = get_centroid(target_mask)
    source_nipple = get_nipple_jas_cc(source_mask, source_centroid)
    target_nipple = get_nipple_jas_cc(target_mask, target_centroid)
    
    
    current_frame = anatomical_frame_uv(source_centroid, source_nipple)
    prior_frame = anatomical_frame_uv(target_centroid, target_nipple)
    mapped_points = []
    for current_point in source_points:
        # map_prior_to_current maps from any image into any other, despite its name.
        
        a = to_anatomical_uv(current_point, current_frame)
        predicted_prior_point = from_anatomical_uv(
            a,
            prior_frame,
        )
        mapped_points.append(predicted_prior_point)
    return mapped_points


def box_outline(box, step: float = OUTLINE_STEP) -> list[tuple[float, float]]:
    """Points (x, y) around a box (ymin, xmin, ymax, xmax), clockwise from the top left corner, at most step apart,
    each corner once."""
    ymin, xmin, ymax, xmax = box
    xs = np.linspace(xmin, xmax, int(np.ceil((xmax - xmin) / step)) + 1)
    ys = np.linspace(ymin, ymax, int(np.ceil((ymax - ymin) / step)) + 1)

    return ([(x, ymin) for x in xs] + [(xmax, y) for y in ys[1:]]
            + [(x, ymax) for x in xs[-2::-1]] + [(xmin, y) for y in ys[-2:0:-1]])


def predict_box(task: dict) -> tuple:
    """The bounding box (ymin, xmin, ymax, xmax) of the current box's outline mapped into the prior image."""
    current_image, prior_image = (np.asarray(Image.open(task[key])) for key in ('current_path', 'prior_path'))
    points = np.array(predict_single_pair(current_image, prior_image, box_outline(task['current_box'])))
    (xmin, ymin), (xmax, ymax) = points.min(axis=0), points.max(axis=0)

    return ymin, xmin, ymax, xmax


def main(
    data_dir: Annotated[Path, Option(help='Output folder of scripts/misc/prepare_embed_dataset.py')] = Path(DATA_DIR),
    splits: Annotated[list[Split], Option(help='Splits to predict; repeat the option for several')]
        = [Split.TRAIN, Split.DEV, Split.TEST],
    output_dir: Annotated[Path, Option(help='Where predictions.csv goes')]
        = Path('./outputs/anatomical_coordinates_uv'),
    workers: Annotated[int, Option(help='Parallel pairs')] = 16,
    reverse: Annotated[bool, Option(help='Know the earlier exam\'s box and predict the later one (the "current" '
                                         'columns then hold the earlier exam)')] = False,
):
    pairs = lesion_pairs(data_dir, splits, reverse)
    print(f'{len(pairs)} lesion pairs:', pairs.split_group.value_counts().to_dict())

    tasks = [{'current_path': current_path, 'prior_path': prior_path, 'current_box': current_box}
             for current_path, prior_path, current_box
             in zip(pairs.current_path, pairs.prior_path, pairs[[f'current_{c}' for c in BOX]].values.tolist())]
    with Pool(workers) as pool:
        predicted = list(tqdm(pool.imap(predict_box, tasks), total=len(tasks), desc='predict', unit='pair'))
    pairs[[f'pred_{c}' for c in BOX]] = np.round(predicted, 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output_dir / 'predictions.csv', index=False)
    print(f'Wrote {output_dir / "predictions.csv"}')


if __name__ == '__main__':
    typer.run(main)
