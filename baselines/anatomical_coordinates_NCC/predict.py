"""Anatomical coordinates + NCC baseline: the anatomical-coordinates baseline, refined by template matching. CC only.

Over the same lesion pairs as baselines.anatomical_coordinates.predict (current box known, prior box predicted, every
later/earlier pair of CC exams), the anatomical-coordinates box on the prior is the starting point. The current box,
cropped from the current image, is the template; it is matched with normalized cross-correlation in the prior image,
its centre at most search_radius pixels (in x and in y) from the anatomy box's centre. The prediction is the anatomy
box moved so its centre lands on the best match. Where the search window, clipped to the image, can't hold the
template, the prediction stays the anatomy box and ncc_score is empty.

Writes predictions.csv to output_dir with the columns of the anatomical-coordinates baseline, plus anat_* (the anatomy
box, before NCC), ncc_score and search_radius. evaluate.py scores it.

Usage:
    python -m baselines.anatomical_coordinates_NCC.predict
    python -m baselines.anatomical_coordinates_NCC.predict --splits dev --splits test --workers 32
"""
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import typer
from PIL import Image
from tqdm import tqdm
from typer import Option

from src.enums import Split

from ..anatomical_coordinates.predict import BOX, DATA_DIR, lesion_pairs
from ..anatomical_coordinates.predict import predict_box as anatomy_predict


SEARCH_RADIUS = 160  # PNG pixels; the anatomy box centre is within this of the true one, in x and y, for ~95% of pairs


def anatomy_ncc_match(current, prior, current_box, predicted_prior_point, search_radius=SEARCH_RADIUS):
    """
    Refine an anatomy-based prior location using raw-image NCC.

    Parameters
    ----------
    current, prior : np.ndarray [H, W]
        Mammograms.

    current_box : (ymin, xmin, ymax, xmax)
        Lesion box in the current image; its crop is the template.

    predicted_prior_point : (x, y)
        Location predicted in prior by anatomical coordinates.

    search_radius : int
        Search +/- this many pixels around the anatomical prediction, for the template centre.

    Returns
    -------
    best_point : np.ndarray [2]
        Best prior centre (x, y) of the template.

    best_score : float
        NCC score.

    Or None when the search window, clipped to the prior image, is smaller than the template.
    """
    ymin, xmin, ymax, xmax = np.round(current_box).astype(int)
    template = current[ymin:ymax, xmin:xmax].astype(np.float32)
    height, width = template.shape

    # Top-left corners range over the predicted point -/+ search_radius, less half the template.
    px, py = np.round(predicted_prior_point).astype(int)
    sx0 = max(0, px - search_radius - width // 2)
    sx1 = min(prior.shape[1], px + search_radius - width // 2 + width)
    sy0 = max(0, py - search_radius - height // 2)
    sy1 = min(prior.shape[0], py + search_radius - height // 2 + height)
    search = prior[sy0:sy1, sx0:sx1].astype(np.float32)

    if search.shape[0] < height or search.shape[1] < width:
        return None

    score_map = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, best_score, _, (best_x, best_y) = cv2.minMaxLoc(score_map)

    return np.array([sx0 + best_x + width / 2, sy0 + best_y + height / 2]), best_score


def predict_box(task: dict) -> tuple:
    """The anatomy box and the NCC box, each (ymin, xmin, ymax, xmax), and the NCC score (NaN without a match)."""
    anatomy = np.array(anatomy_predict(task))
    current, prior = (np.asarray(Image.open(task[key])) for key in ('current_path', 'prior_path'))
    centre = (anatomy[[1, 0]] + anatomy[[3, 2]]) / 2  # (x, y)

    match = anatomy_ncc_match(current, prior, task['current_box'], centre, task['search_radius'])
    if match is None:
        return *anatomy, *anatomy, np.nan

    point, score = match
    dx, dy = point - centre

    return *anatomy, *(anatomy + [dy, dx, dy, dx]), score


def main(
    data_dir: Annotated[Path, Option(help='Output folder of scripts/misc/prepare_embed_dataset.py')] = Path(DATA_DIR),
    splits: Annotated[list[Split], Option(help='Splits to predict; repeat the option for several')]
        = [Split.TRAIN, Split.DEV, Split.TEST],
    output_dir: Annotated[Path, Option(help='Where predictions.csv goes')]
        = Path('./outputs/anatomical_coordinates_NCC'),
    search_radius: Annotated[int, Option(help='PNG pixels around the anatomy box centre to search')] = SEARCH_RADIUS,
    workers: Annotated[int, Option(help='Parallel pairs')] = 16,
    reverse: Annotated[bool, Option(help='Know the earlier exam\'s box and predict the later one (the "current" '
                                         'columns then hold the earlier exam)')] = False,
):
    pairs = lesion_pairs(data_dir, splits, reverse)
    print(f'{len(pairs)} lesion pairs:', pairs.split_group.value_counts().to_dict())

    tasks = [{'current_path': current_path, 'prior_path': prior_path, 'current_box': current_box,
              'search_radius': search_radius}
             for current_path, prior_path, current_box
             in zip(pairs.current_path, pairs.prior_path, pairs[[f'current_{c}' for c in BOX]].values.tolist())]
    with Pool(workers) as pool:
        predicted = np.array(list(tqdm(pool.imap(predict_box, tasks), total=len(tasks), desc='predict', unit='pair')))
    pairs[[f'anat_{c}' for c in BOX]] = np.round(predicted[:, :4], 1)
    pairs[[f'pred_{c}' for c in BOX]] = np.round(predicted[:, 4:8], 1)
    pairs['ncc_score'] = np.round(predicted[:, 8], 3)
    pairs['search_radius'] = search_radius
    print(f'{pairs.ncc_score.isna().sum()} pairs kept the anatomy box: the search window could not hold the template')

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output_dir / 'predictions.csv', index=False)
    print(f'Wrote {output_dir / "predictions.csv"}')


if __name__ == '__main__':
    typer.run(main)
