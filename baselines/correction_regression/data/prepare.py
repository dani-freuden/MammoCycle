"""Data of the correction-regression baseline: the anatomical-coordinates baseline's lesion pairs with their anatomy and
regression targets, and the Dataset that crops them in anatomical coordinates.

Reads predictions.csv of baselines.anatomical_coordinates.predict: every pair of CC exams of a breast, a later (current)
and an earlier (prior) one, and lesion boxed in both, over train, dev and test. a(p, image) is a point's anatomical
coordinates (u, v) = (ρ cos θ, ρ sin θ) in the image's own anatomy: its breast mask (src.data.dataset.breast_mask),
centroid, jas_cc nipple and boundary radii (src.utils.anatomy). Both crops are WINDOW x WINDOW in (u, v), around
a_current = a(current box centre, current) in each image, so a crop pixel is the same anatomical step in both. The
target is (du, dv) = (a_true - a_current) / WINDOW, where a_true = a(true prior box centre, prior), so the corrected
centre is a⁻¹(a_current + WINDOW * (du, dv), prior). Every pair is kept, also those whose true centre lies outside the
prior window (|du| or |dv| > 0.5).

Writes to output_dir:
    pairs.csv     predictions.csv with current_u and current_v (a_current), du and dv
    anatomy.npz   the centroid, nipple and boundary radii of every image, by path
CorrectionDataset reads one split of pairs.csv, and anatomy.npz next to it, and crops the full PNGs at load time. With
jitter, for training, the prior crop's centre moves by a random (eu, ev) of up to jitter each, and the target with it,
(du - eu / WINDOW, dv - ev / WINDOW), so a pair gives a different crop and offset every time.

Usage:
    python -m baselines.correction_regression.data.prepare
"""
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import pandas as pd
import torch
import typer
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from typer import Option

from src.data.dataset import breast_mask
from src.enums import Split
from src.utils.anatomy import anatomical_coords, boundary_radii, get_centroid, get_nipple_jas_cc, image_coords


WINDOW = 0.7  # anatomical units across each crop: 18 of 3521 pairs fall outside; ~550 PNG pixels at the median
SIZE = 224  # side of the crops, the model's input
NUM_ANGLES = 720  # boundary radii per image
IDS = {'patient_id': str, 'prior_exam_id': str, 'current_exam_id': str}


class CorrectionDataset(Dataset):
    """One split of pairs.csv, as the model's batches want it: 'current' and 'prior' crops and 'target' (du, dv)."""

    def __init__(self, pairs_csv: Path, split: Split, jitter: float = 0.0):
        pairs = pd.read_csv(pairs_csv, dtype=IDS)
        self.pairs = pairs[pairs.split_group == str(split)].reset_index(drop=True)
        self.anatomy = load_anatomy(pairs_csv.with_name('anatomy.npz'))
        self.jitter = jitter

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        pair = self.pairs.iloc[index]
        centre = np.array([pair.current_u, pair.current_v])
        # torch's generator, which the DataLoader seeds differently in each worker; a numpy one made in __init__ would
        # be copied into every worker with the same state.
        shift = ((2 * torch.rand(2, dtype=torch.float64) - 1) * self.jitter).numpy()

        return {'current': anatomical_crop(pair.current_path, *self.anatomy[pair.current_path], centre),
                'prior': anatomical_crop(pair.prior_path, *self.anatomy[pair.prior_path], centre + shift),
                'target': torch.tensor(np.array([pair.du, pair.dv]) - shift / WINDOW, dtype=torch.float32)}


def anatomical_crop(path: str, centroid, nipple, radii, centre) -> Tensor:
    """The WINDOW x WINDOW window of an image around the anatomical point centre (u, v), resampled in anatomical
    coordinates to (1, SIZE, SIZE) in [0, 1], zero outside the image. Oriented like the image, nipple on the left: u
    falls to the right and v downwards.

    Sampled from the image at half resolution (cv2.pyrDown, which smooths it against aliasing): a crop pixel spans
    about 1.2 of its pixels at the median lesion, 0.8 at the 5th percentile and 1.9 at the 95th.
    """
    offsets = ((np.arange(SIZE) + 0.5) / SIZE - 0.5) * WINDOW
    u, v = np.meshgrid(centre[0] - offsets, centre[1] - offsets)
    x, y = (image_coords(np.stack([u, v], axis=-1), centroid, nipple, radii) / 2).astype(np.float32).transpose(2, 0, 1)
    with Image.open(path) as image:
        half = cv2.pyrDown(np.asarray(image))  # its pixel i is the image's pixel 2i
    window = cv2.remap(half, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return torch.from_numpy(window)[None].float() / 255


def load_anatomy(path: Path) -> dict[str, tuple]:
    """anatomy.npz as {image path: (centroid, nipple, radii)}."""
    with np.load(path) as anatomy:
        columns = (anatomy[key] for key in ('paths', 'centroids', 'nipples', 'radii'))
        return {str(p): (centroid, nipple, radii) for p, centroid, nipple, radii in zip(*columns)}


def centre(pairs: pd.DataFrame, box: str) -> np.ndarray:
    """The centres (x, y) of the boxes box_*, (N, 2) in PNG pixels."""
    return np.stack([(pairs[f'{box}_xmin'] + pairs[f'{box}_xmax']) / 2,
                     (pairs[f'{box}_ymin'] + pairs[f'{box}_ymax']) / 2], axis=1)


def image_anatomy(path: str, point) -> tuple:
    """An image's anatomy (centroid, nipple, boundary radii), and the anatomical coordinates (u, v) of a point in it."""
    with Image.open(path) as image:
        mask = breast_mask(np.asarray(image))
    centroid = np.array(get_centroid(mask), dtype=float)
    nipple = np.array(get_nipple_jas_cc(mask, centroid), dtype=float)

    return ((centroid, nipple, boundary_radii(mask, centroid, NUM_ANGLES)),
            anatomical_coords(np.array([point]), mask, centroid, nipple, NUM_ANGLES)[0])


def pair_anatomy(task: tuple) -> tuple:
    current_path, current_centre, prior_path, true_centre = task
    return image_anatomy(current_path, current_centre), image_anatomy(prior_path, true_centre)


def main(
    predictions: Annotated[Path, Option(help='predictions.csv of baselines.anatomical_coordinates.predict')]
        = Path('./outputs/anatomical_coordinates/predictions.csv'),
    output_dir: Annotated[Path, Option(help='Where pairs.csv and anatomy.npz go')]
        = Path('./outputs/correction_regression/data'),
    workers: Annotated[int, Option(help='Parallel pairs')] = 16,
):
    pairs = pd.read_csv(predictions, dtype=IDS)
    tasks = list(zip(pairs.current_path, centre(pairs, 'current'), pairs.prior_path, centre(pairs, 'prior')))
    with Pool(workers) as pool:
        results = list(tqdm(pool.imap(pair_anatomy, tasks), total=len(tasks), desc='anatomy', unit='pair'))

    anatomy, a_current, a_true = {}, [], []
    for (current_path, _, prior_path, _), ((current, uv_current), (prior, uv_true)) in zip(tasks, results):
        anatomy[current_path], anatomy[prior_path] = current, prior
        a_current.append(uv_current)
        a_true.append(uv_true)
    a_current, a_true = np.array(a_current), np.array(a_true)
    pairs['current_u'], pairs['current_v'] = a_current.T
    pairs['du'], pairs['dv'] = ((a_true - a_current) / WINDOW).T

    outside = (pairs.du.abs() > 0.5) | (pairs.dv.abs() > 0.5)
    print(f'{len(pairs)} lesion pairs:', pairs.split_group.value_counts().to_dict())
    print(f'{outside.sum()} with the true centre outside the prior window, kept:',
          pairs.split_group[outside].value_counts().to_dict())

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output_dir / 'pairs.csv', index=False)
    paths = list(anatomy)
    np.savez(output_dir / 'anatomy.npz', paths=np.array(paths),
             **{key: np.stack([anatomy[path][i] for path in paths])
                for i, key in enumerate(('centroids', 'nipples', 'radii'))})
    print(f'Wrote {output_dir / "pairs.csv"} and the anatomy of {len(paths)} images to {output_dir / "anatomy.npz"}')


if __name__ == '__main__':
    typer.run(main)
