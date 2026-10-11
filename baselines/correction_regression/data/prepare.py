"""Data of the correction-regression baseline: the anatomical-coordinates baseline's lesion pairs with crop centres and
regression targets, and the Dataset that crops them.

Reads predictions.csv of baselines.anatomical_coordinates.predict: every pair of CC exams of a breast, a later (current)
and an earlier (prior) one, and lesion boxed in both, over train, dev and test. For each pair the current centre is the
current box's centre and the anatomical centre is the predicted (pred_*) box's centre, both rounded to whole PNG pixels,
so each is exactly the centre of its crop. The target is (dx, dy) = anatomical centre - true prior box centre, as a
fraction of WINDOW, so the corrected centre is the anatomical centre - WINDOW * (dx, dy). Every pair is kept, also those
whose true centre lies outside the prior window (|dx| or |dy| > 0.5).

Writes pairs.csv to output_dir: predictions.csv with current_x, current_y, anat_x, anat_y, dx and dy.
CorrectionDataset reads one split of it and crops the full PNGs at load time.

Usage:
    python -m baselines.correction_regression.data.prepare
"""
from pathlib import Path
from typing import Annotated

import pandas as pd
import torch
import typer
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from typer import Option

from src.enums import Split


WINDOW = 448  # PNG pixels around each centre
SIZE = 224  # side of the resized crops, the model's input
IDS = {'patient_id': str, 'prior_exam_id': str, 'current_exam_id': str}


class CorrectionDataset(Dataset):
    """One split of pairs.csv, as the model's batches want it: 'current' and 'prior' crops and 'target' (dx, dy)."""

    def __init__(self, pairs_csv: Path, split: Split):
        pairs = pd.read_csv(pairs_csv, dtype=IDS)
        self.pairs = pairs[pairs.split_group == str(split)].reset_index(drop=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        pair = self.pairs.iloc[index]

        return {'current': crop(pair.current_path, pair.current_x, pair.current_y),
                'prior': crop(pair.prior_path, pair.anat_x, pair.anat_y),
                'target': torch.tensor([pair.dx, pair.dy], dtype=torch.float32)}


def crop(path: str, x: int, y: int) -> Tensor:
    """The WINDOW x WINDOW window of an image centred on (x, y), zero outside the image, resized to (1, SIZE, SIZE) in
    [0, 1]."""
    half = WINDOW // 2
    with Image.open(path) as image:
        window = image.crop((x - half, y - half, x + half, y + half))  # PIL fills outside the image with zeros

    return pil_to_tensor(window.resize((SIZE, SIZE), Image.Resampling.BILINEAR)).float() / 255


def centre(pairs: pd.DataFrame, box: str) -> tuple[pd.Series, pd.Series]:
    """The centres (x, y) of the boxes box_*, in PNG pixels."""
    return (pairs[f'{box}_xmin'] + pairs[f'{box}_xmax']) / 2, (pairs[f'{box}_ymin'] + pairs[f'{box}_ymax']) / 2


def main(
    predictions: Annotated[Path, Option(help='predictions.csv of baselines.anatomical_coordinates.predict')]
        = Path('./outputs/anatomical_coordinates/predictions.csv'),
    output_dir: Annotated[Path, Option(help='Where pairs.csv goes')] = Path('./outputs/correction_regression/data'),
):
    pairs = pd.read_csv(predictions, dtype=IDS)
    current_x, current_y = centre(pairs, 'current')
    anat_x, anat_y = centre(pairs, 'pred')
    true_x, true_y = centre(pairs, 'prior')

    pairs['current_x'], pairs['current_y'] = current_x.round().astype(int), current_y.round().astype(int)
    pairs['anat_x'], pairs['anat_y'] = anat_x.round().astype(int), anat_y.round().astype(int)
    pairs['dx'], pairs['dy'] = (pairs.anat_x - true_x) / WINDOW, (pairs.anat_y - true_y) / WINDOW

    outside = (pairs.dx.abs() > 0.5) | (pairs.dy.abs() > 0.5)
    print(f'{len(pairs)} lesion pairs:', pairs.split_group.value_counts().to_dict())
    print(f'{outside.sum()} with the true centre outside the prior window, kept:',
          pairs.split_group[outside].value_counts().to_dict())

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output_dir / 'pairs.csv', index=False)
    print(f'Wrote {output_dir / "pairs.csv"}')


if __name__ == '__main__':
    typer.run(main)
