"""Correction-regression baseline: correct the anatomical-coordinates baseline's prior box with a trained model. CC only.

For each lesion pair of data/prepare.py's pairs.csv, the model reads both crops in anatomical coordinates, around
a_current, the current box centre's anatomical coordinates, and predicts (du, dv): where the lesion moved, as a fraction
of WINDOW. The corrected centre is a⁻¹(a_current + WINDOW * (du, dv), prior), and, as in the NCC baseline, the
prediction is the anatomy box moved so its centre lands there.

Writes predictions.csv to output_dir: pairs.csv with the anatomical baseline's pred_* renamed anat_* (the anatomy box),
the model's output pred_du and pred_dv, the corrected centre corrected_x and corrected_y, and pred_* (the moved box).
evaluate.py scores it.

Usage:
    python -m baselines.correction_regression.predict --checkpoint runs/correction_regression/lightning_logs/version_0/checkpoints/<best>.ckpt
    python -m baselines.correction_regression.predict --checkpoint <best>.ckpt --splits dev --splits test
"""
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm
from typer import Option

from src.enums import Split
from src.utils.anatomy import image_coords

from ..anatomical_coordinates.predict import BOX
from .data.prepare import WINDOW, CorrectionDataset
from .model.module import CorrectionRegressionModel


@torch.no_grad()
def predict_split(model: CorrectionRegressionModel, dataset: CorrectionDataset, batch_size: int,
                  num_workers: int) -> np.ndarray:
    """The model's (du, dv) for every pair of the dataset, (N, 2), in its order."""
    outputs = [model(batch['current'].to(model.device), batch['prior'].to(model.device)).cpu()
               for batch in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers), desc='predict', unit='batch')]

    return torch.cat(outputs).numpy()


def main(
    checkpoint: Annotated[Path, Option(help='Checkpoint of train.py, e.g. the best val/loss one')],
    pairs_csv: Annotated[Path, Option(help='pairs.csv of data/prepare.py')]
        = Path('./outputs/correction_regression/data/pairs.csv'),
    splits: Annotated[list[Split], Option(help='Splits to predict; repeat the option for several')] = [Split.TEST],
    output_dir: Annotated[Path, Option(help='Where predictions.csv goes')] = Path('./outputs/correction_regression'),
    batch_size: Annotated[int, Option(help='Pairs per batch')] = 64,
    num_workers: Annotated[int, Option(help='DataLoader workers')] = 16,
    gpu: Annotated[int, Option(help='GPU index to predict on')] = 0,
):
    # weights_only=False: the saved hyperparameters hold the Mirai checkpoint as a Path, which torch.load refuses by
    # default; the file is train.py's own.
    model = CorrectionRegressionModel.load_from_checkpoint(checkpoint, map_location=torch.device(f'cuda:{gpu}'),
                                                           weights_only=False).eval()
    pairs = []
    for split in splits:
        dataset = CorrectionDataset(pairs_csv, split)
        output = predict_split(model, dataset, batch_size, num_workers)
        corrected = np.array([image_coords(a, *dataset.anatomy[path]) for a, path in
                              zip(dataset.pairs[['current_u', 'current_v']].values + WINDOW * output,
                                  dataset.pairs.prior_path)])
        pairs.append(dataset.pairs.assign(pred_du=output[:, 0], pred_dv=output[:, 1],
                                          corrected_x=corrected[:, 0], corrected_y=corrected[:, 1]))
    pairs = pd.concat(pairs, ignore_index=True).rename(columns={f'pred_{c}': f'anat_{c}' for c in BOX})
    print(f'{len(pairs)} lesion pairs:', pairs.split_group.value_counts().to_dict())

    # Move the anatomy box by the corrected centre minus its centre.
    shift_x = pairs.corrected_x - (pairs.anat_xmin + pairs.anat_xmax) / 2
    shift_y = pairs.corrected_y - (pairs.anat_ymin + pairs.anat_ymax) / 2
    for c, shift in (('ymin', shift_y), ('xmin', shift_x), ('ymax', shift_y), ('xmax', shift_x)):
        pairs[f'pred_{c}'] = (pairs[f'anat_{c}'] + shift).round(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output_dir / 'predictions.csv', index=False)
    print(f'Wrote {output_dir / "predictions.csv"}')


if __name__ == '__main__':
    typer.run(main)
