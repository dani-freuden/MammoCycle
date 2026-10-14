"""Score a checkpoint on the self-supervised task of dataset.py: the corrected centre's distance to the true point, in
PNG pixels, on pairs made from the images of each split, with dropout off.

Every image gives repeats pairs, each with its own random point, shift and intensity augmentation; the pairs are the
same for every checkpoint while repeats and num_workers stay the same. A model that fits the train images much better
than the held-out dev ones has overfitted them. Beside the model's error is the uncorrected one, the shift's own length.

Usage:
    python -m baselines.correg_self_supervised.evaluate --checkpoint runs/correg_self_supervised/lightning_logs/version_0/checkpoints/last.ckpt
"""
from pathlib import Path
from typing import Annotated

import pandas as pd
import torch
import typer
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm
from typer import Option

from src.enums import Split

from ..correction_regression.data.prepare import WINDOW
from ..correction_regression.model.module import CorrectionRegressionModel
from .dataset import SelfSupervisedDataset


@torch.no_grad()
def errors(model: CorrectionRegressionModel, dataset: Dataset, batch_size: int, num_workers: int) -> Tensor:
    """(N, 2): the corrected and the uncorrected centre's error, in PNG pixels, for every pair of the dataset."""
    out = []
    for batch in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers), desc='evaluate', unit='batch'):
        predicted = model(batch['current'].to(model.device), batch['prior'].to(model.device)).cpu()
        target = batch['target']
        out.append(WINDOW * torch.stack([(target - predicted).norm(dim=1), target.norm(dim=1)], dim=1))

    return torch.cat(out)


def main(
    checkpoint: Annotated[Path, Option(help='Checkpoint of train.py')],
    metadata_csv: Annotated[Path, Option(help='metadata.csv of the prepared images')]
        = Path('/data/rbg/users/dani_fr/embed_prepared_8bit/metadata.csv'),
    splits: Annotated[list[Split], Option(help='Splits to score; repeat the option for several')]
        = [Split.TRAIN, Split.DEV],
    repeats: Annotated[int, Option(help='Pairs per image')] = 5,
    max_shift: Annotated[int, Option(help='As in train.py: PNG pixels the prior crop centre moves, at most, each way')]
        = WINDOW // 2,
    batch_size: Annotated[int, Option(help='Pairs per batch')] = 64,
    num_workers: Annotated[int, Option(help='DataLoader workers')] = 16,
    gpu: Annotated[int, Option(help='GPU index to predict on')] = 0,
):
    # weights_only=False: the saved hyperparameters hold the Mirai checkpoint as a Path, which torch.load refuses by
    # default; the file is train.py's own.
    model = CorrectionRegressionModel.load_from_checkpoint(checkpoint, map_location=torch.device(f'cuda:{gpu}'),
                                                           weights_only=False).eval()
    rows = {}
    for split in splits:
        dataset = SelfSupervisedDataset(metadata_csv, split, max_shift)
        torch.manual_seed(0)  # the DataLoader seeds its workers from it, so every checkpoint sees the same pairs
        error = errors(model, ConcatDataset([dataset] * repeats), batch_size, num_workers)
        rows[f'{split} ({len(dataset)} images x {repeats})'] = {
            'mean': error[:, 0].mean().item(), 'median': error[:, 0].median().item(),
            'p90': error[:, 0].quantile(0.9).item(),
            'uncorrected mean': error[:, 1].mean().item(), 'uncorrected median': error[:, 1].median().item()}

    print(f'\n{checkpoint}\nDistance from the corrected centre to the true point, in PNG pixels:')
    print(pd.DataFrame(rows).T.round(1).to_string())


if __name__ == '__main__':
    typer.run(main)
