"""Train the correction-regression baseline on pairs.csv of data/prepare.py: the train split to train, dev to validate.

Keeps the checkpoint with the lowest val/loss and the last one, under output_dir.

Usage:
    python -m baselines.correction_regression.train
    python -m baselines.correction_regression.train --gpu 1 --batch-size 64 --max-epochs 30
"""
from pathlib import Path
from typing import Annotated

import lightning as L
import typer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from typer import Option

from src.enums import Split

from .data.prepare import CorrectionDataset
from .model.module import CorrectionRegressionModel


MIRAI_CHECKPOINT = Path('/data/rbg/users/dani_fr/projects/mirai-shira/runs/shifa-8bit-2-steps4000/best.ckpt')


def main(
    pairs_csv: Annotated[Path, Option(help='pairs.csv of data/prepare.py, with its anatomy.npz next to it')]
        = Path('./outputs/correction_regression/data/pairs.csv'),
    checkpoint: Annotated[Path, Option(help='mirai-shira checkpoint for the backbone')] = MIRAI_CHECKPOINT,
    output_dir: Annotated[Path, Option(help='Where the logs and checkpoints go')] = Path('./runs/correction_regression'),
    batch_size: Annotated[int, Option(help='Pairs per batch')] = 32,
    max_epochs: Annotated[int, Option(help='Epochs over the train split')] = 80,
    learning_rate: Annotated[float, Option(help='AdamW learning rate')] = 1e-4,
    weight_decay: Annotated[float, Option(help='AdamW weight decay')] = 1e-4,
    huber_delta: Annotated[float, Option(help='Huber delta, in windows (0.05 is 0.035 anatomical units, '
                                              '~28 PNG pixels at the median lesion)')] = 0.05,
    jitter: Annotated[float, Option(help='Train only: anatomical units the prior crop centre moves, at most, in u '
                                         'and v (0.1 is ~80 PNG pixels at the median lesion)')] = 0.1,
    dropout: Annotated[float, Option(help='Dropout on the features going into the head')] = 0.3,
    num_workers: Annotated[int, Option(help='DataLoader workers')] = 16,
    gpu: Annotated[int, Option(help='GPU index to train on')] = 0,
):
    module = CorrectionRegressionModel(checkpoint, huber_delta=huber_delta, learning_rate=learning_rate,
                                       weight_decay=weight_decay, dropout=dropout)
    train = DataLoader(CorrectionDataset(pairs_csv, Split.TRAIN, jitter), batch_size, shuffle=True,
                       num_workers=num_workers)
    dev = DataLoader(CorrectionDataset(pairs_csv, Split.DEV), batch_size, num_workers=num_workers)

    trainer = L.Trainer(
        default_root_dir=output_dir,
        devices=[gpu],
        max_epochs=max_epochs,
        precision='16-mixed',
        gradient_clip_val=1.0,
        # Not save_last=True: Lightning rewrites last.ckpt only when the best one is saved, so it was always the best.
        callbacks=[ModelCheckpoint(monitor='val/loss', mode='min'), ModelCheckpoint(filename='last')],
    )
    trainer.fit(module, train, dev)


if __name__ == '__main__':
    typer.run(main)
