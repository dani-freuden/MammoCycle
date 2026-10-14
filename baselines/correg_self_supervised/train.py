"""Pretrain the correction-regression model without annotations, on dataset.py's pairs made from single images of the
train split, and validate it on the real dev lesion pairs of correction_regression's pairs.csv, so val/error_px is the
error on the real task.

Keeps the checkpoint with the lowest val/loss and the last one, under output_dir.

Usage:
    python -m baselines.correg_self_supervised.train
    python -m baselines.correg_self_supervised.train --gpu 1 --dropout 0 --max-epochs 400
"""
from pathlib import Path
from typing import Annotated

import lightning as L
import typer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from typer import Option

from src.enums import Split

from ..correction_regression.data.prepare import WINDOW, CorrectionDataset
from ..correction_regression.model.module import CorrectionRegressionModel
from ..correction_regression.train import MIRAI_CHECKPOINT
from .dataset import SelfSupervisedDataset


def main(
    metadata_csv: Annotated[Path, Option(help='metadata.csv of the prepared images; the train split is used')]
        = Path('/data/rbg/users/dani_fr/embed_prepared_8bit/metadata.csv'),
    pairs_csv: Annotated[Path, Option(help='pairs.csv of correction_regression/data/prepare.py; dev is validated on')]
        = Path('./outputs/correction_regression/data/pairs.csv'),
    checkpoint: Annotated[Path, Option(help='mirai-shira checkpoint for the backbone')] = MIRAI_CHECKPOINT,
    output_dir: Annotated[Path, Option(help='Where the logs and checkpoints go')]
        = Path('./runs/correg_self_supervised'),
    batch_size: Annotated[int, Option(help='Pairs per batch')] = 32,
    max_epochs: Annotated[int, Option(help='Epochs over the train images, one random pair per image each')] = 200,
    learning_rate: Annotated[float, Option(help='AdamW learning rate')] = 1e-4,
    weight_decay: Annotated[float, Option(help='AdamW weight decay')] = 1e-4,
    huber_delta: Annotated[float, Option(help='Huber delta, in windows (0.05 is ~22 PNG pixels)')] = 0.05,
    max_shift: Annotated[int, Option(help='PNG pixels the prior crop centre moves from the point, at most, in x and y')]
        = WINDOW // 2,
    dropout: Annotated[float, Option(help='Dropout on the features going into the head')] = 0.3,
    num_workers: Annotated[int, Option(help='DataLoader workers')] = 16,
    gpu: Annotated[int, Option(help='GPU index to train on')] = 0,
):
    module = CorrectionRegressionModel(checkpoint, huber_delta=huber_delta, learning_rate=learning_rate,
                                       weight_decay=weight_decay, dropout=dropout)
    train = DataLoader(SelfSupervisedDataset(metadata_csv, Split.TRAIN, max_shift), batch_size, shuffle=True,
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
