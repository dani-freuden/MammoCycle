"""Train LocalAppearanceMatcher on the EMBED CC lesion pairs, then predict every pair fused with the anatomical prior.

Same pairs, anatomy boxes and prior as experimental/mixturemodel/lesion_matching/predict.py (B5), in PNG pixels at
its nominal SPACING_MM. The current crop is CURRENT_SIZE px around the current box centre, the prior crop PRIOR_SIZE px.

Train: half the samples go current -> prior, half prior -> current (both boxes are known). The prior crop is rotated
and scaled at random (ROTATION, SCALE) and centred with the true centre anywhere in it (P_UNIFORM of the samples, and
every prior -> current one) or on the anatomical point plus JITTER, redrawn until the true centre is inside, as in
Step D. Both crops get augment_intensity. Dev: current -> prior, prior crop centred on the anatomical point.

The prior is fitted on the train pairs, and the checkpoint with the best fused truncated mean on dev is kept (see
LocalAppearanceMatcher). That checkpoint then predicts every pair, tau is chosen on dev, and predictions.csv is
written as predict.py writes it.

Usage (from the repo root):
    python -m experimental.cross_cosine_nn.train --gpu 0
    python -m baselines.anatomical_coordinates.evaluate --output-dir outputs/cross_cosine_nn
"""
from pathlib import Path
from typing import Annotated

import lightning as L
import numpy as np
import torch
import typer
from lightning.pytorch.callbacks import ModelCheckpoint
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typer import Option

from baselines.anatomical_coordinates.predict import BOX, DATA_DIR
from experimental.mixturemodel.lesion_matching.predict import (SPACING_MM, centre, decode_all, load_cases, report,
                                                               write_predictions)
from src.data.augmentations import augment_intensity
from src.enums import Split

from .local_appearance_matcher import FUSION_TAUS, LocalAppearanceMatcher

# on sys.path through the imports above
from b5_prior_fusion import AnatomicalPrior, ScoreMap, local_residuals_mm
from geometry import apply_affine, crop_affine, invert_affine, warp_crop


CURRENT_SIZE = 256  # px, ~31 mm
PRIOR_SIZE = 512  # px; the anatomical point is within 256 px of the true centre, in x and y, for >99% of pairs
P_UNIFORM = 0.7  # share of the current -> prior windows with the true centre anywhere in the crop
LIMIT = 0.9 * PRIOR_SIZE / 2  # px; most a uniform window's centre is off the true centre, in x and y
JITTER = 2.0 / SPACING_MM  # px; sd of the anatomical-point window's offset (Step D's 2 mm)
ROTATION, SCALE = 10.0, 0.1  # degrees, fraction
MEAN, STD = 0.449, 0.226  # ImageNet's per-channel statistics averaged: conv1 sums the RGB filters


def crop(path: str, M: np.ndarray, size: int) -> Tensor:
    """(1, size, size) in [0, 1] of the PNG, sampled on the grid M (crop px -> PNG px), zero outside the image."""
    with Image.open(path) as image:
        return torch.from_numpy(warp_crop(np.asarray(image), M, (size, size)))[None].float() / 255


class PairDataset(Dataset):
    """Lesion pairs as LocalAppearanceMatcher's batches want them. items: (current_path, prior_path, current box
    centre, predict.load_cases' case) per pair."""

    def __init__(self, items: list, train: bool):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        current_path, prior_path, p_current, case = self.items[index]
        rng = np.random.default_rng()
        src, dst, p_src, p_dst = current_path, prior_path, p_current, case['p_gt']
        reverse = self.train and rng.random() < 0.5
        if reverse:
            src, dst, p_src, p_dst = dst, src, p_dst, p_src

        center, angle, scale = case['p_anat'], 0.0, 1.0
        for _ in range(20):
            if self.train:
                angle, scale = rng.uniform(-ROTATION, ROTATION), 1 + rng.uniform(-SCALE, SCALE)
                center = (p_dst + rng.uniform(-LIMIT, LIMIT, 2) if reverse or rng.random() < P_UNIFORM
                          else case['p_anat'] + rng.normal(0, JITTER, 2))
            M_prior = crop_affine(center, (PRIOR_SIZE, PRIOR_SIZE), scale, angle)
            target = apply_affine(invert_affine(M_prior), p_dst)[0]
            if not self.train or np.all((target >= 0) & (target <= PRIOR_SIZE - 1)):
                break

        current = crop(src, crop_affine(p_src, (CURRENT_SIZE, CURRENT_SIZE)), CURRENT_SIZE)
        prior = crop(dst, M_prior, PRIOR_SIZE)
        if self.train:
            current, prior = (augment_intensity(image[None])[0] for image in (current, prior))

        item = {'current': (current - MEAN) / STD, 'prior': (prior - MEAN) / STD,
                'target_xy': torch.tensor(target, dtype=torch.float32), 'pixel_spacing': torch.full((2,), SPACING_MM)}
        if not self.train:
            item |= {'M_prior': torch.from_numpy(M_prior),
                     **{key: torch.from_numpy(case[key]) for key in ('p_anat', 'centroid', 'p_gt')}}

        return item


def main(
    data_dir: Annotated[Path, Option(help='Output folder of scripts/misc/prepare_embed_dataset.py')] = Path(DATA_DIR),
    runs_dir: Annotated[Path, Option(help='Where the logs and checkpoints go')] = Path('./runs/cross_cosine_nn'),
    output_dir: Annotated[Path, Option(help='Where predictions.csv goes')] = Path('./outputs/cross_cosine_nn'),
    batch_size: Annotated[int, Option(help='Pairs per batch')] = 32,
    max_epochs: Annotated[int, Option(help='Epochs, each one window per train pair')] = 50,
    learning_rate: Annotated[float, Option(help='AdamW learning rate')] = 1e-4,
    weight_decay: Annotated[float, Option(help='AdamW weight decay')] = 1e-4,
    workers: Annotated[int, Option(help='Parallel pairs for the anatomy, and DataLoader workers')] = 16,
    gpu: Annotated[int, Option(help='GPU index to train on')] = 0,
):
    pairs, anat, cases = load_cases(data_dir, workers)
    items = list(zip(pairs.current_path, pairs.prior_path, map(centre, pairs[[f'current_{c}' for c in BOX]].values),
                     cases))
    split = lambda name: [item for item, group in zip(items, pairs.split_group) if group == str(name)]
    prior = AnatomicalPrior.fit(local_residuals_mm([item[3] for item in split(Split.TRAIN)], SPACING_MM))
    print('prior:', prior.describe())

    module = LocalAppearanceMatcher(lr=learning_rate, weight_decay=weight_decay, spacing_mm=SPACING_MM,
                                    anatomical_prior=prior)
    train = DataLoader(PairDataset(split(Split.TRAIN), train=True), batch_size, shuffle=True, num_workers=workers)
    dev = DataLoader(PairDataset(split(Split.DEV), train=False), batch_size, num_workers=workers)
    best = ModelCheckpoint(monitor='val/fused_tmean30', mode='min')
    trainer = L.Trainer(
        default_root_dir=runs_dir,
        devices=[gpu],
        max_epochs=max_epochs,
        precision='16-mixed',
        gradient_clip_val=1.0,
        callbacks=[best, ModelCheckpoint(filename='last')],
    )
    trainer.fit(module, train, dev)

    print(f'Predicting every pair with {best.best_model_path}')
    device = f'cuda:{gpu}'
    module = LocalAppearanceMatcher.load_from_checkpoint(best.best_model_path).eval().to(device)
    points = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for batch in tqdm(DataLoader(PairDataset(items, train=False), batch_size, num_workers=workers),
                          desc='predict', unit='batch'):
            logits = module(batch['current'].to(device), batch['prior'].to(device)).float().cpu().numpy()
            for k, score in enumerate(logits):
                case = {key: batch[key][k].numpy() for key in ('p_anat', 'centroid', 'p_gt')}
                points.append(decode_all(ScoreMap(score, batch['M_prior'][k].numpy()), prior, case, SPACING_MM,
                                         FUSION_TAUS))
    tau, chosen = report(pairs, cases, points)
    write_predictions(pairs, anat, cases, chosen, tau, output_dir)


if __name__ == '__main__':
    typer.run(main)
