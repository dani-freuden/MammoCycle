"""Train the contrastive random walk (walk.py) on EMBED CC exam pairs, then score the annotated dev and test pairs.

--stage pretrain walks the unannotated pairs: every (later, earlier) CC exam pair of all_images_dir's train-split
patients outside the annotated set, with no labels. --stage finetune walks the annotated train pairs of data_dir, the
current window centred on the lesion, and also trains the lesion node's step toward the true prior centre (soft-target
CE, as Step D). Both minimize cycle + alpha * anatomy (walk.walk_losses); --init starts from another run's best.pt.

After every epoch the annotated dev pairs are scored (walk.score_map, window on p_anat) and fused with the anatomical
prior, fitted on the train pairs' residuals, over TAUS. The epoch with the best truncated mean at its best tau is saved to
output_dir/best.pt. That model then scores dev and test, tau is chosen on dev, and predictions.csv holds both, written
as predict.py writes it.

Usage (from the repo root):
    python -m experimental.mammo_walk.train --stage pretrain --gpu 0 --workers 32
    python -m experimental.mammo_walk.train --stage finetune --init outputs/mammo_walk/pretrain/best.pt --gpu 0 --workers 32
    python -m experimental.mammo_walk.train --stage finetune --checkpoint outputs/mammo_walk/finetune/best.pt
    python -m baselines.anatomical_coordinates.evaluate --output-dir outputs/mammo_walk/finetune
"""
from collections import defaultdict
from enum import auto
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from typer import Option

from baselines.anatomical_coordinates.predict import DATA_DIR
from experimental.mixturemodel.lesion_matching.predict import load_cases, write_predictions
from experimental.mixturemodel.lesion_matching.train_selfsup import ALL_IMAGES_DIR
from experimental.mixturemodel.lesion_matching.train_stepD import SCALE, TAUS, matcher_cases
from src.data.augmentations import augment_intensity
from src.data.records import load_records, to_pairs
from src.enums import Split, StringEnum, View

from .walk import (EDGE_DROPOUT, GRID, NODE_STRIDE, PNG_SPACING_MM, SPACING_MM, TARGET_SIGMA_NODES, PatchEmbedding,
                   WalkPairs, embed_patches, load_image, score_map, walk_losses)

# lesion_matching's modules, on sys.path through walk
from b5_prior_fusion import AnatomicalPrior, choose_tau, local_residuals_mm, predict, print_table, sweep_tau
from geometry import rescale_points
from stepD_matcher import make_optimizer, soft_target_ce


assert np.isclose(SCALE, PNG_SPACING_MM / SPACING_MM), 'matcher_cases must resample to walk.SPACING_MM'


class Stage(StringEnum):
    PRETRAIN = auto()
    FINETUNE = auto()


DEFAULTS = {Stage.PRETRAIN: (5, 50_000, 3e-4), Stage.FINETUNE: (20, None, 1e-4)}  # epochs, pairs per epoch, peak lr
LESION_NODE = GRID * GRID // 2  # the centre node, on p_cur


def unannotated_pairs(all_images_dir: Path, data_dir: Path) -> list:
    """Every (later, earlier) CC exam pair of all_images_dir's train-split patients outside the annotated set of
    data_dir, as WalkPairs pairs."""
    annotated = set(pd.read_csv(data_dir / 'metadata.csv', dtype={'patient_id': str}).patient_id)
    records = load_records(all_images_dir, [Split.TRAIN], View.CC)[Split.TRAIN]

    return [{'current': str(record.exams[1].path), 'prior': str(record.exams[0].path)}
            for record in to_pairs(records) if record.patient_id not in annotated]


def score_cases(model, cases: list, temperature: float, device: str):
    model.eval()
    for case in tqdm(cases, desc='score maps', unit='pair', leave=False):
        case['smap'] = score_map(model, case, temperature, device)


def train(model, train_set: WalkPairs, dev: list, prior: AnatomicalPrior, output_dir: Path, device: str, alpha: float,
          temperature: float, epochs: int, epoch_len: Optional[int], batch_size: int, learning_rate: float,
          workers: int, log_every: int = 50):
    """Train, and save the epoch with the best dev tmean30 at its best tau to output_dir/best.pt."""
    loader = DataLoader(train_set, batch_size, sampler=RandomSampler(train_set, num_samples=epoch_len), drop_last=True,
                        num_workers=workers, persistent_workers=workers > 0, pin_memory=True)
    optimizer = make_optimizer(model, learning_rate)
    total = epochs * len(loader)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[learning_rate * 0.1, learning_rate],
                                                   total_steps=total, pct_start=0.05)
    scaler = torch.amp.GradScaler()
    best, history, step = {'tmean30': np.inf}, defaultdict(list), 0

    for epoch in range(epochs):
        model.train()
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            windows = augment_intensity(torch.cat([batch['current'], batch['prior']]))
            with torch.autocast('cuda'):
                current, prior_nodes = embed_patches(model, windows, NODE_STRIDE).float().chunk(2)
            cycle, anatomy, logits = walk_losses(current, prior_nodes, batch['current_valid'], batch['prior_valid'],
                                                 batch['current_anatomy'], batch['prior_anatomy'], temperature,
                                                 EDGE_DROPOUT)
            terms = {'cycle': cycle.mean(), 'anatomy': anatomy.mean()}
            if 'target' in batch:
                terms['lesion'] = soft_target_ce(logits[:, LESION_NODE].view(-1, GRID, GRID), batch['target'],
                                                 TARGET_SIGMA_NODES, batch['prior_valid'].view(-1, GRID, GRID))
            loss = terms['cycle'] + alpha * terms['anatomy'] + terms.get('lesion', 0)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            for name, value in terms.items():
                history[name].append(value.item())
            if step % log_every == 0:
                print(f'epoch {epoch} step {step}/{total}',
                      ' '.join(f'{name} {np.mean(values[-log_every:]):.3f}' for name, values in history.items()))
            step += 1

        score_cases(model, dev, temperature, device)
        rows = sweep_tau(dev, prior, TAUS, SPACING_MM)
        print(f'\nepoch {epoch}, dev')
        print_table(rows)
        tau = choose_tau(rows)
        tmean30 = dict(rows)[f'fused tau={tau:.4g}']['tmean30']
        if tmean30 < best['tmean30']:
            best.update(tmean30=tmean30, epoch=epoch)
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'tau': tau, 'alpha': alpha,
                        'temperature': temperature}, output_dir / 'best.pt')
        print(f'best: epoch {best["epoch"]}, tmean30 {best["tmean30"]:.2f} mm\n')


def evaluate(model, pairs: pd.DataFrame, anat: np.ndarray, cases: list, matcher: list, prior: AnatomicalPrior,
             temperature: float, output_dir: Path, device: str):
    """Choose tau on dev, report dev and test, and write predictions.csv for both."""
    split = pairs.split_group.values
    evaluated = np.isin(split, [str(Split.DEV), str(Split.TEST)])
    score_cases(model, [c for c, e in zip(matcher, evaluated) if e], temperature, device)

    print('\nDEV, choosing tau')
    rows = sweep_tau([c for c, s in zip(matcher, split) if s == str(Split.DEV)], prior, TAUS, SPACING_MM)
    print_table(rows)
    tau = choose_tau(rows)
    print(f'\nTEST, tau {tau:.4g}')
    print_table(sweep_tau([c for c, s in zip(matcher, split) if s == str(Split.TEST)], prior, [tau], SPACING_MM))

    points = np.array([rescale_points(predict(c['smap'], prior, c, SPACING_MM, tau)[0], 1 / SCALE)
                       for c, e in zip(matcher, evaluated) if e])
    write_predictions(pairs[evaluated].reset_index(drop=True), anat[evaluated],
                      [c for c, e in zip(cases, evaluated) if e], points, tau, output_dir)


def main(
    stage: Annotated[Stage, Option(help='pretrain: unannotated pairs; finetune: annotated train pairs')],
    data_dir: Annotated[Path, Option(help='The annotated set: output folder of scripts/misc/prepare_embed_dataset.py')]
        = Path(DATA_DIR),
    all_images_dir: Annotated[Path, Option(help='Output folder of prepare_embed_dataset.py --all-images')]
        = ALL_IMAGES_DIR,
    output_dir: Annotated[Optional[Path], Option(help='Where best.pt and predictions.csv go; '
                                                      'default outputs/mammo_walk/<stage>')] = None,
    init: Annotated[Optional[Path], Option(help="Start from this best.pt's weights")] = None,
    checkpoint: Annotated[Optional[Path], Option(help='Skip training and evaluate this best.pt')] = None,
    alpha: Annotated[float, Option(help='Weight of the anatomical loss')] = 0.1,
    temperature: Annotated[float, Option(help='Temperature of the walk')] = 0.07,
    epochs: Annotated[Optional[int], Option(help='Default 5 for pretrain, 20 for finetune')] = None,
    epoch_len: Annotated[Optional[int], Option(help='Pairs per epoch; default 50,000 for pretrain, every pair once '
                                                    'for finetune')] = None,
    learning_rate: Annotated[Optional[float], Option(help='Peak lr of the head, the backbone gets 0.1 of it; default '
                                                          '3e-4 for pretrain, 1e-4 for finetune')] = None,
    batch_size: Annotated[int, Option(help='Pairs per batch')] = 4,
    gpu: Annotated[int, Option(help='CUDA device')] = 0,
    workers: Annotated[int, Option(help='Parallel images for loading, and data loader workers')] = 16,
    reverse: Annotated[bool, Option(help='Know the earlier exam\'s box and predict the later one (the "current" '
                                         'columns then hold the earlier exam)')] = False,
):
    device = f'cuda:{gpu}'
    output_dir = output_dir or Path('./outputs/mammo_walk') / str(stage)
    output_dir.mkdir(parents=True, exist_ok=True)
    default_epochs, default_epoch_len, default_lr = DEFAULTS[stage]

    pairs, anat, cases = load_cases(data_dir, workers, reverse)
    paths = sorted(set(pairs.current_path) | set(pairs.prior_path))
    with Pool(workers) as pool:
        images = dict(zip(paths, tqdm(pool.imap(load_image, paths), total=len(paths), desc='images', unit='image')))
    matcher = matcher_cases(pairs, cases, images)
    split = pairs.split_group.values
    prior = AnatomicalPrior.fit(local_residuals_mm([c for c, s in zip(matcher, split) if s == str(Split.TRAIN)],
                                                   SPACING_MM))
    print('prior (train pairs):', prior.describe())

    model = PatchEmbedding()
    if checkpoint or init:
        model.load_state_dict(torch.load(checkpoint or init, map_location='cpu')['model'])
    model.to(device)

    if checkpoint is None:
        if stage == Stage.PRETRAIN:
            walk_pairs, load = unannotated_pairs(all_images_dir, data_dir), load_image
        else:
            walk_pairs = [{'current': current, 'prior': prior_path, **{key: case[key] for key in ('p_cur', 'p_anat',
                                                                                                  'p_gt')}}
                          for case, current, prior_path, s
                          in zip(matcher, pairs.current_path, pairs.prior_path, split) if s == str(Split.TRAIN)]
            load = images.__getitem__
        print(f'{stage}: {len(walk_pairs)} pairs')
        train(model, WalkPairs(walk_pairs, load), [c for c, s in zip(matcher, split) if s == str(Split.DEV)], prior,
              output_dir, device, alpha, temperature, epochs or default_epochs, epoch_len or default_epoch_len,
              batch_size, learning_rate or default_lr, workers)
        model.load_state_dict(torch.load(output_dir / 'best.pt', map_location='cpu')['model'])

    evaluate(model, pairs, anat, cases, matcher, prior, temperature, output_dir, device)


if __name__ == '__main__':
    typer.run(main)
