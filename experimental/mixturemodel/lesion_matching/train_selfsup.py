"""Phase 1 of the self-supervised Step D matcher: train stepD_matcher.SiameseMatcher on synthetic pairs made from
single unannotated images (selfsup.SingleImageSSLDataset), with no annotated pair, then score the annotated dev and
test pairs.

Training images: the CC images of all_images_dir (scripts/misc/prepare_embed_dataset.py --all-images) of patients
outside the annotated set in data_dir, their train split. They are read, resampled from src.constants.SPACING_MM to
CFG.spacing_mm and masked on the fly. The synthetic lesions' box sizes are drawn from the distinct boxes of the
annotated train-split pairs. A frozen synthetic set from the same patients' dev split picks the epoch, by the median
argmax error in mm; the best one is saved to output_dir/best.pt, with its MatcherConfig. ssl_examples.png shows a
few training pairs. --kernel-cells and --search-px set the matcher's correlation kernel and search window (the
defaults are stepD_matcher's: 9 cells, 14.4 mm, in 512 px); the search range is (search_px / stride - kernel_cells)
/ 2 cells each way, so a larger kernel needs a larger window to keep it.

Evaluation, as in train_stepD (window on p_anat): the anatomical prior is fitted and tau chosen on the annotated
train-split pairs, which phase 1 does not train on; the dev and test pairs are then reported as the anatomical point,
the prior mode, appearance only (no labels used at all) and fused, and these tables are saved to tables.csv.
predictions.csv holds the fused points of the dev and test pairs, written as predict.py writes it.

Usage (from the repo root):
    python -m experimental.mixturemodel.lesion_matching.train_selfsup --gpu 0 --workers 32
    python -m experimental.mixturemodel.lesion_matching.train_selfsup --kernel-cells 31 --search-px 688 \
        --output-dir outputs/stepD_selfsup_k31
    python -m experimental.mixturemodel.lesion_matching.train_selfsup --checkpoint outputs/stepD_selfsup/best.pt
    python -m baselines.anatomical_coordinates.evaluate --output-dir outputs/stepD_selfsup
"""
from dataclasses import asdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import torch
import typer
from tqdm import tqdm
from typer import Option

from src.constants import BOX, EMBED_ALL_DIR, EMBED_DATA_DIR, SPACING_MM, Split, View
from src.data.lesion_pairs import Reverse, lesion_pairs

from .b5_prior_fusion import AnatomicalPrior, choose_tau, local_residuals_mm, print_table, sweep_tau
from .geometry import rescale_points
from .predict import decode_all, load_cases, write_predictions
from .selfsup import SingleImageSSLDataset, freeze, grid_error_mm, save_examples
from .stepD_matcher import MatcherConfig, PairDataset, SiameseMatcher, score_maps, train_matcher
from .train_stepD import CFG, SCALE, TAUS, load_image, matcher_cases


EVALUATED = (str(Split.DEV), str(Split.TEST))


def unannotated_images(all_images_dir: Path, data_dir: Path, split: Split) -> list[str]:
    """The CC images of all_images_dir's split whose patients are not in the annotated set of data_dir."""
    annotated = set(pd.read_csv(data_dir / 'metadata.csv', dtype={'patient_id': str}).patient_id)
    metadata = pd.read_csv(all_images_dir / 'metadata.csv', dtype={'patient_id': str})
    keep = (metadata.view.eq(str(View.CC)) & metadata.split_group.eq(str(split))
            & ~metadata.patient_id.isin(annotated))
    return [str(all_images_dir / file) for file in metadata.file_path[keep]]


def load_record(paths: list[str], index: int) -> dict:
    """SingleImageSSLDataset's record of one image: the image at CFG.spacing_mm and its breast mask."""
    image, mask = load_image(paths[index])
    return {'img': image, 'mask': mask}


def box_sizes_mm(data_dir: Path) -> np.ndarray:
    """The larger side, in mm at SPACING_MM, of every distinct box of the annotated train-split pairs."""
    pairs = lesion_pairs(data_dir, [Split.TRAIN])
    boxes = pd.concat([pairs[[f'{exam}_path', *(f'{exam}_{c}' for c in BOX)]].set_axis(['path', *BOX], axis=1)
                       for exam in ('current', 'prior')]).drop_duplicates()
    return np.maximum(boxes.xmax - boxes.xmin, boxes.ymax - boxes.ymin).values * SPACING_MM


def train(cfg: MatcherConfig, all_images_dir: Path, data_dir: Path, output_dir: Path, device: str, epochs: int,
          epoch_len: int, batch_size: int, learning_rate: float, val_samples: int, workers: int) -> SiameseMatcher:
    """Train on synthetic pairs and return the model of the epoch with the lowest synthetic dev median error."""
    sizes = box_sizes_mm(data_dir)
    datasets = {}
    for split in (Split.TRAIN, Split.DEV):
        paths = unannotated_images(all_images_dir, data_dir, split)
        print(f'{split}: {len(paths)} unannotated CC images')
        datasets[split] = partial(SingleImageSSLDataset, len(paths), partial(load_record, paths), cfg,
                                  box_sizes_mm=sizes)
    print(f'{len(sizes)} annotated train boxes, larger side p5/50/95',
          np.percentile(sizes, [5, 50, 95]).round(1), 'mm')
    train_set = datasets[Split.TRAIN](epoch_len=epoch_len)
    save_examples(train_set, str(output_dir / 'ssl_examples.png'))
    val = freeze(datasets[Split.DEV](fixed_seed=123), val_samples)

    best = {'median_mm': np.inf}

    def validate(model, epoch):
        median = float(np.median(grid_error_mm(model, val, cfg, device)))
        if median < best['median_mm']:
            best.update(median_mm=median, epoch=epoch)
            torch.save({'model': model.state_dict(), 'cfg': asdict(cfg), 'epoch': epoch, 'median_mm': median},
                       output_dir / 'best.pt')
        print(f'\nepoch {epoch}: synthetic dev median {median:.2f} mm; best epoch {best["epoch"]}, '
              f'{best["median_mm"]:.2f} mm')

    model = SiameseMatcher(cfg)
    train_matcher(model, train_set, None, epochs=epochs, batch_size=batch_size, lr=learning_rate, device=device,
                  num_workers=workers, val_fn=validate)
    model.load_state_dict(torch.load(output_dir / 'best.pt')['model'])

    return model


def evaluate(model: SiameseMatcher, data_dir: Path, output_dir: Path, device: str, workers: int, reverse: bool):
    """Fit the prior and choose tau on the annotated train-split pairs, report dev and test, and write predictions.csv
    for them. reverse as in lesion_pairs."""
    pairs, anat, cases = load_cases(data_dir, workers, reverse)
    paths = sorted(set(pairs.current_path) | set(pairs.prior_path))
    with Pool(workers) as pool:
        images = dict(zip(paths, tqdm(pool.imap(load_image, paths), total=len(paths), desc='images', unit='image')))
    matcher = matcher_cases(pairs, cases, images)
    for case, smap in zip(matcher, score_maps(model, PairDataset(matcher, model.cfg, train=False), device)):
        case['smap'] = smap

    split = pairs.split_group.values
    train_cases = [c for c, s in zip(matcher, split) if s == str(Split.TRAIN)]
    prior = AnatomicalPrior.fit(local_residuals_mm(train_cases, CFG.spacing_mm))
    print('prior (train pairs):', prior.describe())
    print('\nTRAIN, choosing tau')
    rows = sweep_tau(train_cases, prior, TAUS, CFG.spacing_mm)
    print_table(rows)
    tau = choose_tau(rows)
    tables = [(str(Split.TRAIN), rows)]
    for name in EVALUATED:
        print(f'\n{name.upper()}, tau {tau:.4g}')
        tables.append((name, sweep_tau([c for c, s in zip(matcher, split) if s == name], prior, [tau], CFG.spacing_mm)))
        print_table(tables[-1][1])
    pd.DataFrame([{'split': name, 'method': method, **stats} for name, rows in tables for method, stats in rows]
                 ).to_csv(output_dir / 'tables.csv', index=False)

    evaluated = np.isin(split, EVALUATED)
    points = np.array([rescale_points(decode_all(c['smap'], prior, c, CFG.spacing_mm, [tau])[f'fused tau={tau:.4g}'],
                                      1 / SCALE) for c, e in zip(matcher, evaluated) if e])
    write_predictions(pairs[evaluated].reset_index(drop=True), anat[evaluated],
                      [c for c, e in zip(cases, evaluated) if e], points, tau, output_dir)


def main(
    data_dir: Annotated[Path, Option(help='The annotated set: output folder of scripts/misc/prepare_embed_dataset.py')]
        = EMBED_DATA_DIR,
    all_images_dir: Annotated[Path, Option(help='Output folder of prepare_embed_dataset.py --all-images')]
        = EMBED_ALL_DIR,
    output_dir: Annotated[Path, Option(help='Where best.pt, ssl_examples.png and predictions.csv go')]
        = Path('./outputs/stepD_selfsup'),
    checkpoint: Annotated[Path | None, Option(help='Skip training and evaluate this best.pt')] = None,
    gpu: Annotated[int, Option(help='CUDA device')] = 0,
    epochs: Annotated[int, Option(help='Epochs of synthetic pairs')] = 5,
    epoch_len: Annotated[int, Option(help='Synthetic pairs per epoch')] = 200_000,
    batch_size: Annotated[int, Option(help='Pairs per batch')] = 16,
    learning_rate: Annotated[float, Option(help='Peak learning rate of the head; the backbone gets 0.1 of it')]
        = 3e-4,
    val_samples: Annotated[int, Option(help='Frozen synthetic dev pairs that pick the epoch')] = 500,
    kernel_cells: Annotated[int, Option(help="Correlation kernel, in cells of 1.6 mm (ignored with --checkpoint, "
                                             "which keeps its own)")] = MatcherConfig.kernel_cells,
    search_px: Annotated[int, Option(help='Search window, in pixels of 0.2 mm (ignored with --checkpoint)')]
        = MatcherConfig.search_px,
    workers: Annotated[int, Option(help='Parallel pairs for the anatomy, and data loader workers')] = 16,
    reverse: Reverse = False,
):
    device = f'cuda:{gpu}'
    output_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint is None:
        cfg = MatcherConfig(kernel_cells=kernel_cells, search_px=search_px)
        print(f'kernel {cfg.kernel_cells} cells ({cfg.kernel_cells * cfg.cell_mm:.1f} mm), search window '
              f'{cfg.search_px} px, range +-{cfg.max_offset_mm:.1f} mm')
        model = train(cfg, all_images_dir, data_dir, output_dir, device, epochs, epoch_len, batch_size,
                      learning_rate, val_samples, workers)
    else:
        state = torch.load(checkpoint)
        model = SiameseMatcher(MatcherConfig(**state['cfg']))
        model.load_state_dict(state['model'])
    evaluate(model, data_dir, output_dir, device, workers, reverse)


if __name__ == '__main__':
    typer.run(main)
