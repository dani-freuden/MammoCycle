"""Train the Step D matcher (stepD_matcher.SiameseMatcher) on the EMBED CC lesion pairs, then predict every pair.

Same pairs, anatomy boxes and prior fusion as predict.py (B5), with the network's log p(cell | appearance) in place of
NCC. The PNGs are resampled once from predict.SPACING_MM to CFG.spacing_mm and kept in memory with their breast masks.
Train pairs are used in both directions, since both boxes are known. After every epoch the dev pairs are scored with
the search window on p_anat and fused with the prior over TAUS; the epoch with the best truncated mean at its best tau
is saved to output_dir/best.pt. That model then predicts every pair, tau is chosen on dev again, and predictions.csv
is written as predict.py writes it. The network starts from ImageNet, or with --init from a best.pt (e.g. a
self-supervised one), whose MatcherConfig (kernel, search window) it then keeps; --checkpoint keeps its own too.

Usage (from the repo root):
    python -m experimental.mixturemodel.lesion_matching.train_stepD --gpu 0 --workers 32
    python -m baselines.anatomical_coordinates.evaluate --output-dir outputs/stepD_matcher
    python -m experimental.mixturemodel.lesion_matching.train_stepD --init outputs/stepD_selfsup_k31/best.pt \
        --output-dir outputs/stepD_k31_finetune --gpu 0 --workers 32
"""
import sys
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import typer
from PIL import Image
from tqdm import tqdm
from typer import Option

from baselines.anatomical_coordinates.predict import BOX, DATA_DIR
from src.data.dataset import breast_mask
from src.enums import Split

from .predict import SPACING_MM, centre, decode_all, load_cases, report, write_predictions

sys.path.insert(0, str(Path(__file__).parent))  # stepD_matcher imports its siblings as top-level modules
from b5_prior_fusion import AnatomicalPrior, choose_tau, local_residuals_mm, print_table, sweep_tau
from geometry import rescale_points, resample_to_spacing
from stepD_matcher import MatcherConfig, PairDataset, SiameseMatcher, score_maps, train_matcher


CFG = MatcherConfig()
SCALE = SPACING_MM / CFG.spacing_mm  # working pixels per PNG pixel
TAUS = np.logspace(-1, 1, 9)  # the score is log p(cell), so tau = 1 adds it to the log prior as it is


def load_image(path: str) -> tuple:
    """The PNG at CFG.spacing_mm, and its breast mask."""
    image, _ = resample_to_spacing(np.asarray(Image.open(path)), SPACING_MM, CFG.spacing_mm)
    return image, breast_mask(image)


def matcher_cases(pairs, cases: list, images: dict) -> list:
    """predict.load_cases' cases in working pixels, with the images, boxes and prior mask PairDataset and the fusion
    need."""
    size = lambda box: SCALE * (box[[3, 2]] - box[[1, 0]])  # (w, h)

    return [{'img_cur': images[current_path][0], 'img_prior': images[prior_path][0],
             'mask_prior': images[prior_path][1], 'p_cur': rescale_points(centre(current_box), SCALE),
             'box_wh': size(current_box), 'box_wh_prior': size(prior_box),
             **{key: rescale_points(case[key], SCALE) for key in ('p_anat', 'centroid', 'p_gt')}}
            for case, current_path, prior_path, current_box, prior_box
            in zip(cases, pairs.current_path, pairs.prior_path, pairs[[f'current_{c}' for c in BOX]].values,
                   pairs[[f'prior_{c}' for c in BOX]].values)]


def main(
    data_dir: Annotated[Path, Option(help='Output folder of scripts/misc/prepare_embed_dataset.py')] = Path(DATA_DIR),
    output_dir: Annotated[Path, Option(help='Where best.pt and predictions.csv go')]
        = Path('./outputs/stepD_matcher'),
    gpu: Annotated[int, Option(help='CUDA device')] = 0,
    epochs: Annotated[int, Option(help='Epochs, each one window per train pair')] = 30,
    batch_size: Annotated[int, Option(help='Pairs per batch')] = 16,
    learning_rate: Annotated[float, Option(help='Peak learning rate of the head; the backbone gets 0.1 of it')]
        = 3e-4,
    workers: Annotated[int, Option(help='Parallel pairs for the anatomy, and data loader workers')] = 16,
    checkpoint: Annotated[Path | None, Option(help='Skip training and predict with this best.pt')] = None,
    init: Annotated[Path | None, Option(help='Fine-tune from this best.pt, with its MatcherConfig (e.g. '
                                             'outputs/stepD_selfsup_k31/best.pt)')] = None,
    reverse: Annotated[bool, Option(help='Know the earlier exam\'s box and predict the later one (the "current" '
                                         'columns then hold the earlier exam)')] = False,
):
    start = torch.load(checkpoint or init, map_location='cpu') if checkpoint or init else None
    cfg = MatcherConfig(**start['cfg']) if start else CFG  # the kernel and window go with the weights
    pairs, anat, cases = load_cases(data_dir, workers, reverse)
    paths = sorted(set(pairs.current_path) | set(pairs.prior_path))
    with Pool(workers) as pool:
        images = dict(zip(paths, tqdm(pool.imap(load_image, paths), total=len(paths), desc='images', unit='image')))
    matcher = matcher_cases(pairs, cases, images)
    train = [c for c, split in zip(matcher, pairs.split_group) if split == str(Split.TRAIN)]
    dev = [c for c, split in zip(matcher, pairs.split_group) if split == str(Split.DEV)]
    prior = AnatomicalPrior.fit(local_residuals_mm(train, CFG.spacing_mm))
    print('prior:', prior.describe())

    device = f'cuda:{gpu}'
    output_dir.mkdir(parents=True, exist_ok=True)
    best = {'tmean30': np.inf}

    def validate(model, epoch):
        for case, smap in zip(dev, score_maps(model, PairDataset(dev, cfg, train=False), device)):
            case['smap'] = smap
        rows = sweep_tau(dev, prior, TAUS, CFG.spacing_mm)
        print(f'\nepoch {epoch}, dev')
        print_table(rows)
        tau = choose_tau(rows)
        tmean30 = dict(rows)[f'fused tau={tau:.4g}']['tmean30']
        if tmean30 < best['tmean30']:
            best.update(tmean30=tmean30, epoch=epoch)
            torch.save({'model': model.state_dict(), 'cfg': asdict(cfg), 'epoch': epoch, 'tau': tau},
                       output_dir / 'best.pt')
        print(f'best: epoch {best["epoch"]}, tmean30 {best["tmean30"]:.2f} mm')

    model = SiameseMatcher(cfg)
    if init:
        model.load_state_dict(start['model'])
    if checkpoint is None:
        train_matcher(model, PairDataset(train, cfg, train=True), None, epochs=epochs, batch_size=batch_size,
                      lr=learning_rate, device=device, num_workers=workers, val_fn=validate)
    model.load_state_dict(torch.load(checkpoint or output_dir / 'best.pt')['model'])
    smaps = score_maps(model, PairDataset(matcher, cfg, train=False), device)
    points = [{name: rescale_points(p, 1 / SCALE) for name, p in decode_all(smap, prior, case, CFG.spacing_mm,
                                                                            TAUS).items()}
              for smap, case in zip(tqdm(smaps, desc='decode', unit='pair'), matcher)]
    tau, chosen = report(pairs, cases, points)
    write_predictions(pairs, anat, cases, chosen, tau, output_dir)


if __name__ == '__main__':
    typer.run(main)
