"""Tune DeformConfig so the synthetic pairs are hard the way real pairs are, with NCC as the yardstick: no model and
no GPU.

NCC finds the lesion within 5 mm in 24% of the real train pairs with a 14.4 mm template (the matcher's kernel), in 40%
with 1.5 x the box and in 42% with 51 mm (diagnose_context, 2026-10-18): on real pairs more surrounding tissue helps,
and a small patch is matched poorly. For each setting in SETTINGS, on top of the current DeformConfig, this prints the
same three shares on synthetic pairs and their gap to the real ones (the sum of the absolute differences), saves the
table to output_dir/calibration.csv, and draws six of each setting's pairs to output_dir/<setting>.png.

NCC can only place a template where it fits in the search window, so a 51 mm template's centre moves at most REACH_PX
from the window centre. The synthetic truth is placed anywhere within +-38 mm, where that reaches only about 45% of
the pairs (98.7% of the real ones, whose truth sits near the anatomical point). So only synthetic pairs whose truth all
three templates can reach are scored: the first `samples` of them.

The pairs are made as train_selfsup makes them (unannotated dev-split CC images, random breast points, box sizes from
the annotated train pairs), frozen with the same seed for every setting. Edit SETTINGS to try others.

Usage (from the repo root):
    python -m experimental.mixturemodel.lesion_matching.calibrate_deform
"""
import sys
from functools import partial
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from torch.utils.data import DataLoader
from typer import Option

from baselines.anatomical_coordinates.predict import DATA_DIR
from src.enums import Split

from .diagnose_context import HIT_MM, NCC_TEMPLATES
from .measure_gap import ncc_error_mm, truth
from .train_selfsup import ALL_IMAGES_DIR, box_sizes_mm, load_record, unannotated_images
from .train_stepD import CFG

sys.path.insert(0, str(Path(__file__).parent))  # the siblings import each other as top-level modules
from selfsup import DeformConfig, SingleImageSSLDataset, save_examples


REAL = {'NCC 14.4 mm': 0.24, 'NCC 1.5 x box': 0.40, 'NCC 51 mm': 0.42}  # real train pairs, diagnose_context
SETTINGS = {'current defaults': {}}  # calibrated on 2026-10-18: 0.26 / 0.44 / 0.48, gap 0.13 (round 3)
REACH_PX = (CFG.search_px - CFG.template_px) / 2  # how far the largest NCC template's centre can move


def reachable(pair: dict) -> bool:
    """Whether every NCC template can be placed on the pair's true centre."""
    return bool(np.all(np.abs(truth(pair) - (CFG.search_px - 1) / 2) <= REACH_PX))


def main(
    data_dir: Annotated[Path, Option(help='The annotated set: output folder of scripts/misc/prepare_embed_dataset.py')]
        = Path(DATA_DIR),
    all_images_dir: Annotated[Path, Option(help='Output folder of prepare_embed_dataset.py --all-images')]
        = ALL_IMAGES_DIR,
    output_dir: Annotated[Path, Option(help='Where each setting\'s example pairs go')]
        = Path('./outputs/stepD_selfsup/calibration'),
    samples: Annotated[int, Option(help='Scored synthetic pairs per setting, all reachable')] = 500,
    workers: Annotated[int, Option(help='Data loader workers making the pairs')] = 24,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = unannotated_images(all_images_dir, data_dir, Split.DEV)
    sizes = box_sizes_mm(data_dir)

    table = {'real train pairs (target)': REAL | {'pairs': 2971}}
    for name, overrides in SETTINGS.items():
        # about 45% are reachable, so make 2.5 x as many; in order, so the same pairs each run
        dataset = SingleImageSSLDataset(len(paths), partial(load_record, paths), CFG, DeformConfig(**overrides),
                                        epoch_len=round(2.5 * samples), box_sizes_mm=sizes, fixed_seed=123)
        pairs = [pair for pair in DataLoader(dataset, batch_size=None, num_workers=workers) if reachable(pair)]
        pairs = pairs[:samples]
        table[name] = {label: np.mean([ncc_error_mm(pair, side) < HIT_MM for pair in pairs])
                       for label, side in NCC_TEMPLATES.items()} | {'pairs': len(pairs)}
        save_examples(dataset, str(output_dir / f'{name.replace(" + ", "_").replace(" ", "_").replace(",", "")}.png'))
        print(f'{name}: done', flush=True)

    table = pd.DataFrame(table).T
    table['pairs'] = table.pairs.astype(int)
    table['gap to real'] = (table[list(REAL)] - pd.Series(REAL)).abs().sum(axis=1)
    print(f'\nshare of pairs NCC finds within {HIT_MM:g} mm, by template size (synthetic: pairs within reach only)')
    print(table.round(2).to_string())
    table.round(3).to_csv(output_dir / 'calibration.csv')
    print(f'\nwrote {output_dir / "calibration.csv"}, and example pairs of each setting')


if __name__ == '__main__':
    typer.run(main)
