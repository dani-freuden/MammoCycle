"""Does the matcher fail on real pairs because it compares too small a patch around the lesion? Two checks, no training.

The matcher correlates only the central kernel of its template, KERNEL_PX (kernel_cells x stride, 14.4 mm), while
NCC's template is 1.5 x the box (12 mm up to the whole 51 mm template crop). On real pairs NCC gets better with lesion
size and the matcher does not (diagnose_gap). In a prior the lesion itself may be fainter or not there yet, so the
tissue around it may be what finds it.

1. Lesion removed. Synthetic pairs around the real lesions of the annotated current images of the train-split pairs,
   with the current DeformConfig, and with the lesion faded out of the search window every time (fade 0.9-1.0). If
   the matcher collapses there while the large NCC holds up, the generator reproduces the real failure.
2. Template size on real pairs. NCC on the real train pairs with a KERNEL_PX template and with the whole template
   crop, beside the matcher and the adaptive NCC of diagnose_gap's real_pairs.csv, by box size. If the small NCC goes
   flat like the matcher and the large one holds up, the patch size is the cause.

Both score appearance only, decoded over the matcher's grid, as measure_gap.py does.

Usage (from the repo root), after diagnose_gap.py:
    python -m experimental.mixturemodel.lesion_matching.diagnose_context --checkpoint outputs/stepD_selfsup/best.pt
"""
import sys
from functools import partial
from pathlib import Path
from typing import Annotated

import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm
from typer import Option

from .diagnose_gap import HIT_MM, lesion_boxes, load_annotated, train_pairs
from .measure_gap import ncc_error_mm, real_cases
from .train_stepD import CFG

sys.path.insert(0, str(Path(__file__).parent))  # the siblings import each other as top-level modules
from b5_prior_fusion import print_table, summarize
from selfsup import DeformConfig, SingleImageSSLDataset, grid_error_mm
from stepD_matcher import PairDataset, SiameseMatcher


KERNEL_PX = CFG.kernel_cells * CFG.stride  # the template area the matcher correlates
NCC_TEMPLATES = {f'NCC {KERNEL_PX * CFG.spacing_mm:.1f} mm': KERNEL_PX, 'NCC 1.5 x box': None,
                 f'NCC {CFG.template_px * CFG.spacing_mm:.0f} mm': CFG.template_px}
LESION_REMOVED = dict(p_fade=1.0, fade=(0.9, 1.0))


def lesion_removed_check(model: SiameseMatcher, pairs: pd.DataFrame, samples: int, device: str, workers: int):
    """Check 1: synthetic pairs around the real lesions, with the lesion kept as usual or removed from the search."""
    paths, boxes = lesion_boxes(pairs)
    rows = []
    for name, overrides in (('lesion as usual', {}), ('lesion removed', LESION_REMOVED)):
        dataset = SingleImageSSLDataset(len(paths), partial(load_annotated, paths, boxes), CFG,
                                        DeformConfig(p_real_box=1.0, **overrides), epoch_len=samples, fixed_seed=123)
        synthetic = list(DataLoader(dataset, batch_size=None, num_workers=workers))
        rows.append((f'{name}: matcher', summarize(grid_error_mm(model, synthetic, CFG, device))))
        rows += [(f'{name}: {label}', summarize([ncc_error_mm(s, side) for s in synthetic]))
                 for label, side in NCC_TEMPLATES.items()]
        print(f'{name}: done', flush=True)
    print(f'\nCHECK 1: synthetic pairs around the real lesions ({samples} each)')
    print_table(rows)


def template_size_check(real_pairs_csv: Path, predictions: Path, workers: int):
    """Check 2: the share of real train pairs each method gets within HIT_MM, by box size."""
    rows = pd.read_csv(real_pairs_csv)
    real = PairDataset(real_cases(predictions, workers), CFG, train=False)
    assert len(real) == len(rows), f'{real_pairs_csv} has {len(rows)} pairs, {predictions} {len(real)} train pairs'
    errors = {label: [] for label, side in NCC_TEMPLATES.items() if side is not None}
    # by index: PairDataset wraps its index around, so iterating it directly never ends
    for i in tqdm(range(len(real)), desc='NCC', unit='pair'):
        sample = real[i]
        for label in errors:
            errors[label].append(ncc_error_mm(sample, NCC_TEMPLATES[label]))
    methods = {'matcher': rows.matcher_mm, 'NCC 1.5 x box': rows.ncc_mm, **errors}
    hits = pd.DataFrame({label: pd.Series(e) < HIT_MM for label, e in methods.items()})
    hits = hits[['matcher', *NCC_TEMPLATES]]
    size = pd.qcut(rows.box_mm, 6)
    table = hits.groupby(size, observed=True).mean()
    table.insert(0, 'pairs', size.value_counts().sort_index().values)
    table.index = table.index.astype(str)
    table.loc['all'] = [len(hits), *hits.mean()]
    table['pairs'] = table.pairs.astype(int)
    print(f'\nCHECK 2: {len(rows)} real train pairs, share within {HIT_MM:g} mm, by box size (mm)')
    print(table.round(2).to_string())


def main(
    checkpoint: Annotated[Path, Option(help='best.pt of train_selfsup')],
    predictions: Annotated[Path, Option(help='predictions.csv with every pair and its anatomy box (anat_*)')]
        = Path('./outputs/b5_prior_fusion/predictions.csv'),
    real_pairs: Annotated[Path, Option(help="diagnose_gap's real_pairs.csv, from the same predictions.csv")]
        = Path('./outputs/stepD_selfsup/diagnosis/real_pairs.csv'),
    samples: Annotated[int, Option(help='Synthetic pairs per setting in check 1')] = 500,
    gpu: Annotated[int, Option(help='CUDA device')] = 0,
    workers: Annotated[int, Option(help='Data loader workers and parallel image loading')] = 24,
):
    model = SiameseMatcher(CFG)
    model.load_state_dict(torch.load(checkpoint)['model'])
    lesion_removed_check(model, train_pairs(predictions), samples, f'cuda:{gpu}', workers)
    template_size_check(real_pairs, predictions, workers)


if __name__ == '__main__':
    typer.run(main)
