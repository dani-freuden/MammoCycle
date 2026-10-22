"""Which change to the synthetic pairs makes them as hard as real ones: knobs 2-4 of DeformConfig, each on top of the
current defaults (knob 1), then all together.

For each setting, the phase-1 matcher and NCC are scored by appearance only on 500 frozen synthetic dev pairs, as
measure_gap.py scores them. On the real annotated train pairs (measure_gap.py) the matcher has a median of 10.8 mm
and 31% within 5 mm, and NCC 7.6 mm and 40%.

Usage (from the repo root):
    python -m experimental.mixturemodel.lesion_matching.ablate_deform --checkpoint outputs/stepD_selfsup/best.pt
"""
import sys
from functools import partial
from pathlib import Path
from typing import Annotated

import torch
import typer
from torch.utils.data import DataLoader
from typer import Option

from baselines.anatomical_coordinates.predict import DATA_DIR
from src.enums import Split

from .measure_gap import ncc_error_mm
from .train_selfsup import ALL_IMAGES_DIR, box_sizes_mm, load_record, unannotated_images
from .train_stepD import CFG

sys.path.insert(0, str(Path(__file__).parent))  # the siblings import each other as top-level modules
from b5_prior_fusion import print_table, summarize
from selfsup import DeformConfig, SingleImageSSLDataset, grid_error_mm
from stepD_matcher import SiameseMatcher


SUPERPOSITION = dict(p_superpos=0.9, superpos=(0.2, 0.5))
GEOMETRY = dict(elastic_mm=6.0, rot_deg=15.0, scale_range=0.2)
CHANGE = dict(bias_field=0.5, p_fade=0.3, lesion_scale=(0.6, 1.6), p_lesion_scale=0.7)
SETTINGS = {'current defaults (knob 1)': {},
            '+ 2 superposition': SUPERPOSITION,
            '+ 3 geometry': GEOMETRY,
            '+ 4 tissue/lesion change': CHANGE,
            '+ 2, 3 and 4': SUPERPOSITION | GEOMETRY | CHANGE}


def main(
    checkpoint: Annotated[Path, Option(help='best.pt of train_selfsup')],
    data_dir: Annotated[Path, Option(help='The annotated set: output folder of scripts/misc/prepare_embed_dataset.py')]
        = Path(DATA_DIR),
    all_images_dir: Annotated[Path, Option(help='Output folder of prepare_embed_dataset.py --all-images')]
        = ALL_IMAGES_DIR,
    samples: Annotated[int, Option(help='Frozen synthetic dev pairs per setting')] = 500,
    gpu: Annotated[int, Option(help='CUDA device')] = 0,
    workers: Annotated[int, Option(help='Data loader workers making the pairs')] = 24,
):
    device = f'cuda:{gpu}'
    model = SiameseMatcher(CFG)
    model.load_state_dict(torch.load(checkpoint)['model'])
    paths = unannotated_images(all_images_dir, data_dir, Split.DEV)
    sizes = box_sizes_mm(data_dir)

    rows = []
    for name, overrides in SETTINGS.items():
        dataset = SingleImageSSLDataset(len(paths), partial(load_record, paths), CFG, DeformConfig(**overrides),
                                        epoch_len=samples, box_sizes_mm=sizes, fixed_seed=123)
        pairs = list(DataLoader(dataset, batch_size=None, num_workers=workers))  # in order, so the same pairs each run
        rows += [(f'{name}: matcher', summarize(grid_error_mm(model, pairs, CFG, device))),
                 (f'{name}: NCC', summarize([ncc_error_mm(pair) for pair in pairs]))]
        print(f'{name}: done', flush=True)

    print('\nreal annotated train pairs (measure_gap.py): matcher median 10.79 mm, <5mm 0.31; '
          'NCC median 7.64 mm, <5mm 0.40\n')
    print_table(rows)


if __name__ == '__main__':
    typer.run(main)
