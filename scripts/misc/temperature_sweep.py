"""Choose the tracker temperature with the real weights, before training.

1. Identical frames: records of three copies of one exam. A perfect tracker has zero error here, so what remains is
   the pull of the many weak matches towards the frame centre. Take the largest temperature whose long cycle stays
   under about 4 px (half a cell).
2. Real pairs: the starting point of training. Same-patient cycles should close clearly better than different-patient
   ones; a low peak probability and a spread of hundreds of pixels mean the matches are dominated by the weak tail.

Usage:
    python scripts/misc/temperature_sweep.py --config pairs
"""
import os
from typing import Annotated

import pandas as pd
import torch
import typer
from dotenv import load_dotenv
from torch import Tensor
from typer import Option

from src.config import Config
from src.data.module import CycleDataModule
from src.models.encoder import ResNetEncoder, build_encoder
from src.models.module import MammoCycleModule

load_dotenv()
DATA_DIR = os.getenv('DATA_DIR')


@torch.no_grad()
def sweep(cfg: Config, encoder: ResNetEncoder, frames: Tensor, temperatures: list[float]) -> pd.DataFrame:
    """Cycle errors and localizer diagnostics of records frames [B, T, 1, H, W] for each temperature."""
    rows = []
    for temperature in temperatures:
        module_config = cfg.module.model_copy(update={'temperature': temperature})
        module = MammoCycleModule(encoder, tissue_threshold=cfg.data.tissue_threshold, **module_config.model_dump())
        output = module.cycle_step(frames, torch.Generator().manual_seed(0))
        rows.append({
            'temperature': temperature,
            **{f'cycle_error_px/{name}': value.mean().item() for name, value in output.cycle_errors.items()},
            'different_patient_error_px': module.different_patient_error(output).mean().item(),
            'similarity_loss': output.similarity_loss.item(),
            **{name: value.mean().item() for name, value in output.first_hop.diagnostics.items()},
        })

    return pd.DataFrame(rows).set_index('temperature')


def main(
    config: Annotated[str, Option(prompt=True, help='Name of a config in config/ with a calibrated encoder')],
    data_dir: Annotated[str, Option(prompt=True, help='Path to the prepared data directory')] = DATA_DIR,
    temperatures: Annotated[str, Option(prompt=True, help='Comma-separated temperatures')] = '0.1,0.07,0.05,0.03,0.02',
    num_pairs: Annotated[int, Option(prompt=True, help='Dev pairs to track')] = 16,
    num_workers: Annotated[int, Option(prompt=True, help='DataLoader workers')] = 8,
    device: Annotated[str, Option(prompt=True, help='Device to run on')] = 'cuda',
):
    cfg = Config.from_name(config)
    encoder = build_encoder(cfg.encoder).to(device).eval()
    data = CycleDataModule(data_dir, cfg.data.height, cfg.data.width, num_pairs, num_workers=num_workers)
    data.setup('validate')
    pairs = next(iter(data.val_dataloader()))['frames'].to(device)
    temperatures = [float(value) for value in temperatures.split(',')]

    with pd.option_context('display.width', 250, 'display.max_columns', None, 'display.float_format', '{:.3f}'.format):
        print('Identical frames, three copies of the newer exam of each pair:')
        print(sweep(cfg, encoder, pairs[:, -1:].expand(-1, 3, -1, -1, -1), temperatures), '\n')
        print('Real pairs:')
        print(sweep(cfg, encoder, pairs, temperatures))


if __name__ == '__main__':
    typer.run(main)
