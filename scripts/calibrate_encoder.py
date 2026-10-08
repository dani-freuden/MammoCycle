"""Calibrate the encoder's output normalization on training frames and save the encoder.

The trunk's features are all non-negative, so before calibration any two tissue cells have a high cosine similarity;
after it, unrelated cells should be near orthogonal. Calibrate once per frame size, then point the config's
encoder.calibrated at the saved file.

Usage:
    python scripts/calibrate_encoder.py --config overfit --out runs/calibration/mirai_512x320.pt
"""
import os
from pathlib import Path
from typing import Annotated

import torch
import torch.nn.functional as F
import typer
from dotenv import load_dotenv
from torch import Tensor
from typer import Option

from src.config import Config
from src.data.module import CycleDataModule
from src.models.encoder import ResNetEncoder, build_encoder

load_dotenv()
DATA_DIR = os.getenv('DATA_DIR')


@torch.no_grad()
def tissue_cell_similarity(encoder: ResNetEncoder, images: Tensor, threshold: float, pairs: int = 100_000) -> float:
    """Mean cosine similarity between random pairs of tissue cells of images [N, 1, H, W]."""
    features = F.normalize(encoder(images), dim=1).permute(0, 2, 3, 1)  # [N, h, w, C]
    tissue = F.avg_pool2d((images > threshold).float(), encoder.stride)[:, 0] > 0.5
    cells = features[tissue]
    first, second = torch.randint(len(cells), (2, pairs), generator=torch.Generator().manual_seed(0))

    return (cells[first] * cells[second]).sum(dim=1).mean().item()


def main(
    config: Annotated[str, Option(prompt=True, help='Name of a config in config/, for its data and encoder')],
    out: Annotated[Path, Option(prompt=True, help='Where to save the calibrated encoder')],
    data_dir: Annotated[str, Option(prompt=True, help='Path to the prepared data directory')] = DATA_DIR,
    num_frames: Annotated[int, Option(prompt=True, help='Training frames to calibrate on')] = 512,
    batch_size: Annotated[int, Option(prompt=True, help='Records per batch')] = 8,
    num_workers: Annotated[int, Option(prompt=True, help='DataLoader workers')] = 8,
    device: Annotated[str, Option(prompt=True, help='Device to run the encoder on')] = 'cuda',
):
    cfg = Config.from_name(config)
    encoder = build_encoder(cfg.encoder.model_copy(update={'calibrated': None})).to(device)
    data = CycleDataModule(data_dir, cfg.data.height, cfg.data.width, batch_size, num_workers=num_workers)
    data.setup('fit')

    batches, count = [], 0
    for batch in data.train_dataloader():
        batches.append(batch['frames'].flatten(0, 1).to(device))
        count += len(batches[-1])
        if count >= num_frames:
            break

    threshold = cfg.data.tissue_threshold
    before = tissue_cell_similarity(encoder, batches[0], threshold)
    encoder.calibrate(batches, threshold)
    after = tissue_cell_similarity(encoder, batches[0], threshold)
    print(f'Calibrated on {count} frames. Mean cosine similarity of random tissue cells: {before:.3f} -> {after:.3f}')

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), out)
    print(f'Saved to {out}')


if __name__ == '__main__':
    typer.run(main)
