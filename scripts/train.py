"""Train MammoCycleModule with a config from config/.

Usage:
    python scripts/train.py --config pairs
    python scripts/train.py --config pairs --gpus 0 --gpus 1   # DDP; batch_size is per GPU
    python scripts/train.py --config overfit --max-records 8   # one batch of pairs, repeated: the loss must fall
"""
import os
from typing import Annotated, List, Optional

import lightning as L
import typer
from dotenv import load_dotenv
from typer import Option

from src.config import Config
from src.data.augmentations import FrameAugmentation
from src.data.module import CycleDataModule
from src.models.encoder import build_encoder
from src.models.module import MammoCycleModule

load_dotenv()
DATA_DIR = os.getenv('DATA_DIR')


def main(
    config: Annotated[str, Option(help='Name of a config in config/')],
    data_dir: Annotated[str, Option(help='Path to the prepared data directory')] = DATA_DIR,
    max_records: Annotated[Optional[int], Option(help='Records per split, for quick runs')] = 51000,
    num_workers: Annotated[int, Option(help='DataLoader workers')] = 8,
    gpus: Annotated[List[int], Option(help='GPU indices to train on; none given uses every visible GPU')] = [],
):
    cfg = Config.from_name(config)
    module = MammoCycleModule(build_encoder(cfg.encoder), tissue_threshold=cfg.data.tissue_threshold,
                              **cfg.module.model_dump())
    augmentation = None if cfg.augmentation is None else FrameAugmentation(**cfg.augmentation.model_dump())
    data = CycleDataModule(data_dir, cfg.data.height, cfg.data.width, cfg.train.batch_size, cfg.train.max_exams,
                           augmentation, num_workers, max_records, cfg.data.view)
    
    trainer = L.Trainer(
        default_root_dir=f'runs/{config}',
        devices=gpus or 'auto',
        use_distributed_sampler=False,  # CycleDataModule shards the training batches; every GPU validates on all pairs
        max_epochs=cfg.train.max_epochs,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.gradient_clip_val,
        val_check_interval=cfg.train.val_check_interval,
        limit_val_batches=cfg.train.limit_val_batches,
    )
    trainer.fit(module, data)


if __name__ == '__main__':
    typer.run(main)
