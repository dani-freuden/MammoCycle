import random
from pathlib import Path

import pandas as pd
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from src.enums import Column, Split, View

from .augmentations import FrameAugmentation
from .dataset import CycleDataset
from .records import Record, load_records, to_pairs
from .sampler import LengthBatchSampler


STAGE_SPLITS = {
    'fit': [Split.TRAIN, Split.DEV],
    'validate': [Split.DEV],
    'test': [Split.TEST],
}


class CycleDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        height: int = 1024,
        width: int = 640,
        batch_size: int = 8,
        max_exams: int = 4,
        augmentation: FrameAugmentation | None = None,
        num_workers: int = 8,
        max_records: int | None = None,
        view: View | None = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.max_exams = max_exams
        self.augmentation = augmentation
        self.num_workers = num_workers
        self.max_records = max_records
        self.view = view  # None keeps every view

        self._datasets: dict[Split, CycleDataset] = {}

    def setup(self, stage: str):
        if stage not in STAGE_SPLITS:
            raise ValueError(f'Unsupported {stage = }, expected one of {list(STAGE_SPLITS)}')

        records = load_records(self.data_dir, STAGE_SPLITS[stage], self.view)
        if self.max_records is not None:
            for split in records:
                # A seeded random sample, the same in every run, so a subset does not follow the order of metadata.csv.
                records[split] = random.Random(0).sample(records[split], min(self.max_records, len(records[split])))

        if self.trainer is not None and self.trainer.is_global_zero:
            self._save_records(records, Path(self.trainer.log_dir) / f'{stage}_patients.csv')

        for split, split_records in records.items():
            if split == Split.TRAIN:
                self._datasets[split] = CycleDataset(split_records, self.height, self.width, self.max_exams)
                continue

            # Training cycles through whole records, but the model registers two exams, so evaluate on pairs.
            # Shuffled once, so the next sample in a batch, whose prior the different-patient error borrows, is
            # rarely another pair of the same record.
            pairs = to_pairs(split_records)
            random.Random(0).shuffle(pairs)
            self._datasets[split] = CycleDataset(pairs, self.height, self.width)

    def train_dataloader(self) -> DataLoader:
        dataset = self._datasets[Split.TRAIN]
        # Records have different numbers of exams, and the default collate can only stack equal lengths.
        lengths = [min(len(record.exams), self.max_exams) for record in dataset.records]
        # Lightning can't shard a custom batch sampler (train.py turns use_distributed_sampler off), so it shards itself.
        rank, world_size = (0, 1) if self.trainer is None else (self.trainer.global_rank, self.trainer.world_size)
        batch_sampler = LengthBatchSampler(lengths, self.batch_size, rank=rank, world_size=world_size)

        return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return self._pairs_dataloader(Split.DEV)

    def test_dataloader(self) -> DataLoader:
        return self._pairs_dataloader(Split.TEST)

    def on_after_batch_transfer(self, batch: dict[str, Tensor], dataloader_idx: int) -> dict[str, Tensor]:
        # On the device, and for training batches only.
        if self.augmentation is not None and self.trainer.training:
            batch['frames'] = self.augmentation(batch['frames'])

        return batch

    @staticmethod
    def _save_records(records: dict[Split, list[Record]], path: Path):
        # One row per record, not per patient: max_records samples records, so a patient can be in the run with only
        # some of their breasts and views. The columns are named as in metadata.csv, to merge back into it.
        columns = [str(c) for c in (Column.SPLIT_GROUP, Column.PATIENT_ID, Column.LATERALITY, Column.VIEW)]
        rows = [(split, record.patient_id, record.side, record.view)
                for split, split_records in records.items() for record in split_records]
        pd.DataFrame(rows, columns=columns).to_csv(path, index=False)

    def _pairs_dataloader(self, split: Split) -> DataLoader:
        return DataLoader(
            self._datasets[split],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
