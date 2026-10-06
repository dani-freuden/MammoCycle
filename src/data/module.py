from pathlib import Path

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.enums import Split

from .dataset import CycleDataset
from .records import load_records, to_pairs
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
        image_size: int = 240,
        batch_size: int = 32,
        num_workers: int = 8,
        max_records: int | None = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_records = max_records

        self._datasets: dict[Split, CycleDataset] = {}

    def setup(self, stage: str):
        if stage not in STAGE_SPLITS:
            raise ValueError(f'Unsupported {stage = }, expected one of {list(STAGE_SPLITS)}')

        records = load_records(self.data_dir, STAGE_SPLITS[stage])
        if self.max_records is not None:
            for split in records:
                records[split] = records[split][:self.max_records]

        for split, split_records in records.items():
            if split != Split.TRAIN:
                # Training cycles through a whole record, but the model registers two exams, so evaluate on pairs.
                split_records = to_pairs(split_records)

            self._datasets[split] = CycleDataset(split_records, image_size=self.image_size)

    def train_dataloader(self) -> DataLoader:
        dataset = self._datasets[Split.TRAIN]
        # Records have different numbers of exams, and the default collate can only stack equal lengths.
        batch_sampler = LengthBatchSampler([len(record.exams) for record in dataset.records], self.batch_size)

        return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return self._pairs_dataloader(Split.DEV)

    def test_dataloader(self) -> DataLoader:
        return self._pairs_dataloader(Split.TEST)

    def _pairs_dataloader(self, split: Split) -> DataLoader:
        return DataLoader(
            self._datasets[split],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
