from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from .records import Record


class CycleDataset(Dataset):
    """Each item is every exam of one record, in date order.

    Returns:
        {'frames': Tensor shaped [len(record.exams), 1, image_size, image_size], values in [0, 1]}
    """

    def __init__(self, records: list[Record], image_size: int):
        self.records = records
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        frames = [self._load(exam.path) for exam in self.records[index].exams]

        return {'frames': torch.stack(frames)}

    def _load(self, path: Path) -> Tensor:
        # The prepared PNGs are 8-bit grayscale, 1664x2048, every breast facing left.
        with Image.open(path) as image:
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)

        return pil_to_tensor(image).float() / 255
