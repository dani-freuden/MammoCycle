import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from .records import Record


class CycleDataset(Dataset):
    """Each item is the exams of one record in date order, at most max_exams of them.

    Returns:
        {'frames': Tensor shaped [num_exams, 1, height, width], values in [0, 1]}
    """

    def __init__(self, records: list[Record], height: int, width: int, max_exams: int | None = None):
        self.records = records
        self.height = height
        self.width = width
        self.max_exams = max_exams

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        exams = self.records[index].exams
        if self.max_exams is not None and len(exams) > self.max_exams:
            # A new random subset every time, not necessarily consecutive, varies the time gaps between frames.
            exams = [exams[i] for i in sorted(random.sample(range(len(exams)), self.max_exams))]

        return {'frames': torch.stack([load_frame(exam.path, self.height, self.width) for exam in exams])}


def load_frame(path: Path, height: int, width: int) -> Tensor:
    """Crop a prepared image to the breast and fit it into a height x width canvas, without stretching.

    The prepared PNGs are 8-bit with the chest wall on the right edge. The crop is the bounding box of the largest
    connected region of non-zero pixels, which leaves out the markers in the background. It is resized to fill the
    canvas height, or its width when the breast is wider than the canvas, and padded with zeros on the left (away from
    the chest wall) and at the bottom.
    """
    with Image.open(path) as image:
        image.load()

    _, _, stats, _ = cv2.connectedComponentsWithStats((np.asarray(image) > 0).astype(np.uint8))
    x, y, w, h = stats[1 + stats[1:, cv2.CC_STAT_AREA].argmax(), :4]  # label 0 is the background
    scale = min(height / h, width / w)
    breast = image.crop((x, y, x + w, y + h)).resize((round(w * scale), round(h * scale)), Image.Resampling.BILINEAR)

    canvas = Image.new('L', (width, height))
    canvas.paste(breast, (width - breast.width, 0))

    return pil_to_tensor(canvas).float() / 255
