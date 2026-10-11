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

    def __init__(self, records: list[Record], height: int, width: int, max_exams: int | None = None,
                 clear_background: bool = False):
        self.records = records
        self.height = height
        self.width = width
        self.max_exams = max_exams
        self.clear_background = clear_background  # for images whose background is not exactly zero, see load_frame

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        exams = self.records[index].exams
        if self.max_exams is not None and len(exams) > self.max_exams:
            # A new random subset every time, not necessarily consecutive, varies the time gaps between frames.
            exams = [exams[i] for i in sorted(random.sample(range(len(exams)), self.max_exams))]

        frames = [load_frame(exam.path, self.height, self.width, self.clear_background) for exam in exams]
        return {'frames': torch.stack(frames)}


def load_frame(path: Path, height: int, width: int, clear_background: bool = False) -> Tensor:
    """Crop a prepared image to the breast and fit it into a height x width canvas, without stretching.

    The prepared PNGs are 8-bit with the chest wall on the right edge. The crop is the bounding box of the largest
    connected region of non-zero pixels, which leaves out the markers in the background. It is resized to fill the
    canvas height, or its width when the breast is wider than the canvas, and padded with zeros on the left (away from
    the chest wall) and at the bottom.

    With clear_background, for images whose background is not exactly zero, the breast is found by breast_mask
    instead, and every pixel outside it is set to zero before the crop.
    """
    with Image.open(path) as image:
        image.load()

    if clear_background:
        breast = breast_mask(np.asarray(image))
        image = Image.fromarray(np.asarray(image) * breast.astype(np.uint8))
        x, y, w, h = cv2.boundingRect(breast.astype(np.uint8))
    else:
        _, _, stats, _ = cv2.connectedComponentsWithStats((np.asarray(image) > 0).astype(np.uint8))
        x, y, w, h = stats[1 + stats[1:, cv2.CC_STAT_AREA].argmax(), :4]  # label 0 is the background
    scale = min(height / h, width / w)
    breast = image.crop((x, y, x + w, y + h)).resize((round(w * scale), round(h * scale)), Image.Resampling.BILINEAR)

    canvas = Image.new('L', (width, height))
    canvas.paste(breast, (width - breast.width, 0))

    return pil_to_tensor(canvas).float() / 255


def breast_mask(image: np.ndarray) -> np.ndarray:
    """The breast in an 8-bit image: the largest contour of a triangle threshold, filled, as a boolean mask.

    The triangle method puts the threshold at the foot of the dominant histogram peak, the background, so it adapts
    when the background is not at zero. Otsu does not do here: on about a quarter of the EMBED images its threshold
    falls inside the breast and leaves out the fat and the skin.
    """
    _, binary = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 1, thickness=cv2.FILLED)

    return mask.astype(bool)
