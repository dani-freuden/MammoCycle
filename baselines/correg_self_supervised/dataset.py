"""Self-supervised data of the correction-regression baseline: pairs made from a single image, with no annotations.

For each image of one split and view of the prepared EMBED set, a random point p of the breast
(src.data.dataset.breast_mask) is the current crop's centre, and the prior crop, from the same image, is centred on
c = p + (ex, ey), a random shift of up to max_shift PNG pixels each way. The target follows CorrectionDataset's
convention, (prior crop centre - true point) / WINDOW, so (dx, dy) = (ex, ey) / WINDOW and the corrected centre
c - WINDOW * (dx, dy) is p. The default max_shift, WINDOW / 2, keeps p inside the prior crop: about 29 mm on the Hologic
images (0.1305 mm per PNG pixel). Each crop gets its own gamma, gain, blur and noise
(src.data.augmentations.augment_intensity), and no spatial deformation.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from src.data.augmentations import augment_intensity
from src.data.dataset import breast_mask
from src.enums import Column, Split, View

from ..correction_regression.data.prepare import WINDOW, crop


class SelfSupervisedDataset(Dataset):
    """One split and view of the prepared images, a new random pair per image every time, as CorrectionDataset gives
    them: 'current' and 'prior' crops and 'target' (dx, dy)."""

    def __init__(self, metadata_csv: Path, split: Split, view: View = View.CC, max_shift: int = WINDOW // 2):
        metadata = pd.read_csv(metadata_csv)
        files = metadata.file_path[metadata.split_group.eq(str(split)) & metadata[str(Column.VIEW)].eq(str(view))]
        self.paths = [str(metadata_csv.parent / file) for file in files]
        self.max_shift = max_shift

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        path = self.paths[index]
        with Image.open(path) as image:
            ys, xs = np.nonzero(breast_mask(np.asarray(image)))
        # torch's generator, which the DataLoader seeds differently in each worker.
        i = torch.randint(len(xs), ()).item()
        x, y = int(xs[i]), int(ys[i])
        ex, ey = torch.randint(-self.max_shift, self.max_shift + 1, (2,)).tolist()
        current, prior = augment_intensity(torch.stack([crop(path, x, y), crop(path, x + ex, y + ey)]))

        return {'current': current, 'prior': prior,
                'target': torch.tensor([ex / WINDOW, ey / WINDOW], dtype=torch.float32)}
