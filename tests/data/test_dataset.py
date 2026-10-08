import os
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from dotenv import load_dotenv
from PIL import Image

from src.data.augmentations import FrameAugmentation
from src.data.dataset import CycleDataset, load_frame
from src.data.module import CycleDataModule
from src.data.records import Exam, Record
from src.enums import Side, View


def write_image(path: Path, level: int = 100) -> Path:
    """A prepared-like image: a 1000 x 500 breast against the right edge, and a small marker far from it."""
    image = np.zeros((2048, 1664), dtype=np.uint8)
    image[500:1500, 1164:] = level
    image[100:120, 100:160] = 255
    Image.fromarray(image).save(path)
    return path


def test_frame_is_the_breast_fitted_to_the_canvas_against_the_chest_wall(tmp_path):
    frame = load_frame(write_image(tmp_path / 'image.png'), height=256, width=160)

    assert frame.shape == (1, 256, 160) and frame.dtype == torch.float32
    tissue = frame[0] > 0
    # The marker is cropped away and the 2:1 breast fills the height without stretching: 256 x 128, on the right.
    assert tissue[:, 32:].all() and not tissue[:, :32].any()
    assert torch.allclose(frame[0, 128, 100], torch.tensor(100 / 255))


def test_a_wide_breast_shrinks_to_the_canvas_width(tmp_path):
    frame = load_frame(write_image(tmp_path / 'image.png'), height=256, width=64)  # the breast would be 128 wide

    tissue = frame[0] > 0
    assert tissue[:128].all() and not tissue[128:].any()  # 128 x 64, padding at the bottom


def test_records_are_capped_at_max_exams_in_date_order(tmp_path):
    exams = tuple(Exam(f'e{i}', date(1910 + i, 1, 1), write_image(tmp_path / f'{i}.png', level=10 * (i + 1)))
                  for i in range(5))
    dataset = CycleDataset([Record('p', Side.LEFT, View.CC, exams)], height=64, width=32, max_exams=3)

    for _ in range(5):
        levels = dataset[0]['frames'][:, 0, 32, 24]
        assert levels.shape == (3,) and (levels.diff() > 0).all()


def test_augmentation_runs_on_training_batches_only():
    data = CycleDataModule('unused', augmentation=FrameAugmentation())
    frames = torch.rand(2, 2, 1, 64, 32)

    data.trainer = SimpleNamespace(training=False)
    assert torch.equal(data.on_after_batch_transfer({'frames': frames}, 0)['frames'], frames)
    data.trainer = SimpleNamespace(training=True)
    assert not torch.equal(data.on_after_batch_transfer({'frames': frames}, 0)['frames'], frames)


load_dotenv()
DATA_DIR = os.getenv('DATA_DIR')


@pytest.mark.skipif(DATA_DIR is None or not Path(DATA_DIR).exists(), reason='needs the prepared data')
def test_real_batches_have_the_format_the_module_expects():
    data = CycleDataModule(DATA_DIR, height=512, width=320, batch_size=2, max_exams=4, num_workers=0, max_records=20)
    data.setup('fit')

    for loader, lengths in ((data.train_dataloader(), range(2, 5)), (data.val_dataloader(), [2])):
        frames = next(iter(loader))['frames']
        assert frames.dtype == torch.float32 and frames.shape[0] <= 2 and frames.shape[2:] == (1, 512, 320)
        assert frames.shape[1] in lengths
        assert frames.min() == 0 and frames.max() <= 1
