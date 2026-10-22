"""Enums and constants shared across the repo, and the machine-specific paths, read from .env (see .env.example)."""
import os
from enum import Enum, auto
from pathlib import Path

from dotenv import load_dotenv


class StringEnum(Enum):
    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list[str]) -> str:
        return name.lower()

    def __str__(self):
        return self.value


# Values match the prepared metadata.csv.
class View(StringEnum):
    CC = 'CC'
    MLO = 'MLO'

class Side(StringEnum):
    LEFT = 'L'
    RIGHT = 'R'

class Column(StringEnum):
    PATIENT_ID = auto()
    EXAM_ID = auto()
    STUDY_DATE = auto()
    LATERALITY = auto()
    VIEW = auto()
    FILE_PATH = auto()
    SPLIT_GROUP = auto()

class Split(StringEnum):
    TRAIN = auto()
    DEV = auto()
    TEST = auto()


BOX = ['ymin', 'xmin', 'ymax', 'xmax']  # the order of every box's columns, in PNG pixels
PAIR_KEY = ['patient_id', 'laterality', 'lesion_id', 'prior_exam_id', 'current_exam_id']  # identifies a lesion pair
ID_DTYPES = {'patient_id': str, 'prior_exam_id': str, 'current_exam_id': str}  # read a pairs csv with these
# ponytail: one nominal mm per PNG pixel for every image (Hologic 0.106-0.140, GE has no PixelSpacing); resample each
# image to a working spacing (geometry.resample_to_spacing) if the per-image spacing matters
SPACING_MM = 0.12


def _env_path(name: str) -> Path | None:
    value = os.getenv(name)
    return Path(value) if value else None


load_dotenv()
EMBED_DATA_DIR = _env_path('EMBED_DATA_DIR')  # the annotated set, scripts/misc/prepare_embed_dataset.py
EMBED_ALL_DIR = _env_path('EMBED_ALL_DIR')  # the same with --all-images
EMBED_ROOT = _env_path('EMBED_ROOT')  # EMBED Open Data, with tables/ and images/
MAMMOTRACKER_CSV_DIR = _env_path('MAMMOTRACKER_CSV_DIR')  # the mammotracker ROI CSVs
