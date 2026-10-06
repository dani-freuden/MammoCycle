from enum import Enum, auto
from typing import List, Mapping, Union

class StringEnum(Enum):
    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: List[str]
    ) -> str:
        return name.lower()

    def __str__(self):
        return self.value


class IntegerEnum(Enum):
    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: List[int]
    ) -> int:
        if not last_values:
            raise ValueError(
                'Can not use auto() on the first member of IntegerEnum, it must be specified explicitly.'
            )

        return last_values[-1] + 1

    def __int__(self):
        return self.value


# Values match the prepared Shifa metadata.csv.
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


