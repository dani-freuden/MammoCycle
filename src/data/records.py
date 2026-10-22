from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date
from itertools import combinations, pairwise
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.constants import Column, Side, Split, View


METADATA_CSV = 'metadata.csv'


@dataclass(frozen=True)
class Exam:
    exam_id: str
    date: date
    path: Path


@dataclass(frozen=True)
class Record:
    """One patient, one breast, one view, followed over time."""
    patient_id: str
    side: Side
    view: View
    exams: tuple[Exam, ...]  # sorted by date


def load_records(data_dir: Path, splits: Iterable[Split], view: View = None) -> dict[Split, list[Record]]:
    """Read the prepared metadata.csv and group its images into records of at least two exams.

    Dates are shifted per patient for anonymization, so they are only comparable within a record.
    """
    splits = list(splits)
    df = pd.read_csv(data_dir / METADATA_CSV, usecols=[str(column) for column in Column], dtype=str)
    df = df[df[str(Column.SPLIT_GROUP)].isin([str(split) for split in splits])]

    columns = (Column.SPLIT_GROUP, Column.PATIENT_ID, Column.LATERALITY, Column.VIEW,
               Column.EXAM_ID, Column.STUDY_DATE, Column.FILE_PATH)
    groups = defaultdict(list)
    
    # filter for only specifc view if chosen:
    if view is not None:
        df = df[df[str(Column.VIEW)] == str(view)]  

    for split, patient_id, side, view, exam_id, study_date, file_path in zip(*(df[str(c)] for c in columns)):
        exam = Exam(exam_id=exam_id, date=date.fromisoformat(study_date), path=data_dir / file_path)
        groups[split, patient_id, side, view].append(exam)

    records = {split: [] for split in splits}

    for (split, patient_id, side, view), exams in groups.items():
        if len(exams) < 2:
            continue

        # A few patients have two exams on the same date; exam_id makes the order deterministic.
        exams = tuple(sorted(exams, key=lambda exam: (exam.date, exam.exam_id)))
        records[Split(split)].append(Record(patient_id, Side(side), View(view), exams))

    return records


def to_pairs(records: list[Record], consecutive: bool = False) -> list[Record]:
    """Split each record into exam pairs, earlier exam first: every pair, n exams giving n (n - 1) / 2 records of two
    as in MammoTracker's exhaustive lesion pairs, or with consecutive only the n - 1 consecutive ones."""
    return [
        replace(record, exams=pair)
        for record in records
        for pair in (pairwise(record.exams) if consecutive else combinations(record.exams, 2))
    ]
