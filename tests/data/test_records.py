from datetime import date
from pathlib import Path

import pytest

from src.data.records import Exam, Record, load_records, to_pairs
from src.enums import Side, Split, View


# Columns in a different order than Column, plus one load_records should ignore.
METADATA = """\
split_group,patient_id,exam_id,study_date,view,laterality,file_path,manufacturer
train,p1,e2,1912-05-01,CC,L,png/p1_e2.png,X
train,p1,e1,1910-01-01,CC,L,png/p1_e1.png,X
train,p1,e1,1910-01-01,MLO,L,png/p1_e1_mlo.png,X
train,p2,b,1920-01-01,CC,R,png/p2_b.png,X
train,p2,a,1920-01-01,CC,R,png/p2_a.png,X
dev,p3,e1,1930-01-01,MLO,R,png/p3_e1.png,X
dev,p3,e2,1931-01-01,MLO,R,png/p3_e2.png,X
"""


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / 'metadata.csv').write_text(METADATA)
    return tmp_path


def _by_key(records):
    return {(record.patient_id, record.side, record.view): record for record in records}


def test_groups_by_patient_side_view(data_dir):
    records = _by_key(load_records(data_dir, [Split.TRAIN])[Split.TRAIN])

    assert set(records) == {('p1', Side.LEFT, View.CC), ('p2', Side.RIGHT, View.CC)}


def test_drops_records_with_a_single_exam(data_dir):
    records = _by_key(load_records(data_dir, [Split.TRAIN])[Split.TRAIN])

    assert ('p1', Side.LEFT, View.MLO) not in records


def test_sorts_exams_by_date(data_dir):
    record = _by_key(load_records(data_dir, [Split.TRAIN])[Split.TRAIN])['p1', Side.LEFT, View.CC]

    assert [exam.exam_id for exam in record.exams] == ['e1', 'e2']
    assert [exam.date for exam in record.exams] == [date(1910, 1, 1), date(1912, 5, 1)]
    assert record.exams[0].path == data_dir / 'png/p1_e1.png'


def test_breaks_same_date_ties_by_exam_id(data_dir):
    record = _by_key(load_records(data_dir, [Split.TRAIN])[Split.TRAIN])['p2', Side.RIGHT, View.CC]

    assert [exam.exam_id for exam in record.exams] == ['a', 'b']


def test_returns_only_requested_splits(data_dir):
    records = load_records(data_dir, [Split.DEV])

    assert set(records) == {Split.DEV}
    assert [record.patient_id for record in records[Split.DEV]] == ['p3']


def test_requested_split_without_rows_is_empty(data_dir):
    records = load_records(data_dir, [Split.TRAIN, Split.TEST])

    assert records[Split.TEST] == []


def _record(patient_id, num_exams):
    exams = tuple(Exam(f'e{i}', date(1910 + i, 1, 1), Path(f'png/{patient_id}_e{i}.png')) for i in range(num_exams))
    return Record(patient_id, Side.LEFT, View.CC, exams)


def test_to_pairs_splits_into_consecutive_pairs():
    pairs = to_pairs([_record('p1', 4)])

    assert [[exam.exam_id for exam in pair.exams] for pair in pairs] == [['e0', 'e1'], ['e1', 'e2'], ['e2', 'e3']]


def test_to_pairs_keeps_patient_side_view():
    pairs = to_pairs([_record('p1', 3)])

    assert {(pair.patient_id, pair.side, pair.view) for pair in pairs} == {('p1', Side.LEFT, View.CC)}


def test_to_pairs_keeps_a_two_exam_record_whole():
    record = _record('p1', 2)

    assert to_pairs([record]) == [record]


def test_to_pairs_keeps_record_order():
    pairs = to_pairs([_record('p1', 2), _record('p2', 3)])

    assert [pair.patient_id for pair in pairs] == ['p1', 'p2', 'p2']
