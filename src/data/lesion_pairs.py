"""The EMBED CC lesion pairs every method predicts and is scored on, and the command-line options that choose them."""
from pathlib import Path
from typing import Annotated

import pandas as pd
from typer import Option

from src.constants import BOX, Split, View
from src.data.records import load_records, to_pairs


DataDir = Annotated[Path, Option(help='Output folder of scripts/misc/prepare_embed_dataset.py')]
Reverse = Annotated[bool, Option(help='Know the earlier exam\'s box and predict the later one (the "current" '
                                      'columns then hold the earlier exam)')]


def lesion_pairs(data_dir: Path, splits: list[Split], reverse: bool = False) -> pd.DataFrame:
    """One row per pair of CC exams of a breast, a later (current) and an earlier (prior) one, and mammotracker lesion
    boxed in both exams. With reverse the roles swap: the earlier exam is the current one, whose box is known, and the
    later one is the prior, where the box is predicted."""
    metadata = pd.read_csv(data_dir / 'metadata.csv', dtype={'patient_id': str, 'exam_id': str, 'sop_uid': str})
    metadata['mm_per_pixel'] = metadata.pixel_spacing * metadata.cols / metadata.resized_width
    boxes = pd.read_csv(data_dir / 'boxes.csv', dtype={'sop_uid': str})
    boxes = (boxes[boxes.source == 'mammotracker']
             .merge(metadata[['sop_uid', 'exam_id', 'laterality', 'mm_per_pixel']], on='sop_uid')
             .rename(columns={f'png_{c}': c for c in BOX}))
    by_image = {key: group[['lesion_id', 'kind', 'mm_per_pixel', *BOX]]
                for key, group in boxes.groupby(['exam_id', 'laterality'])}

    rows = []
    for split, records in load_records(data_dir, splits, View.CC).items():
        for record in to_pairs(records):
            prior, current = record.exams[::-1] if reverse else record.exams
            prior_boxes, current_boxes = (by_image.get((exam.exam_id, str(record.side))) for exam in (prior, current))
            if prior_boxes is None or current_boxes is None:
                continue

            shared = prior_boxes.merge(current_boxes, on='lesion_id', suffixes=('_prior', '_current'))
            for lesion in shared.to_dict('records'):
                rows.append({'split_group': str(split), 'patient_id': record.patient_id, 'laterality': str(record.side),
                             'lesion_id': lesion['lesion_id'], 'kind': lesion['kind_current'],
                             'prior_exam_id': prior.exam_id, 'prior_date': prior.date,
                             'current_exam_id': current.exam_id, 'current_date': current.date,
                             'prior_path': str(prior.path), 'current_path': str(current.path),
                             **{f'prior_{c}': lesion[f'{c}_prior'] for c in BOX},
                             **{f'current_{c}': lesion[f'{c}_current'] for c in BOX},
                             'prior_mm_per_pixel': lesion['mm_per_pixel_prior']})

    return pd.DataFrame(rows)


def pair_tasks(pairs: pd.DataFrame) -> list[dict]:
    """One dict per pair, for a worker pool: its current_path, prior_path and current_box (ymin, xmin, ymax, xmax)."""
    return [{'current_path': current_path, 'prior_path': prior_path, 'current_box': current_box}
            for current_path, prior_path, current_box
            in zip(pairs.current_path, pairs.prior_path, pairs[[f'current_{c}' for c in BOX]].values.tolist())]
