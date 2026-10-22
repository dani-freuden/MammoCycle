"""Check the prepared EMBED images: is the breast where the orientation put it, and do the boxes land on tissue?

For a random share of the prepared images that have boxes, the PNG is loaded and the breast found with
breast_mask:
1. Orientation: the breast's centroid must be in the right half, since the chest wall was put on the right edge. For
   images flipped by their PatientOrientation this checks the tag; for those where the centroid decided, it only
   checks that the prepared image agrees with the reduced copy the decision was made on.
2. Boxes: each box, mapped into the PNG as boxes.csv does (prepare_embed_dataset.ORDER), must have at least
   MIN_ON_BREAST of its area on the breast. Three other readings are scored as well, to show what would fit a box that
   fails: not mirrored, x and y swapped, and both.

Writes check_failures.csv to the prepared folder: one row per box that fails, or whose image fails the orientation
check, with what failed.

Usage:
    python scripts/misc/check_embed_prepared.py   # prompts for every option
    python scripts/misc/check_embed_prepared.py --percent 100 --workers 32
"""
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from PIL import Image
from tqdm import tqdm
from typer import Option

from prepare_embed_dataset import GEOMETRY, GIVEN, map_box, standard_box
from src.constants import EMBED_DATA_DIR, Column
from src.utils.anatomy import breast_mask


MIN_ON_BREAST = 0.5
# Each reading turns a standard_box (ymin, xmin, ymax, xmax, chest wall on the left) of an image cols wide into the box
# map_box should receive. Pre-mirroring undoes the mirror map_box applies.
READINGS = {
    'convention': lambda y1, x1, y2, x2, cols: (y1, x1, y2, x2),
    'unmirrored': lambda y1, x1, y2, x2, cols: (y1, cols - x2, y2, cols - x1),
    'swapped': lambda y1, x1, y2, x2, cols: (x1, y1, x2, y2),
    'swapped_unmirrored': lambda y1, x1, y2, x2, cols: (x1, cols - y2, x2, cols - y1),
}


def share_on_breast(breast: np.ndarray, box) -> float:
    """Share of a box's area, in PNG pixels, that lies on the breast. Parts outside the image count as off it."""
    ymin, xmin, ymax, xmax = (round(v) for v in box)
    area = (ymax - ymin) * (xmax - xmin)
    if area <= 0:
        return 0.0

    return breast[max(ymin, 0):max(ymax, 0), max(xmin, 0):max(xmax, 0)].sum() / area


def check_image(task: dict) -> tuple[bool, list[dict]]:
    """Whether the breast's centroid is in the right half, and each box's share on the breast under every reading."""
    breast = breast_mask(np.asarray(Image.open(task['path'])))
    centroid_right = np.nonzero(breast)[1].mean() > breast.shape[1] / 2

    cols = task['geometry'][GEOMETRY.index('cols')]
    shares = [{name: share_on_breast(breast, map_box(read(*box, cols), *task['geometry']))
               for name, read in READINGS.items()} for box in task['boxes']]

    return centroid_right, shares


def main(
    prepared_dir: Annotated[Path, Option(prompt=True, help='Output folder of prepare_embed_dataset.py')]
        = EMBED_DATA_DIR,
    percent: Annotated[float, Option(prompt=True, help='Percent of the images with boxes to check')] = 5.0,
    workers: Annotated[int, Option(prompt=True, help='Parallel checks')] = 16,
    seed: Annotated[int, Option(prompt=True, help='Seed for sampling the images')] = 0,
):
    ids = {str(Column.PATIENT_ID): str, str(Column.EXAM_ID): str, 'sop_uid': str}
    images = pd.read_csv(prepared_dir / 'metadata.csv', dtype=ids)
    boxes = pd.read_csv(prepared_dir / 'boxes.csv', dtype=ids)
    images = images[images.sop_uid.isin(boxes.sop_uid)].sample(frac=percent / 100, random_state=seed)
    boxes = boxes[boxes.sop_uid.isin(images.sop_uid)]

    by_image = {sop_uid: group for sop_uid, group in boxes.groupby('sop_uid')}
    tasks = [{'path': str(prepared_dir / row[str(Column.FILE_PATH)]), 'geometry': [row[c] for c in GEOMETRY],
              'boxes': [standard_box(source, given) for source, given
                        in zip(by_image[row['sop_uid']].source, by_image[row['sop_uid']][GIVEN].values.tolist())]}
             for _, row in images.iterrows()]
    with Pool(workers) as pool:
        results = list(tqdm(pool.imap(check_image, tasks), total=len(tasks), desc='check', unit='img'))

    rows = []
    for (_, image), (centroid_right, shares) in zip(images.iterrows(), results):
        for (_, box), share in zip(by_image[image['sop_uid']].iterrows(), shares):
            rows.append({**image[['sop_uid', str(Column.PATIENT_ID), str(Column.FILE_PATH), 'image_type',
                                  'manufacturer', 'patient_orientation', 'orientation_source', 'flipped']],
                         'centroid_right': centroid_right,
                         **box[['source', 'kind', 'lesion_id', *GIVEN, 'png_ymin', 'png_xmin', 'png_ymax', 'png_xmax']],
                         **share})
    checked = pd.DataFrame(rows)
    box_ok = checked.convention >= MIN_ON_BREAST
    checked['failed'] = np.select([~checked.centroid_right & ~box_ok, ~checked.centroid_right, ~box_ok],
                                  ['orientation+box', 'orientation', 'box'], default='')

    out_path = prepared_dir / 'check_failures.csv'
    checked[checked.failed != ''].to_csv(out_path, index=False)

    per_image = checked.drop_duplicates('sop_uid')
    print(f'\nChecked {len(per_image)} images ({percent:g}% of those with boxes) and {len(checked)} boxes')
    print('Centroid not in the right half, by orientation source:',
          per_image.groupby('orientation_source').centroid_right.apply(lambda ok: f'{(~ok).sum()} of {len(ok)}')
          .to_dict())
    print(f'\nShare of boxes with at least {MIN_ON_BREAST:.0%} of their area on the breast, per reading:')
    print((checked[list(READINGS)] >= MIN_ON_BREAST).groupby(checked.source).mean().round(3).to_string())
    print(f'\nWrote {(checked.failed != "").sum()} failing boxes to {out_path}')


if __name__ == '__main__':
    typer.run(main)
