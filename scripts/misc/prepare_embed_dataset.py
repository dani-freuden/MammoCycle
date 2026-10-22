"""Prepare the annotated CC images of EMBED Open Data in the format of the Shifa set: 8-bit PNGs, chest wall on the right.

Selection: CC views, full-field 2D or C-View (no spot compression, magnification or view modifier), that carry a box
from EMBED's ROI_coords or from the mammotracker CSVs. Each exam and breast keeps one image (2D before C-View, then
the highest InstanceNumber), and only breasts with at least two such exams are kept, since a record needs two.

With --all-images, every exam, breast and view keeps one full-field 2D CC or MLO image instead (the highest
InstanceNumber), annotated or not, for training on single images. Patients of the annotated set keep their split
there (--splits-from), so its dev and test pairs stay out of training; the other patients are split 80/10/10. No
boxes.csv is written.

Each image is converted as mirai-shira converts Shifa: dcmj2pnm +on2 --min-max-window, to a 16-bit PNG. It is then
1. flipped so the chest wall is on the right. PatientOrientation says where the rows run: towards A (anterior) puts the
   chest wall on the left, towards P on the right. Without a usable tag, the side of the breast's centroid decides.
2. resized without stretching into 1664 x 2048, padded on the left and at the bottom.
3. reduced to 8 bits with round(v / 257), which is how the Shifa 8-bit cache relates to its 16-bit images.

Writes to out_dir:
    png/<patient>/<sop_uid>.png
    metadata.csv  one row per image: the columns records.py reads, split_group (80/10/10 by patient), the geometry,
                  and pixel_spacing in mm on the stored image (DICOM PixelSpacing; missing on GE images)
    boxes.csv     one row per box: the four numbers as its source gives them (given_0 to given_3), and the box
                  mapped into the PNG as (ymin, xmin, ymax, xmax), see ORDER (not with --all-images)
    failed.csv    images that could not be converted

PNGs that already exist are not converted again. dcmj2pnm must be on PATH (conda install -c conda-forge dcmtk).

Usage:
    python scripts/misc/prepare_embed_dataset.py   # prompts for every option but --splits-from
    python scripts/misc/prepare_embed_dataset.py --workers 32
    python scripts/misc/prepare_embed_dataset.py --all-images --out-dir $EMBED_ALL_DIR
"""
import ast
import os
import random
import shutil
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import pandas as pd
import typer
from PIL import Image
from tqdm import tqdm
from typer import Option

from src.constants import EMBED_DATA_DIR, EMBED_ROOT, MAMMOTRACKER_CSV_DIR, Column, Split
from src.utils.anatomy import breast_mask


HEIGHT, WIDTH = 2048, 1664
SPLIT_SHARES = {Split.TRAIN: 0.8, Split.DEV: 0.1, Split.TEST: 0.1}
NAS_PREFIX = '/mnt/NAS2/mammo/anon_dicom/'  # anon_dicom_path as recorded on the EMBED side
FLIP_BY_ORIENTATION = {'A': True, 'P': False}  # first PatientOrientation value -> flip to put the chest wall right
# Both sources give boxes in a frame with the chest wall on the left, whatever the stored orientation of the DICOM:
# EMBED as (ymin, xmin, ymax, xmax), mammotracker as (xmin, ymin, xmax, ymax). Measured with check_embed_prepared.py,
# where all but 1 of the 2,626 boxes then lie at least half on the breast.
ORDER = {'embed': (0, 1, 2, 3), 'mammotracker': (1, 0, 3, 2)}  # positions of ymin, xmin, ymax, xmax in the given four
GIVEN = ['given_0', 'given_1', 'given_2', 'given_3']
GEOMETRY = ['rows', 'cols', 'resized_height', 'resized_width', 'pad_left']  # metadata.csv columns map_box needs

METADATA_COLUMNS = {
    'empi_anon': str(Column.PATIENT_ID),
    'acc_anon': str(Column.EXAM_ID),
    'study_date_anon': str(Column.STUDY_DATE),
    'ImageLateralityFinal': str(Column.LATERALITY),
    'ViewPosition': str(Column.VIEW),
    'FinalImageType': 'image_type',
    'spot_mag': 'spot_mag',
    '0_ViewCodeSequence_0_ViewModifierCodeSequence_CodeMeaning': 'view_modifier',
    'InstanceNumber': 'instance_number',
    'PatientOrientation': 'patient_orientation',
    'Rows': 'rows',
    'Columns': 'cols',
    'PixelSpacing': 'pixel_spacing',
    'Manufacturer': 'manufacturer',
    'ROI_coords': 'roi_coords',
    'anon_dicom_path': 'dicom_path',
}


def load_images(embed_root: Path, views: list[str], image_types: list[str]) -> pd.DataFrame:
    """The images of these views and types that can be prepared: full-field, without spot, magnification or view
    modifier."""
    df = pd.read_csv(embed_root / 'tables' / 'EMBED_OpenData_metadata.csv', usecols=list(METADATA_COLUMNS),
                     dtype=str).rename(columns=METADATA_COLUMNS)
    df = df[df[str(Column.VIEW)].isin(views) & df.image_type.isin(image_types)
            & df.spot_mag.isna() & df.view_modifier.isna()].copy()

    df['dicom_path'] = df.dicom_path.str.replace(NAS_PREFIX, f'{embed_root}/images/', regex=False)
    df['sop_uid'] = df.dicom_path.map(lambda path: Path(path).stem)
    df[str(Column.STUDY_DATE)] = pd.to_datetime(df[str(Column.STUDY_DATE)]).dt.strftime('%Y-%m-%d')
    df[['instance_number', 'rows', 'cols']] = df[['instance_number', 'rows', 'cols']].astype(float).astype(int)
    # Stored as '("0.065238", "0.065238")', rows then columns; they are equal on every prepared image.
    df['pixel_spacing'] = df.pixel_spacing.map(lambda value: float(ast.literal_eval(value)[0])
                                               if isinstance(value, str) else np.nan)
    first =df.patient_orientation.map(lambda value: ast.literal_eval(value)[0] if isinstance(value, str) else None)
    df['flip_by_tag'] = first.map(FLIP_BY_ORIENTATION)  # NaN when the tag is missing or names neither A nor P

    return df


def flatten_boxes(value) -> list[tuple[int, ...]]:
    """Every 4-number box in a parsed ROI_coords value; some EMBED values nest lists or hold empty tuples."""
    if isinstance(value, (list, tuple)) and len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
        return [tuple(int(v) for v in value)]
    return [box for item in (value if isinstance(value, (list, tuple)) else []) for box in flatten_boxes(item)]


def load_boxes(images: pd.DataFrame, mammotracker_dir: Path) -> pd.DataFrame:
    """One row per box of either source, with the four numbers in the order the source gives them."""
    rows = [('embed', sop_uid, '', None, box)
            for sop_uid, value in zip(images.sop_uid, images.roi_coords) if isinstance(value, str)
            for box in flatten_boxes(ast.literal_eval(value))]

    for kind in ('mass', 'calc'):
        df = pd.read_csv(mammotracker_dir / f'embed_open_data_{kind}_roi_coords.csv', dtype=str)
        for sop_uid, coords, lesion_ids in zip(df.anon_dicom_path_ID, df.ROI_coords, df.Lesion_IDs):
            boxes, lesion_ids = flatten_boxes(ast.literal_eval(coords)), ast.literal_eval(lesion_ids)
            rows += [('mammotracker', sop_uid, kind, lesion_id, box) for box, lesion_id in zip(boxes, lesion_ids)]

    boxes = pd.DataFrame(rows, columns=['source', 'sop_uid', 'kind', 'lesion_id', 'box'])
    boxes[GIVEN] = pd.DataFrame(boxes.pop('box').tolist(), index=boxes.index)

    return boxes


def select(images: pd.DataFrame, boxes: pd.DataFrame) -> pd.DataFrame:
    """One annotated image per exam and breast, in breasts with at least two such exams."""
    print(f'CC full-field 2D or C-View images: {len(images)}')
    annotated = images[images.sop_uid.isin(boxes.sop_uid)]
    by_source = {source: annotated.sop_uid.isin(group.sop_uid).sum() for source, group in boxes.groupby('source')}
    print(f'  with a box: {len(annotated)} (embed: {by_source.get("embed", 0)}, '
          f'mammotracker: {by_source.get("mammotracker", 0)})')

    # The preferred image sorts last: 2D before C-View, then the highest InstanceNumber.
    breast = [str(Column.PATIENT_ID), str(Column.LATERALITY)]
    one = (annotated.assign(is_2d=annotated.image_type.eq('2D')).sort_values(['is_2d', 'instance_number'])
           .drop_duplicates([str(Column.EXAM_ID), str(Column.LATERALITY)], keep='last').drop(columns='is_2d'))
    print(f'  one per exam and breast: {len(one)} ({one.image_type.eq("cview").sum()} of them C-View)')

    kept = one[one.groupby(breast)[str(Column.EXAM_ID)].transform('size') >= 2]
    print(f'  in breasts with at least two exams: {len(kept)} images, '
          f'{kept.groupby(breast).ngroups} breasts, {kept[str(Column.PATIENT_ID)].nunique()} patients')

    return kept.reset_index(drop=True)


def assign_splits(patients: pd.Series, seed: int) -> pd.Series:
    """A seeded random split by patient, in the shares of SPLIT_SHARES."""
    unique = sorted(patients.unique())
    random.Random(seed).shuffle(unique)
    bounds = np.cumsum([round(share * len(unique)) for share in SPLIT_SHARES.values()])
    bounds[-1] = len(unique)
    split_of = {patient: str(split) for split, start, stop in zip(SPLIT_SHARES, [0, *bounds[:-1]], bounds)
                for patient in unique[start:stop]}

    return patients.map(split_of)


def fitted_size(rows: int, cols: int) -> tuple[int, int]:
    """(height, width) of an image of rows x cols resized without stretching into HEIGHT x WIDTH."""
    scale = min(HEIGHT / rows, WIDTH / cols)
    return round(rows * scale), round(cols * scale)


def breast_on_right(image: np.ndarray) -> bool:
    """True if the centroid of the breast (breast_mask) is in the right half of an 8-bit image."""
    return np.nonzero(breast_mask(image))[1].mean() > image.shape[1] / 2


def prepare_image(task: dict) -> dict:
    """Convert one DICOM to its PNG. Returns whether it was flipped, on what grounds, or the error."""
    result = {'sop_uid': task['sop_uid'], 'flipped': task['flip_by_tag'], 'orientation_source': 'tag', 'error': None}
    out_path = Path(task['out_path'])
    if task['flip_by_tag'] is not None and out_path.exists():
        return result  # without a usable tag the flip is not recorded anywhere, so those are converted again

    try:
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = os.path.join(tmp, 'raw.png')
            done = subprocess.run(['dcmj2pnm', '+on2', '--min-max-window', task['dicom_path'], raw_path],
                                  capture_output=True, text=True)
            if done.returncode != 0 or not os.path.exists(raw_path):
                raise RuntimeError(done.stderr.strip()[:300] or f'dcmj2pnm exited {done.returncode}')
            raw = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)

        if raw.shape != (task['rows'], task['cols']):
            raise ValueError(f'image is {raw.shape}, the metadata says {(task["rows"], task["cols"])}')
        if task['flip_by_tag'] is None:
            right = breast_on_right(np.round(raw[::4, ::4] / 257).astype(np.uint8))
            result.update(flipped=not right, orientation_source='pixels')
        if result['flipped']:
            raw = raw[:, ::-1]

        # Resized in 16 bits as mirai-shira does it: PIL bilinear in mode I, which every Pillow version supports.
        height, width = fitted_size(*raw.shape)
        resized = np.asarray(Image.fromarray(raw.astype(np.int32)).resize((width, height), Image.Resampling.BILINEAR))
        canvas = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        canvas[:height, WIDTH - width:] = np.round(resized / 257).astype(np.uint8)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix('.tmp.png')
        cv2.imwrite(str(tmp_path), canvas)
        os.replace(tmp_path, out_path)
    except Exception as exc:
        result.update(flipped=None, orientation_source=None, error=f'{type(exc).__name__}: {exc}'[:300])

    return result


def standard_box(source: str, given) -> tuple:
    """(ymin, xmin, ymax, xmax), with the chest wall on the left, from the four numbers as a source gives them."""
    return tuple(given[i] for i in ORDER[source])


def map_box(box, rows: int, cols: int, resized_height: int, resized_width: int, pad_left: int) -> tuple:
    """Map a standard_box into the prepared PNG: mirrored, since the PNG has the chest wall on the right whatever the
    DICOM's stored orientation, then resized and padded like the image."""
    ymin, xmin, ymax, xmax = box
    sy, sx = resized_height / rows, resized_width / cols

    return ymin * sy, pad_left + (cols - xmax) * sx, ymax * sy, pad_left + (cols - xmin) * sx


def main(
    embed_root: Annotated[Path, Option(prompt=True, help='EMBED Open Data root, with tables/ and images/')]
        = EMBED_ROOT,
    mammotracker_dir: Annotated[Path, Option(prompt=True, help='Folder with the mammotracker ROI CSVs')]
        = MAMMOTRACKER_CSV_DIR,
    out_dir: Annotated[Path, Option(prompt=True, help='Where the PNGs and CSVs go')]
        = EMBED_DATA_DIR,
    all_images: Annotated[bool, Option(prompt=True, help='One 2D image per exam, breast and view (CC and MLO), '
                                                         'annotated or not')]
        = False,
    splits_from: Annotated[Path, Option(help='With --all-images: the annotated metadata.csv, whose patients keep '
                                             'their split')]
        = EMBED_DATA_DIR / 'metadata.csv',
    workers: Annotated[int, Option(prompt=True, help='Parallel conversions')] = 32,
    seed: Annotated[int, Option(prompt=True, help='Seed for the patient split')] = 0,
):
    if shutil.which('dcmj2pnm') is None:
        raise SystemExit('dcmj2pnm is not on PATH; install DCMTK first: conda install -c conda-forge dcmtk')
    if all_images and out_dir.resolve() == splits_from.resolve().parent:
        raise SystemExit(f'--all-images would overwrite the annotated set in {out_dir}; give another --out-dir')

    if all_images:
        images = (load_images(embed_root, ['CC', 'MLO'], ['2D']).sort_values('instance_number')
                  .drop_duplicates([str(Column.EXAM_ID), str(Column.LATERALITY), str(Column.VIEW)], keep='last'))
        print(f'2D CC and MLO images, one per exam, breast and view: {len(images)} '
              f'({images[str(Column.VIEW)].value_counts().to_dict()}), '
              f'{images[str(Column.PATIENT_ID)].nunique()} patients')
        annotated = pd.read_csv(splits_from, dtype=str).groupby(str(Column.PATIENT_ID))[str(Column.SPLIT_GROUP)].first()
        patients = images[str(Column.PATIENT_ID)]
        images[str(Column.SPLIT_GROUP)] = patients.map(annotated).fillna(assign_splits(patients, seed))
    else:
        images = load_images(embed_root, ['CC'], ['2D', 'cview'])
        boxes = load_boxes(images, mammotracker_dir)
        images = select(images, boxes)
        images[str(Column.SPLIT_GROUP)] = assign_splits(images[str(Column.PATIENT_ID)], seed)
    images[str(Column.FILE_PATH)] = 'png/' + images[str(Column.PATIENT_ID)] + '/' + images.sop_uid + '.png'
    print(f'Without a usable PatientOrientation, so the breast centroid decides: '
          f'{images.flip_by_tag.isna().sum()} of {len(images)}')

    tasks = [{'sop_uid': row.sop_uid, 'dicom_path': row.dicom_path, 'out_path': str(out_dir / row.file_path),
              'rows': row.rows, 'cols': row.cols, 'flip_by_tag': None if pd.isna(row.flip_by_tag) else row.flip_by_tag}
             for row in images.itertuples()]
    with Pool(workers) as pool:
        results = list(tqdm(pool.imap_unordered(prepare_image, tasks), total=len(tasks), desc='prepare', unit='img'))

    results = pd.DataFrame(results)
    images = images.drop(columns='flip_by_tag').merge(results, on='sop_uid')
    failed = images[images.error.notna()]
    images = images[images.error.isna()].drop(columns=['error', 'roi_coords', 'spot_mag', 'view_modifier'])
    images[['resized_height', 'resized_width']] = [fitted_size(r, c) for r, c in zip(images.rows, images.cols)]
    images['pad_left'] = WIDTH - images.resized_width

    out_dir.mkdir(parents=True, exist_ok=True)
    images.to_csv(out_dir / 'metadata.csv', index=False)
    failed[['sop_uid', 'dicom_path', 'error']].to_csv(out_dir / 'failed.csv', index=False)

    print(f'\nPrepared {len(images)} images, {len(failed)} failed (failed.csv)')
    print('Orientation:', images.groupby(['orientation_source', 'flipped']).size().to_dict())
    print('Splits:', images.groupby(str(Column.SPLIT_GROUP))[str(Column.PATIENT_ID)].agg(['nunique', 'size'])
          .rename(columns={'nunique': 'patients', 'size': 'images'}).to_dict('index'))

    if not all_images:
        geometry = images.set_index('sop_uid')[GEOMETRY]
        boxes = boxes[boxes.sop_uid.isin(geometry.index)].reset_index(drop=True)
        mapped = [map_box(standard_box(source, given), *geometry.loc[sop_uid])
                  for source, sop_uid, given in zip(boxes.source, boxes.sop_uid, boxes[GIVEN].values.tolist())]
        boxes[['png_ymin', 'png_xmin', 'png_ymax', 'png_xmax']] = np.round(mapped, 1)
        boxes.to_csv(out_dir / 'boxes.csv', index=False)
        print('Boxes:', boxes.source.value_counts().to_dict())
    print(f'Wrote to {out_dir}')


if __name__ == '__main__':
    typer.run(main)
