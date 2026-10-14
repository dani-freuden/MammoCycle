"""
Usage:
    python -m baselines.mere_mask.predict
    python -m baselines.mere_mask.predict --splits dev --splits test --workers 32
"""
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import SimpleITK as sitk
import typer
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
from typer import Option

from src.data.dataset import breast_mask
from src.data.records import load_records, to_pairs
from src.enums import Split, View


load_dotenv()
DATA_DIR = os.getenv('EMBED_DATA_DIR')

OUTLINE_STEP = 8  # PNG pixels between the points mapped along each edge of a box
SHRINK = 8  # the signed distance maps are registered at 1/SHRINK of the PNG size; full size moves points by ~2 px
BOX = ['ymin', 'xmin', 'ymax', 'xmax']


def signed_distance(mask):
    img = sitk.GetImageFromArray(mask.astype(np.uint8))

    return sitk.SignedMaurerDistanceMap(
        img,
        insideIsPositive=True,
        squaredDistance=False,
        useImageSpacing=True,
    )


def register_masks_bspline(source_mask, target_mask):
    """A B-spline transform that maps points (x, y) of source_mask into target_mask, registering their signed distance
    maps. ITK's transform maps the fixed image's points into the moving one, so the source is the fixed image."""
    source, target = (sitk.BinShrink(signed_distance(mask), [SHRINK, SHRINK]) for mask in (source_mask, target_mask))

    # Coarse smooth deformation grid
    transform = sitk.BSplineTransformInitializer(
        image1=source,
        transformDomainMeshSize=[5, 5],
        order=3,
    )

    reg = sitk.ImageRegistrationMethod()

    reg.SetMetricAsMeanSquares()

    reg.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=100,
    )

    reg.SetInitialTransform(
        transform,
        inPlace=False,
    )

    reg.SetInterpolator(sitk.sitkLinear)

    return reg.Execute(
        source,
        target,
    )


def predict_single_pair(source_image, target_image, source_points):
    """Map points (x, y) on source_image into target_image by registering the breast masks."""
    transform = register_masks_bspline(breast_mask(source_image), breast_mask(target_image))

    return [transform.TransformPoint(point) for point in source_points]


def box_outline(box, step: float = OUTLINE_STEP) -> list[tuple[float, float]]:
    """Points (x, y) around a box (ymin, xmin, ymax, xmax), clockwise from the top left corner, at most step apart,
    each corner once."""
    ymin, xmin, ymax, xmax = box
    xs = np.linspace(xmin, xmax, int(np.ceil((xmax - xmin) / step)) + 1)
    ys = np.linspace(ymin, ymax, int(np.ceil((ymax - ymin) / step)) + 1)

    return ([(x, ymin) for x in xs] + [(xmax, y) for y in ys[1:]]
            + [(x, ymax) for x in xs[-2::-1]] + [(xmin, y) for y in ys[-2:0:-1]])


def predict_box(task: dict) -> tuple:
    """The bounding box (ymin, xmin, ymax, xmax) of the current box's outline mapped into the prior image."""
    current_image, prior_image = (np.asarray(Image.open(task[key])) for key in ('current_path', 'prior_path'))
    points = np.array(predict_single_pair(current_image, prior_image, box_outline(task['current_box'])))
    (xmin, ymin), (xmax, ymax) = points.min(axis=0), points.max(axis=0)

    return ymin, xmin, ymax, xmax


def lesion_pairs(data_dir: Path, splits: list[Split]) -> pd.DataFrame:
    """One row per pair of CC exams of a breast, a later (current) and an earlier (prior) one, and mammotracker lesion
    boxed in both exams."""
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
            prior, current = record.exams
            prior_boxes, current_boxes = (by_image.get((exam.exam_id, str(record.side))) for exam in record.exams)
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


def main(
    data_dir: Annotated[Path, Option(help='Output folder of scripts/misc/prepare_embed_dataset.py')] = Path(DATA_DIR),
    splits: Annotated[list[Split], Option(help='Splits to predict; repeat the option for several')]
        = [Split.TRAIN, Split.DEV, Split.TEST],
    output_dir: Annotated[Path, Option(help='Where predictions.csv goes')]
        = Path('./outputs/mere_mask_bspline'),
    workers: Annotated[int, Option(help='Parallel pairs')] = 16,
):
    pairs = lesion_pairs(data_dir, splits)
    print(f'{len(pairs)} lesion pairs:', pairs.split_group.value_counts().to_dict())

    tasks = [{'current_path': current_path, 'prior_path': prior_path, 'current_box': current_box}
             for current_path, prior_path, current_box
             in zip(pairs.current_path, pairs.prior_path, pairs[[f'current_{c}' for c in BOX]].values.tolist())]
    with Pool(workers) as pool:
        predicted = list(tqdm(pool.imap(predict_box, tasks), total=len(tasks), desc='predict', unit='pair'))
    pairs[[f'pred_{c}' for c in BOX]] = np.round(predicted, 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output_dir / 'predictions.csv', index=False)
    print(f'Wrote {output_dir / "predictions.csv"}')


if __name__ == '__main__':
    typer.run(main)
