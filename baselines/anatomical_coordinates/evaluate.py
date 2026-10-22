"""Score the anatomical-coordinates baseline with the metrics of MammoTracker (arXiv:2507.00328).

Reads predictions.csv from predict.py and scores each predicted box against the true prior box:
    AO          average overlap: the mean IoU over all lesion pairs
    accuracy    the mean IoU over the successful pairs, those with IoU > 0
    robustness  the failure ratio: the share of pairs with IoU = 0
    l2_mm       the mean distance between the two box centres, in mm (only pairs whose prior image has a
                PixelSpacing, which the GE images lack)
Prints them per split and in total, and per lesion type over all pairs. Writes evaluation.csv next to it:
predictions.csv with an iou and an l2_mm column. Plots both against the years between the two exams to gap.png.

With visualize_ratio above 0, that share of the pairs in each IoU bin (IoU exactly 0, then BIN_WIDTH wide bins), and
at least one, is drawn to visualizations/<bin>/ (visualizations_<split>/<bin>/ with --split, which draws only that
split's pairs): the current image with its box beside the prior image with the true and the predicted box, every box
with its corners marked. The current box's outline is drawn where the anatomy maps it in the prior image, with its
corners numbered on both images; for this baseline the predicted box is that outline's bounding box, while other
methods scored here predict their own box.

Usage:
    python -m baselines.anatomical_coordinates.evaluate
    python -m baselines.anatomical_coordinates.evaluate --visualize-ratio 0.1
    python -m baselines.anatomical_coordinates.evaluate --output-dir outputs/stepD_matcher_mm --visualize-ratio 0.1 \
        --split test
"""
from pathlib import Path
from typing import Annotated

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from matplotlib.patches import Rectangle
from PIL import Image
from tqdm import tqdm
from typer import Option

from src.enums import Split

from .predict import BOX, box_outline, predict_single_pair


BIN_WIDTH = 0.2
SEED = 0  # for sampling the pairs to draw, so reruns draw the same ones
COLORS = {'current': '#ffd60a', 'prior': '#34c759', 'pred': '#ff3b30'}


def iou(a, b) -> float:
    """Intersection over union of two boxes (ymin, xmin, ymax, xmax)."""
    height = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    width = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda box: (box[2] - box[0]) * (box[3] - box[1])

    return height * width / (area(a) + area(b) - height * width)


def centre_distance(a, b) -> float:
    """Distance between the centres of two boxes (ymin, xmin, ymax, xmax), in their pixels."""
    return np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)


def summarise(pairs: pd.DataFrame) -> dict:
    return {'pairs': len(pairs), 'AO': pairs.iou.mean(), 'accuracy': pairs.iou[pairs.iou > 0].mean(),
            'robustness': (pairs.iou == 0).mean(), 'l2_mm': pairs.l2_mm.mean()}


def iou_bin(value: float) -> str:
    """iou_0 for a failure, else the BIN_WIDTH wide bin, e.g. iou_0.2-0.4; IoU 1 goes in the last bin."""
    if value == 0:
        return 'iou_0'
    low = min(int(value / BIN_WIDTH), round(1 / BIN_WIDTH) - 1) * BIN_WIDTH

    return f'iou_{low:.1f}-{low + BIN_WIDTH:.1f}'


def box_corners(ymin, xmin, ymax, xmax) -> list[tuple]:
    """The corners (x, y) of a box, clockwise from the top left."""
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


def save_figure(pair: pd.Series, path: Path):
    """The current image with its box, beside the prior image with the true and the predicted box.

    Every box has its corners marked. The current box's outline (predict.box_outline) is drawn where the anatomy maps
    it in the prior image, with its corners as crosses, numbered 1 to 4 as on the current box.
    """
    images = [np.asarray(Image.open(pair[f'{exam}_path'])) for exam in ('current', 'prior')]
    fig, axes = plt.subplots(1, 2, figsize=(8, 5.5))
    for ax, image, exam in zip(axes, images, ('current', 'prior')):
        ax.imshow(image, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'{exam} {pair[f"{exam}_date"]}', fontsize=9)
        ax.axis('off')
    for ax, box, style in ((axes[0], 'current', '-'), (axes[1], 'prior', '-'), (axes[1], 'pred', '--')):
        ymin, xmin, ymax, xmax = (pair[f'{box}_{c}'] for c in BOX)
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, ec=COLORS[box], ls=style, lw=1.5))
        ax.scatter(*zip(*box_corners(ymin, xmin, ymax, xmax)), s=12, color=COLORS[box], zorder=3)

    # The same outline predict.py maps; this baseline's predicted box is the bounding box of where it lands.
    current_box = [pair[f'current_{c}'] for c in BOX]
    outline = box_outline(current_box)
    mapped = np.array(predict_single_pair(*images, outline))
    axes[1].plot(*np.vstack([mapped, mapped[:1]]).T, color=COLORS['current'], lw=1)

    current_corners = box_corners(*current_box)
    mapped_corners = mapped[[outline.index(corner) for corner in current_corners]]
    axes[1].scatter(*mapped_corners.T, marker='x', s=30, color=COLORS['current'], zorder=3)
    for ax, corners in ((axes[0], current_corners), (axes[1], mapped_corners)):
        for number, (x, y) in enumerate(corners, 1):
            ax.annotate(str(number), (x, y), xytext=(3, 3), textcoords='offset points', color=COLORS['current'],
                        fontsize=7)

    fig.suptitle(f'patient {pair.patient_id} {pair.laterality} CC, {pair.kind} lesion {pair.lesion_id:.0f}, '
                 f'IoU {pair.iou:.2f}\nyellow: current box, and its outline mapped into the prior image by '
                 f'anatomical coordinates (x: corners)\n'
                 f'green: true prior box; red dashed: the predicted box', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def visualize(predictions: pd.DataFrame, ratio: float, out_dir: Path, draw=save_figure):
    """Draw ratio of the pairs in each IoU bin, and at least one, into out_dir/<bin>/ with draw(pair, path)."""
    bins = predictions.iou.map(iou_bin)
    sample = pd.concat([group.sample(max(1, round(ratio * len(group))), random_state=SEED)
                        for _, group in predictions.groupby(bins)])
    print('figures per bin:', sample.iou.map(iou_bin).value_counts().sort_index().to_dict())

    for _, pair in tqdm(sample.iterrows(), total=len(sample), desc='visualize', unit='pair'):
        path = (out_dir / iou_bin(pair.iou) / f'{pair.iou:.3f}_{pair.patient_id}_{pair.laterality}_'
                                              f'lesion{pair.lesion_id:.0f}_{pair.current_date}_{pair.prior_date}.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        draw(pair, path)


def plot_against_gap(predictions: pd.DataFrame, path: Path):
    """IoU and l2_mm of every pair against the years between its two exams, with the mean per whole year of gap."""
    gap = (pd.to_datetime(predictions.current_date) - pd.to_datetime(predictions.prior_date)).dt.days.abs() / 365.25
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, column, label in zip(axes, ('iou', 'l2_mm'), ('IoU', 'centre distance (mm)')):
        ax.scatter(gap, predictions[column], s=6, alpha=0.25, color='#2a78d6', linewidths=0)
        means = predictions[column].groupby(gap.round()).mean()
        ax.plot(means.index, means.values, color='#0b0b0b', lw=2, marker='o', ms=4, label='mean per year')
        ax.set(xlabel='years between the exams', ylabel=label)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def evaluate(output_dir: Path) -> pd.DataFrame:
    """Score output_dir/predictions.csv, print the tables and write evaluation.csv next to it."""
    ids = {'patient_id': str, 'prior_exam_id': str, 'current_exam_id': str}
    predictions = pd.read_csv(output_dir / 'predictions.csv', dtype=ids)
    predicted = predictions[[f'pred_{c}' for c in BOX]].values
    true = predictions[[f'prior_{c}' for c in BOX]].values
    predictions['iou'] = [iou(p, t) for p, t in zip(predicted, true)]
    distance = np.array([centre_distance(p, t) for p, t in zip(predicted, true)])
    predictions['l2_mm'] = distance * predictions.prior_mm_per_pixel

    splits = [str(split) for split in Split if str(split) in set(predictions.split_group)]
    groups = {split: predictions[predictions.split_group == split] for split in splits} | {'total': predictions}
    table = pd.DataFrame({name: summarise(group) for name, group in groups.items()}).T.astype({'pairs': int})
    print(table.to_string(float_format='{:.3f}'.format))
    by_kind = pd.DataFrame({kind: summarise(group) for kind, group in predictions.groupby('kind')}).T
    print('\n' + by_kind.astype({'pairs': int}).to_string(float_format='{:.3f}'.format))
    print(f'\nl2_mm over the {predictions.l2_mm.notna().sum()} of {len(predictions)} pairs whose prior image has a '
          f'PixelSpacing')

    predictions.to_csv(output_dir / 'evaluation.csv', index=False)

    return predictions


def main(
    output_dir: Annotated[Path, Option(help='Folder with predict.py\'s predictions.csv')]
        = Path('./outputs/anatomical_coordinates'),
    visualize_ratio: Annotated[float, Option(help='Share of the pairs in each IoU bin to draw, 0 to 1; '
                                                  'at least one per bin when above 0')] = 0.0,
    split: Annotated[Split | None, Option(help='Draw only this split\'s pairs, into visualizations_<split>')] = None,
):
    predictions = evaluate(output_dir)
    plot_against_gap(predictions, output_dir / 'gap.png')
    if visualize_ratio > 0:
        drawn = predictions if split is None else predictions[predictions.split_group == str(split)]
        visualize(drawn, visualize_ratio, output_dir / ('visualizations' if split is None else f'visualizations_{split}'))


if __name__ == '__main__':
    typer.run(main)
