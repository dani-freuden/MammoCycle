"""Score the correction-regression baseline with the metrics of MammoTracker (arXiv:2507.00328).

The same metrics, tables and evaluation.csv as baselines.anatomical_coordinates.evaluate, on predict.py's
predictions.csv. The predicted box is the anatomy box moved, so its size, and with it the IoU, is still the anatomical
baseline's; the centre distance (l2_mm) is the model's own. With visualize_ratio above 0, that share of the pairs in
each IoU bin, and at least one, is drawn to visualizations/<bin>/: the current image with its box beside the prior
image with the true, the anatomy and the corrected box, each image with the window the model saw.

Usage:
    python -m baselines.correction_regression.evaluate
    python -m baselines.correction_regression.evaluate --visualize-ratio 0.1
"""
from pathlib import Path
from typing import Annotated

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.patches import Rectangle
from PIL import Image
from typer import Option

from ..anatomical_coordinates.evaluate import evaluate, iou, visualize
from ..anatomical_coordinates.predict import BOX
from ..anatomical_coordinates_NCC.evaluate import STYLES
from .data.prepare import WINDOW


def save_figure(pair, path: Path):
    """The current image with its box, beside the prior image with the true, the anatomy and the corrected box, and on
    each image the window the model saw."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 5.5))
    for ax, exam in zip(axes, ('current', 'prior')):
        ax.imshow(np.asarray(Image.open(pair[f'{exam}_path'])), cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'{exam} {pair[f"{exam}_date"]}', fontsize=9)
        ax.axis('off')
    for ax, box in ((axes[0], 'current'), (axes[1], 'prior'), (axes[1], 'anat'), (axes[1], 'pred')):
        ymin, xmin, ymax, xmax = (pair[f'{box}_{c}'] for c in BOX)
        style, color = STYLES[box]
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, ec=color, ls=style, lw=1.5))
    for ax, (x, y) in ((axes[0], (pair.current_x, pair.current_y)), (axes[1], (pair.anat_x, pair.anat_y))):
        ax.add_patch(Rectangle((x - WINDOW / 2, y - WINDOW / 2), WINDOW, WINDOW, fill=False, ec='white', ls=':', lw=1))

    anatomy_iou = iou([pair[f'anat_{c}'] for c in BOX], [pair[f'prior_{c}'] for c in BOX])
    fig.suptitle(f'patient {pair.patient_id} {pair.laterality} CC, {pair.kind} lesion {pair.lesion_id:.0f}, '
                 f'IoU {pair.iou:.2f} (anatomy {anatomy_iou:.2f}), '
                 f'correction ({-WINDOW * pair.pred_dx:.0f}, {-WINDOW * pair.pred_dy:.0f}) px\n'
                 f'yellow: current box; green: true prior box; blue dashed: anatomy box; red dashed: corrected box\n'
                 f'white dotted: the {WINDOW} px windows the model saw', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main(
    output_dir: Annotated[Path, Option(help='Folder with predict.py\'s predictions.csv')]
        = Path('./outputs/correction_regression'),
    visualize_ratio: Annotated[float, Option(help='Share of the pairs in each IoU bin to draw, 0 to 1; '
                                                  'at least one per bin when above 0')] = 0.0,
):
    predictions = evaluate(output_dir)
    if visualize_ratio > 0:
        visualize(predictions, visualize_ratio, output_dir / 'visualizations', save_figure)


if __name__ == '__main__':
    typer.run(main)
