"""Score the correction-regression baseline with the metrics of MammoTracker (arXiv:2507.00328).

The same metrics, tables and evaluation.csv as baselines.anatomical_coordinates.evaluate, on predict.py's
predictions.csv. The predicted box is the anatomy box moved, so its size, and with it the IoU, is still the anatomical
baseline's; the centre distance (l2_mm) is the model's own. With visualize_ratio above 0, that share of the pairs in
each IoU bin, and at least one, is drawn to visualizations/<bin>/: the current image with its box beside the prior
image with the true, the anatomy and the corrected box, each image with the outline of the window the model saw.

Usage:
    python -m baselines.correction_regression.evaluate
    python -m baselines.correction_regression.evaluate --visualize-ratio 0.1
"""
from functools import partial
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

from src.utils.anatomy import image_coords

from ..anatomical_coordinates.evaluate import evaluate, iou, visualize
from ..anatomical_coordinates.predict import BOX
from ..anatomical_coordinates_NCC.evaluate import STYLES
from .data.prepare import WINDOW, load_anatomy


def footprint(centroid, nipple, radii, centre) -> np.ndarray:
    """The outline (x, y) of the WINDOW x WINDOW window around the anatomical point centre (u, v), in the image."""
    side, half = np.linspace(-WINDOW / 2, WINDOW / 2, 50), np.full(50, WINDOW / 2)
    du = np.concatenate([side, half, side[::-1], -half])  # around the square, one edge after the other
    dv = np.concatenate([half, side[::-1], -half, side])

    return image_coords(np.stack([centre[0] + du, centre[1] + dv], axis=-1), centroid, nipple, radii)


def save_figure(pair, path: Path, anatomy: dict[str, tuple]):
    """The current image with its box, beside the prior image with the true, the anatomy and the corrected box, and on
    each image the outline of the window the model saw."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 5.5))
    for ax, exam in zip(axes, ('current', 'prior')):
        ax.imshow(np.asarray(Image.open(pair[f'{exam}_path'])), cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'{exam} {pair[f"{exam}_date"]}', fontsize=9)
        ax.axis('off')
    for ax, box in ((axes[0], 'current'), (axes[1], 'prior'), (axes[1], 'anat'), (axes[1], 'pred')):
        ymin, xmin, ymax, xmax = (pair[f'{box}_{c}'] for c in BOX)
        style, color = STYLES[box]
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, ec=color, ls=style, lw=1.5))
    for ax, exam in zip(axes, ('current', 'prior')):
        outline = footprint(*anatomy[pair[f'{exam}_path']], (pair.current_u, pair.current_v))
        ax.plot(*np.vstack([outline, outline[:1]]).T, color='white', ls=':', lw=1)

    anatomy_iou = iou([pair[f'anat_{c}'] for c in BOX], [pair[f'prior_{c}'] for c in BOX])
    fig.suptitle(f'patient {pair.patient_id} {pair.laterality} CC, {pair.kind} lesion {pair.lesion_id:.0f}, '
                 f'IoU {pair.iou:.2f} (anatomy {anatomy_iou:.2f}), '
                 f'correction ({WINDOW * pair.pred_du:+.3f}, {WINDOW * pair.pred_dv:+.3f}) in (u, v)\n'
                 f'yellow: current box; green: true prior box; blue dashed: anatomy box; red dashed: corrected box\n'
                 f'white dotted: the windows the model saw, {WINDOW} anatomical units wide', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main(
    output_dir: Annotated[Path, Option(help='Folder with predict.py\'s predictions.csv')]
        = Path('./outputs/correction_regression'),
    visualize_ratio: Annotated[float, Option(help='Share of the pairs in each IoU bin to draw, 0 to 1; '
                                                  'at least one per bin when above 0')] = 0.0,
    anatomy_npz: Annotated[Path, Option(help='anatomy.npz of data/prepare.py, for the windows in the drawings')]
        = Path('./outputs/correction_regression/data/anatomy.npz'),
):
    predictions = evaluate(output_dir)
    if visualize_ratio > 0:
        visualize(predictions, visualize_ratio, output_dir / 'visualizations',
                  partial(save_figure, anatomy=load_anatomy(anatomy_npz)))


if __name__ == '__main__':
    typer.run(main)
