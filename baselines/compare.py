"""Compare evaluations: their metrics side by side, and their curves in one figure.

Each evaluation is a folder with an evaluate.py's evaluation.csv, named by the folder. Only the lesion pairs found in
every evaluation are compared, so one scored on fewer pairs (say only dev and test) cuts the others to the same pairs.
Prints the metrics of baselines.anatomical_coordinates.evaluate, one row per evaluation, and plots three panels:
    success     the share of pairs with IoU above a threshold; the area under the curve is the AO
    precision   the share of pairs whose box centres are within a distance, in mm (pairs with a PixelSpacing only)
    gap         the mean IoU per whole year between the two exams

Usage:
    python -m baselines.compare     # every outputs/*/evaluation.csv
    python -m baselines.compare outputs/anatomical_coordinates outputs/anatomical_coordinates_NCC
"""
from functools import reduce
from pathlib import Path
from typing import Annotated

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from typer import Argument, Option

from src.constants import ID_DTYPES, PAIR_KEY

from .anatomical_coordinates.evaluate import summarise


COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
MAX_MM = 40  # the precision panel's range


def load(folders: list[Path]) -> dict[str, pd.DataFrame]:
    """Each folder's evaluation.csv by the folder's name, cut to the pairs found in all of them."""
    evaluations = {folder.name: pd.read_csv(folder / 'evaluation.csv', dtype=ID_DTYPES).set_index(PAIR_KEY)
                   for folder in folders}
    common = reduce(pd.Index.intersection, [evaluation.index for evaluation in evaluations.values()])
    for name, evaluation in evaluations.items():
        if len(evaluation) > len(common):
            print(f'{name}: left out {len(evaluation) - len(common)} of its {len(evaluation)} pairs, not in all')

    return {name: evaluation.loc[common] for name, evaluation in evaluations.items()}


def plot(evaluations: dict[str, pd.DataFrame], path: Path):
    """The success, precision and gap panels, one line per evaluation."""
    thresholds, distances = np.linspace(0, 1, 101), np.linspace(0, MAX_MM, 101)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for (name, evaluation), color in zip(evaluations.items(), COLORS):
        iou, l2_mm = evaluation.iou.values, evaluation.l2_mm.dropna().values
        axes[0].plot(thresholds, (iou[:, None] > thresholds).mean(axis=0), color=color, lw=2,
                     label=f'{name} (AO {iou.mean():.3f})')
        axes[1].plot(distances, (l2_mm[:, None] <= distances).mean(axis=0), color=color, lw=2,
                     label=f'{name} (median {np.median(l2_mm):.1f} mm)')
        gap = (pd.to_datetime(evaluation.current_date) - pd.to_datetime(evaluation.prior_date)).dt.days / 365.25
        means = evaluation.iou.groupby(gap.round()).mean()
        axes[2].plot(means.index, means.values, color=color, lw=2, marker='o', ms=5, label=name)

    axes[0].set(title='success', xlabel='IoU threshold', ylabel='share of pairs with IoU above it')
    axes[1].set(title='precision', xlabel='centre distance (mm)', ylabel='share of pairs within it')
    axes[2].set(title='gap', xlabel='years between the exams', ylabel='mean IoU')
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main(
    folders: Annotated[list[Path] | None, Argument(help='Folders with an evaluation.csv; every outputs/*/ if none')]
        = None,
    out: Annotated[Path, Option(help='Where the figure goes')] = Path('./outputs/comparison.png'),
):
    folders = folders or sorted(path.parent for path in Path('./outputs').glob('*/evaluation.csv'))
    if len(folders) > len(COLORS):
        raise typer.BadParameter(f'at most {len(COLORS)} evaluations, one per color')
    evaluations = load(folders)
    table = pd.DataFrame({name: summarise(evaluation) for name, evaluation in evaluations.items()}).T
    print(table.astype({'pairs': int}).to_string(float_format='{:.3f}'.format))
    plot(evaluations, out)
    print(f'Wrote {out}')


if __name__ == '__main__':
    typer.run(main)
