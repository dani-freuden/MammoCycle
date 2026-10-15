#%%
"""How far a lesion moves in normalised breast coordinates between two CC exams: Δa = a(prior box centre) −
a(current box centre).

a is to_anatomical_coords in each image's own anatomy (breast_mask, centroid, jas_cc nipple), as in the
anatomical-coordinates baseline. u runs along the centroid -> nipple axis and v across it. Both are fractions of the
boundary radius along the ray, so the centroid is the origin, the nipple (1, 0) and the breast boundary the unit
circle. Only the true boxes are used, so nothing is predicted. The pairs are every (later, earlier) EMBED CC lesion
pair of lesion_pairs, over all splits. Many pairs share a lesion, so each pair counts once, not each lesion.

Δa has no single size in mm, since a unit is the boundary radius along each ray. In mm it is measured in the prior
image, where the tracker searches: a(current) is placed there with from_anatomical_coords, and distance_mm is how far
it lands from the prior box centre, in prior pixels times the prior's mm per pixel. Pairs whose prior image has no
PixelSpacing (GE) have none.

Saves next to this file:
- anatomical_coords_residuals.csv: one row per pair, with a(current), a(prior), Δa, |Δa| and distance_mm.
- delta_distribution.png: the joint histogram of (Δu, Δv) with its marginals, and the histogram of |Δa| with its
  median and 90th and 95th percentiles.
- delta_distribution_mm.png: the histogram of distance_mm with its median and 90th and 95th percentiles.
- delta_location.png: |Δa| against where the lesion is, at the current box centre in one row and the prior one in the
  other: per pair, as a binned median map, and as the median against ρ and against θ.
The planes are drawn the way the images are: nipple on the left, the top of the image up.

Run from the repo root: python -m experimental.anatomical_coords_residuals (about half a minute on 32 workers).
"""
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Circle
from PIL import Image
from scipy.stats import binned_statistic_2d
from tqdm import tqdm

from baselines.anatomical_coordinates.predict import BOX, DATA_DIR, lesion_pairs
from src.data.dataset import breast_mask
from src.enums import Split
from src.utils.anatomy import from_anatomical_coords, get_centroid, get_nipple_jas_cc, to_anatomical_coords


OUT_DIR = Path('experimental')
WORKERS = 32
EDGES = np.linspace(-1.2, 1.2, 17)  # cells of the binned median map, 0.15 breast radii wide
MIN_PAIRS = 10  # cells of the map with fewer pairs are left empty
PROFILE_BINS = 10  # equal-count bins of ρ and θ
INK, MUTED, GRID, SURFACE, ACCENT = '#0b0b0b', '#898781', '#e1e0d9', '#fcfcfb', '#2a78d6'


def centre(box) -> tuple[float, float]:
    """The centre (x, y) of a box (ymin, xmin, ymax, xmax)."""
    ymin, xmin, ymax, xmax = box
    return (xmin + xmax) / 2, (ymin + ymax) / 2


def anatomy(image: np.ndarray) -> tuple:
    """The breast mask, centroid and nipple of an image, as to_anatomical_coords and from_anatomical_coords take them."""
    mask = breast_mask(image)
    centroid = get_centroid(mask)

    return mask, centroid, get_nipple_jas_cc(mask, centroid)


def calculate_anatomical_residuals(prior_img, current_img, lesion_prior, lesion_current) -> dict:
    """a(current box centre), a(prior box centre) and Δa, each (u, v) in its own image's anatomy, and distance_px: how
    far a(current), placed in the prior image, lands from the prior box centre, in prior pixels. Boxes are (ymin, xmin,
    ymax, xmax)."""
    prior_anatomy = anatomy(prior_img)
    prior = to_anatomical_coords(centre(lesion_prior), *prior_anatomy)
    current = to_anatomical_coords(centre(lesion_current), *anatomy(current_img))
    mapped = from_anatomical_coords(current, *prior_anatomy)

    return {'current_u': current[0], 'current_v': current[1], 'prior_u': prior[0], 'prior_v': prior[1],
            'du': prior[0] - current[0], 'dv': prior[1] - current[1],
            'distance_px': np.linalg.norm(mapped - centre(lesion_prior))}


def residuals_of_pair(task: tuple) -> dict:
    prior_path, current_path, prior_box, current_box = task
    prior_img, current_img = (np.asarray(Image.open(path)) for path in (prior_path, current_path))

    return calculate_anatomical_residuals(prior_img, current_img, prior_box, current_box)


#%% Δa of every pair
pairs = lesion_pairs(Path(DATA_DIR), list(Split))
tasks = list(zip(pairs.prior_path, pairs.current_path, pairs[[f'prior_{c}' for c in BOX]].values,
                 pairs[[f'current_{c}' for c in BOX]].values))
with Pool(WORKERS) as pool:
    results = list(tqdm(pool.imap(residuals_of_pair, tasks), total=len(tasks), unit='pair'))

pairs = pairs.join(pd.DataFrame(results))
pairs['magnitude'] = np.hypot(pairs.du, pairs.dv)
pairs['distance_mm'] = pairs.distance_px * pairs.prior_mm_per_pixel  # NaN without a PixelSpacing
columns = ['split_group', 'patient_id', 'laterality', 'lesion_id', 'kind', 'current_date', 'prior_date',
           'current_u', 'current_v', 'prior_u', 'prior_v', 'du', 'dv', 'magnitude', 'distance_mm']
pairs[columns].to_csv(OUT_DIR / 'anatomical_coords_residuals.csv', index=False)


#%% Statistics
quantiles = pairs.magnitude.quantile([0.5, 0.9, 0.95])
in_mm = pairs.distance_mm.dropna()
mm_quantiles = in_mm.quantile([0.5, 0.9, 0.95])
print(f'{len(pairs)} pairs ({pairs.groupby(["patient_id", "laterality", "lesion_id"]).ngroups} lesions)')
print(f'mean (du, dv) = ({pairs.du.mean():+.3f}, {pairs.dv.mean():+.3f}), std = ({pairs.du.std():.3f}, '
      f'{pairs.dv.std():.3f})')
print(f'|Δa| median {quantiles[0.5]:.3f}, 90th percentile {quantiles[0.9]:.3f}, 95th percentile {quantiles[0.95]:.3f}')
print(f'In the prior image, over the {len(in_mm)} pairs with a PixelSpacing: median {mm_quantiles[0.5]:.1f} mm, 90th '
      f'percentile {mm_quantiles[0.9]:.1f} mm, 95th percentile {mm_quantiles[0.95]:.1f} mm')


#%% Plots
def style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def breast_plane(ax, title: str):
    """Axes over the anatomical plane: the boundary as the dashed unit circle, the centroid and the nipple marked."""
    ax.add_patch(Circle((0, 0), 1, fill=False, ec=MUTED, ls='--', lw=1))
    ax.scatter([0, 1], [0, 0], marker='+', s=60, color=INK, zorder=3)
    ax.annotate('centroid', (0, 0), xytext=(4, -12), textcoords='offset points', color=INK, fontsize=7)
    ax.annotate('nipple', (1, 0), xytext=(4, -12), textcoords='offset points', color=INK, fontsize=7)
    ax.set(xlim=(EDGES[-1], EDGES[0]), ylim=(EDGES[0], EDGES[-1]), aspect='equal')  # nipple on the left
    ax.set_xlabel('u (left = toward the nipple)', fontsize=8, color=INK)
    ax.set_ylabel('v (up = image top)', fontsize=8, color=INK)
    ax.set_title(title, fontsize=9, color=INK)


def profile(ax, x: pd.Series, xlabel: str, title: str):
    """|Δa| of every pair against x, with its median and interquartile range in equal-count bins of x."""
    bins = pd.qcut(x, PROFILE_BINS)
    centres = x.groupby(bins, observed=True).median()
    spread = pairs.magnitude.groupby(bins, observed=True).quantile([0.25, 0.5, 0.75]).unstack()
    ax.scatter(x, pairs.magnitude, s=3, color=MUTED, alpha=0.3, lw=0)
    ax.fill_between(centres, spread[0.25], spread[0.75], color=ACCENT, alpha=0.25, lw=0, label='interquartile range')
    ax.plot(centres, spread[0.5], color=ACCENT, marker='o', ms=3, label='median')
    ax.set_xlabel(xlabel, fontsize=8, color=INK)
    ax.set_ylabel('|Δa|', fontsize=8, color=INK)
    ax.set_title(title, fontsize=9, color=INK)
    ax.legend(loc='upper right', fontsize=7, frameon=False)


def percentile_histogram(ax, values: pd.Series, quantiles: pd.Series, unit: str, digits: int):
    """A histogram of values, with lines at the median and the 90th and 95th percentiles."""
    ax.hist(values, bins=80, color=ACCENT, alpha=0.8)
    for q, ls, label in ((0.5, '-', 'median'), (0.9, '--', '90th percentile'), (0.95, ':', '95th percentile')):
        ax.axvline(quantiles[q], color=INK, lw=1.2, ls=ls, label=f'{label} {quantiles[q]:.{digits}f} {unit}')
    ax.set_ylabel('pairs', fontsize=9, color=INK)
    ax.legend(loc='upper right', fontsize=8, frameon=False)


# Distribution of Δa
fig, axes = plt.subplot_mosaic([['top', '.', 'magnitude'], ['joint', 'right', 'magnitude']], figsize=(14, 6.5),
                               width_ratios=[4, 1, 5], height_ratios=[1, 4], facecolor=SURFACE, layout='constrained')
joint, top, right, magnitude = axes['joint'], axes['top'], axes['right'], axes['magnitude']
top.sharex(joint)
right.sharey(joint)

limit = 1.05 * pairs[['du', 'dv']].abs().values.max()
bins = np.linspace(-limit, limit, 81)
*_, image = joint.hist2d(pairs.du, pairs.dv, bins=bins, cmap='viridis', norm=LogNorm(), cmin=1)
fig.colorbar(image, ax=joint, location='left', label='pairs per bin', shrink=0.8)
joint.axhline(0, color=GRID, lw=1, zorder=0)
joint.axvline(0, color=GRID, lw=1, zorder=0)
joint.invert_xaxis()  # Δu toward the nipple on the left, as in the images
joint.set_aspect('equal', adjustable='box')
joint.set_xlabel('Δu (left = toward the nipple)', fontsize=9, color=INK)
joint.set_ylabel('Δv (up = toward the image top)', fontsize=9, color=INK)
for ax, values, orientation, line in ((top, pairs.du, 'vertical', top.axvline), (right, pairs.dv, 'horizontal',
                                                                                  right.axhline)):
    ax.hist(values, bins=bins, orientation=orientation, color=ACCENT, alpha=0.8)
    line(values.mean(), color=INK, lw=1)
    for sign in (-1, 1):
        line(values.mean() + sign * values.std(), color=INK, lw=1, ls=':')
top.set_title(f'Δu: mean {pairs.du.mean():+.3f}, std {pairs.du.std():.3f} (line: mean, dotted: ±1 std)', fontsize=8,
              color=INK)
right.set_title(f'Δv: mean {pairs.dv.mean():+.3f}\nstd {pairs.dv.std():.3f}', fontsize=8, color=INK)
top.tick_params(labelbottom=False)
right.tick_params(labelleft=False)

percentile_histogram(magnitude, pairs.magnitude, quantiles, '', 3)
magnitude.set_xlabel('|Δa| (breast radii)', fontsize=9, color=INK)
magnitude.set_title('|Δa|', fontsize=10, color=INK)
for ax in axes.values():
    style(ax)
fig.suptitle(f'Δa = a(prior box centre) − a(current box centre), all {len(pairs)} EMBED CC lesion pairs, each in its '
             f'own image\'s normalised breast coordinates', fontsize=10, color=INK)
fig.savefig(OUT_DIR / 'delta_distribution.png', dpi=130, facecolor=SURFACE)

# Δa in mm
fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE, layout='constrained')
percentile_histogram(ax, in_mm, mm_quantiles, 'mm', 1)
ax.set_xlabel('distance from a(current), placed in the prior image, to the prior box centre (mm)', fontsize=9,
              color=INK)
style(ax)
ax.set_title(f'Δa in mm, in the prior image: {len(in_mm)} of the {len(pairs)} pairs\n(the {len(pairs) - len(in_mm)} '
             f'whose prior image has no PixelSpacing, all GE, are left out)', fontsize=10, color=INK)
fig.savefig(OUT_DIR / 'delta_distribution_mm.png', dpi=130, facecolor=SURFACE)

# |Δa| against where the lesion is
fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor=SURFACE, layout='constrained')
norm = Normalize(0, quantiles[0.95])
for row, where in zip(axes, ('current', 'prior')):
    u, v = pairs[f'{where}_u'], pairs[f'{where}_v']
    scatter, binned, along_rho, along_theta = row

    order = np.argsort(pairs.magnitude.values)  # the largest on top
    points = scatter.scatter(u.values[order], v.values[order], c=pairs.magnitude.values[order], s=8, cmap='viridis',
                             norm=norm, lw=0)
    breast_plane(scatter, f'|Δa| of every pair, at the {where} box centre')

    median = binned_statistic_2d(u, v, pairs.magnitude, 'median', bins=[EDGES, EDGES]).statistic
    count = binned_statistic_2d(u, v, None, 'count', bins=[EDGES, EDGES]).statistic
    binned.pcolormesh(EDGES, EDGES, np.where(count >= MIN_PAIRS, median, np.nan).T, cmap='viridis', norm=norm)
    breast_plane(binned, f'median |Δa| per cell (≥ {MIN_PAIRS} pairs), at the {where} box centre')

    profile(along_rho, np.hypot(u, v), 'ρ: distance from the centroid (breast radii)',
            f'|Δa| against ρ at the {where} box centre')
    profile(along_theta, np.degrees(np.arctan2(v, u)), 'θ (°): 0 toward the nipple, +90 image top, ±180 chest wall',
            f'|Δa| against θ at the {where} box centre')
fig.colorbar(points, ax=axes[:, :2], label='|Δa| (colours capped at the 95th percentile)', extend='max', shrink=0.6)
for ax in axes.flat:
    style(ax)
fig.suptitle(f'|Δa| against breast location, all {len(pairs)} EMBED CC lesion pairs: at the current box centre (top) '
             f'and the prior one (bottom)', fontsize=10, color=INK)
fig.savefig(OUT_DIR / 'delta_location.png', dpi=110, facecolor=SURFACE)
plt.show()
