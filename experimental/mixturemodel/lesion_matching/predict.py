"""B5 on the EMBED CC lesion pairs: NCC around the anatomical-coordinates point, fused with the anatomical prior.

Same pairs as baselines.anatomical_coordinates_NCC.predict (current box known, prior box predicted). p_anat is the
anatomy box's centre. The prior (b5_prior_fusion.AnatomicalPrior) is fitted on the train pairs' residuals, tau is
chosen on dev by the truncated mean, and every pair is then predicted with that tau: the anatomy box moved so its
centre lands on the decoded point. Train predictions are in-sample for the prior.

Writes predictions.csv to output_dir with the columns of the anatomical-coordinates baseline, plus anat_* (the anatomy
box) and tau.

Usage (from the repo root):
    python -m experimental.mixturemodel.lesion_matching.predict --workers 32
    python -m baselines.anatomical_coordinates.evaluate --output-dir outputs/b5_prior_fusion
"""
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from PIL import Image
from tqdm import tqdm
from typer import Option

from baselines.anatomical_coordinates.predict import predict_box as anatomy_predict
from src.constants import BOX, EMBED_DATA_DIR, SPACING_MM, Split
from src.data.lesion_pairs import DataDir, Reverse, lesion_pairs, pair_tasks
from src.utils.anatomy import breast_mask, get_centroid
from src.utils.boxes import centre

from .b5_prior_fusion import (AnatomicalPrior, choose_tau, dog_filter, local_residuals_mm, log_prior_map,
                             ncc_score_map, predict, print_table, summarize)


TAUS = np.logspace(-3, 0, 13)


def anatomy(task: dict) -> tuple:
    """The anatomy box (ymin, xmin, ymax, xmax) and the prior breast's centroid (x, y)."""
    return anatomy_predict(task), get_centroid(breast_mask(np.asarray(Image.open(task['prior_path']))))


def load_cases(data_dir: Path, workers: int, reverse: bool = False) -> tuple:
    """The lesion pairs of every split (lesion_pairs, reverse as there), their anatomy boxes [N, 4] and one B5 case
    per pair: p_anat, centroid and p_gt (x, y) in PNG pixels."""
    pairs = lesion_pairs(data_dir, list(Split), reverse)
    print(f'{len(pairs)} lesion pairs:', pairs.split_group.value_counts().to_dict())

    tasks = pair_tasks(pairs)
    with Pool(workers) as pool:
        anat, centroids = zip(*tqdm(pool.imap(anatomy, tasks), total=len(tasks), desc='anatomy', unit='pair'))
    anat = np.array(anat)
    cases = [{'p_anat': centre(box), 'centroid': np.array(centroid, float), 'p_gt': centre(true)}
             for box, centroid, true in zip(anat, centroids, pairs[[f'prior_{c}' for c in BOX]].values)]

    return pairs, anat, cases


def decode_all(smap, prior: AnatomicalPrior, case: dict, spacing_mm: float, taus) -> dict:
    """The point (x, y) of every method: anatomy, prior only, appearance only and fused per tau."""
    lp = log_prior_map(smap, prior, case, spacing_mm)
    points = {'anatomical point': case['p_anat'],
              'prior mode (bias-corrected)': prior.mode_px(case['p_anat'], case['centroid'], spacing_mm),
              'appearance only': predict(smap, prior, case, spacing_mm, None, lp_map=lp)[0]}
    for tau in taus:
        points[f'fused tau={tau:.4g}'] = predict(smap, prior, case, spacing_mm, tau, lp_map=lp)[0]

    return points


def report(pairs, cases: list, points: list) -> tuple:
    """Print every method on dev, choose tau there, and print anatomy, prior only, appearance only and the chosen
    fusion per split. points: decode_all per pair, in PNG pixels. Returns tau and the chosen points [N, 2]."""
    gt = np.array([c['p_gt'] for c in cases])
    predicted = {name: np.array([p[name] for p in points]) for name in points[0]}
    errors = {name: np.linalg.norm(p - gt, axis=1) * SPACING_MM for name, p in predicted.items()}

    print(f'\nDEV, mm at a nominal {SPACING_MM} mm/px')
    dev = (pairs.split_group == str(Split.DEV)).values
    rows = [(name, summarize(e[dev])) for name, e in errors.items()]
    print_table(rows)
    tau = choose_tau(rows)
    chosen = f'fused tau={tau:.4g}'
    for split in map(str, Split):
        in_split = (pairs.split_group == split).values
        print(f'\n{split.upper()}')
        print_table([(name, summarize(errors[name][in_split])) for name in [*list(errors)[:3], chosen]])

    return tau, predicted[chosen]


def write_predictions(pairs, anat: np.ndarray, cases: list, points: np.ndarray, tau: float, output_dir: Path):
    """predictions.csv: the anatomy box moved so its centre lands on each pair's point (x, y)."""
    dx, dy = (points - np.array([c['p_anat'] for c in cases])).T
    pairs[[f'anat_{c}' for c in BOX]] = np.round(anat, 1)
    pairs[[f'pred_{c}' for c in BOX]] = np.round(anat + np.stack([dy, dx, dy, dx], axis=1), 1)
    pairs['tau'] = tau

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output_dir / 'predictions.csv', index=False)
    print(f'Wrote {output_dir / "predictions.csv"}')


def match(task: dict) -> dict:
    """decode_all with the NCC score map of one pair."""
    current, prior = (np.asarray(Image.open(task[key])) for key in ('current_path', 'prior_path'))
    case = task['case'] | {'mask_prior': breast_mask(prior)}
    ymin, xmin, ymax, xmax = task['current_box']
    smap = ncc_score_map(dog_filter(current, SPACING_MM), centre(task['current_box']), (xmax - xmin, ymax - ymin),
                         dog_filter(prior, SPACING_MM), case['p_anat'], SPACING_MM)

    return decode_all(smap, task['prior'], case, SPACING_MM, TAUS)


def main(
    data_dir: DataDir = EMBED_DATA_DIR,
    output_dir: Annotated[Path, Option(help='Where predictions.csv goes')] = Path('./outputs/b5_prior_fusion'),
    workers: Annotated[int, Option(help='Parallel pairs')] = 16,
    reverse: Reverse = False,
):
    pairs, anat, cases = load_cases(data_dir, workers, reverse)
    train = (pairs.split_group == str(Split.TRAIN)).values
    prior = AnatomicalPrior.fit(local_residuals_mm([c for c, t in zip(cases, train) if t], SPACING_MM))
    print('prior:', prior.describe())

    tasks = [task | {'case': case, 'prior': prior} for task, case in zip(pair_tasks(pairs), cases)]
    with Pool(workers) as pool:
        points = list(tqdm(pool.imap(match, tasks), total=len(tasks), desc='match', unit='pair'))

    tau, chosen = report(pairs, cases, points)
    write_predictions(pairs, anat, cases, chosen, tau, output_dir)


if __name__ == '__main__':
    typer.run(main)
