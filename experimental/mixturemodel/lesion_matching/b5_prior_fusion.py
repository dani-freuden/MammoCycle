"""
B5: zero-shot local search fused with the anatomical prior.

    log p(x | images)  =  s(x) / tau  +  log p_anat(x)  + const

* s(x)          appearance score of placing the lesion at prior-image point x
                (NCC of the current-lesion template, or cosine similarity of
                pretrained features; anything that returns a ScoreMap works)
* p_anat(x)     where the anatomical mapping says the lesion should be: a
                Gaussian (+ small uniform outlier part) around the anatomical
                prediction, fitted on the *training* pairs' residuals and
                expressed in the local radial/tangential frame
* tau           temperature that converts the score into a log-likelihood.
                tau -> inf : prior only (= anatomical baseline + bias fix)
                tau -> 0   : appearance only (= plain template matching)
                Chosen on the validation split, never on test.

Typical use
-----------
    prior = AnatomicalPrior.fit(local_residuals_mm(train_cases))
    for c in val_cases:
        c["smap"] = ncc_score_map(c["img_cur_dog"], c["p_cur"], c["box_wh"],
                                  c["img_prior_dog"], c["p_anat"], spacing)
    table = sweep_tau(val_cases, prior, taus=np.logspace(-3, 0, 13), spacing=spacing)
    # pick tau with the best median error, then run once on test.

Each case is a dict with (all points in working-spacing pixels, prior image):
    p_cur (current lesion center, current image), box_wh (current box w, h),
    p_anat (anatomical prediction), centroid (prior breast centroid),
    p_gt (ground truth; only needed for fitting / evaluation),
    optional mask_prior (bool breast mask of the prior image).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import (apply_affine, compose, crop_affine, local_frame,
                      to_local_mm, warp_crop)


# --------------------------------------------------------------------------- #
# 1. The anatomical prior
# --------------------------------------------------------------------------- #
def local_residuals_mm(cases, spacing_mm: float) -> np.ndarray:
    """(N, 2) residuals p_gt - p_anat in the local (radial, tangential) frame, mm."""
    out = []
    for c in cases:
        F = local_frame(c["p_anat"], c["centroid"])
        d = np.asarray(c["p_gt"], float) - np.asarray(c["p_anat"], float)
        out.append(to_local_mm(d, F, spacing_mm)[0])
    return np.array(out)


@dataclass
class AnatomicalPrior:
    mu: np.ndarray          # (2,) mean residual, local frame, mm
    cov: np.ndarray         # (2, 2) covariance, local frame, mm^2
    eps: float              # weight of the uniform "anatomy failed" component
    area_mm2: float         # support of the uniform component

    @classmethod
    def fit(cls, r_mm: np.ndarray, area_mm2: float = 100.0 ** 2,
            n_iter: int = 100, min_var_mm2: float = 1.0) -> "AnatomicalPrior":
        """EM for a Gaussian + uniform-outlier mixture on local residuals.
        The uniform part keeps a few gross anatomical failures from inflating
        the Gaussian (which would make the prior too weak everywhere)."""
        r = np.asarray(r_mm, float)
        mu = np.median(r, axis=0)
        mad = 1.4826 * np.median(np.abs(r - mu), axis=0) + 1e-3
        cov = np.diag(mad ** 2)
        eps = 0.05
        for _ in range(n_iter):
            g = (1 - eps) * _gauss_pdf(r, mu, cov)
            u = eps / area_mm2
            w = g / (g + u)
            sw = w.sum()
            mu = (w[:, None] * r).sum(0) / sw
            z = r - mu
            cov = (w[:, None, None] * z[:, :, None] * z[:, None, :]).sum(0) / sw
            cov += min_var_mm2 * np.eye(2) * 1e-3
            eps = float(np.clip(1 - w.mean(), 1e-3, 0.5))
        return cls(mu=mu, cov=cov, eps=eps, area_mm2=area_mm2)

    def log_prob(self, pts_px, p_anat, centroid, spacing_mm: float) -> np.ndarray:
        """Log density (per mm^2) at image points `pts_px` (N, 2)."""
        F = local_frame(p_anat, centroid)
        d = np.atleast_2d(pts_px) - np.asarray(p_anat, float)
        loc = to_local_mm(d, F, spacing_mm)
        lg = np.log(1 - self.eps) + _gauss_logpdf(loc, self.mu, self.cov)
        lu = np.log(self.eps) - np.log(self.area_mm2)
        return np.logaddexp(lg, lu)

    def mode_px(self, p_anat, centroid, spacing_mm: float) -> np.ndarray:
        """Prior-only prediction = anatomical point + mean (bias) correction."""
        F = local_frame(p_anat, centroid)
        return np.asarray(p_anat, float) + F @ self.mu / spacing_mm

    def describe(self) -> str:
        sd = np.sqrt(np.diag(self.cov))
        rho = self.cov[0, 1] / (sd[0] * sd[1])
        return (f"bias (radial, tangential) = ({self.mu[0]:+.1f}, {self.mu[1]:+.1f}) mm | "
                f"sd = ({sd[0]:.1f}, {sd[1]:.1f}) mm | corr = {rho:+.2f} | "
                f"outlier weight = {self.eps:.2f}")


def _gauss_logpdf(x, mu, cov):
    z = np.atleast_2d(x) - mu
    icov = np.linalg.inv(cov)
    maha = np.einsum("ni,ij,nj->n", z, icov, z)
    return -0.5 * maha - np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(cov))


def _gauss_pdf(x, mu, cov):
    return np.exp(_gauss_logpdf(x, mu, cov))


# --------------------------------------------------------------------------- #
# 2. Appearance score maps
# --------------------------------------------------------------------------- #
@dataclass
class ScoreMap:
    score: np.ndarray       # (H, W) appearance score per candidate position
    M_grid: np.ndarray      # 2x3: grid (col, row) -> prior image px (lesion center)


def dog_filter(img: np.ndarray, spacing_mm: float,
               fine_mm: float = 0.4, coarse_mm: float = 8.0) -> np.ndarray:
    """Difference of Gaussians on the whole (working-spacing) image.
    Removes the slow thickness/density gradient that otherwise dominates NCC,
    and a little pixel noise. Run once per image and cache."""
    img = img.astype(np.float32)
    a = cv2.GaussianBlur(img, (0, 0), fine_mm / spacing_mm)
    b = cv2.GaussianBlur(img, (0, 0), coarse_mm / spacing_mm)
    return a - b


def template_side_px(box_wh, spacing_mm, margin=1.5, min_mm=12.0, max_mm=60.0) -> int:
    side_mm = np.clip(margin * max(box_wh) * spacing_mm, min_mm, max_mm)
    return int(round(side_mm / spacing_mm))


def ncc_score_map(img_cur, p_cur, box_wh, img_prior, p_center, spacing_mm,
                  radius_mm: float = 40.0, scales=(0.85, 1.0, 1.15),
                  margin: float = 1.5) -> ScoreMap:
    """Normalized cross-correlation of a template around the current lesion,
    searched within +-radius_mm of `p_center` (normally p_anat) on the prior.

    Multi-scale: the lesion may have grown or shrunk over the year, so the
    template is rescaled and the max over scales is kept at each position.
    The grid has one cell per working pixel and (2 * radius + 1)^2 cells.
    """
    T = template_side_px(box_wh, spacing_mm, margin)
    r = int(round(radius_mm / spacing_mm))
    tmpl = warp_crop(img_cur, crop_affine(p_cur, (T, T)), (T, T))

    best = None
    for s in scales:
        Ts = T + 2 * int(round((T * s - T) / 2))     # keep parity -> integer alignment
        t_s = cv2.resize(tmpl, (Ts, Ts), interpolation=cv2.INTER_LINEAR)
        S = Ts + 2 * r
        M_s = crop_affine(p_center, (S, S))
        search = warp_crop(img_prior, M_s, (S, S))
        res = cv2.matchTemplate(search.astype(np.float32), t_s.astype(np.float32),
                                cv2.TM_CCOEFF_NORMED)        # (2r+1, 2r+1)
        res = np.nan_to_num(res, nan=-1.0)
        best = res if best is None else np.maximum(best, res)

    # grid cell (j, i) <-> template top-left at (j, i) <-> template center at
    # (j + (Ts-1)/2, i + (Ts-1)/2) in the search crop. For every scale this is
    # the same image point, because S - Ts = 2r is scale independent.
    Ts1 = T
    M_search = crop_affine(p_center, (Ts1 + 2 * r, Ts1 + 2 * r))
    shift = np.array([[1, 0, (Ts1 - 1) / 2], [0, 1, (Ts1 - 1) / 2]], float)
    return ScoreMap(score=best, M_grid=compose(M_search, shift))


# --------------------------------------------------------------------------- #
# 3. Fusion and decoding
# --------------------------------------------------------------------------- #
def grid_points(smap: ScoreMap) -> np.ndarray:
    H, W = smap.score.shape
    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    return apply_affine(smap.M_grid, np.stack([jj.ravel(), ii.ravel()], 1))


def log_prior_map(smap: ScoreMap, prior: AnatomicalPrior, case, spacing_mm) -> np.ndarray:
    pts = grid_points(smap)
    lp = prior.log_prob(pts, case["p_anat"], case["centroid"], spacing_mm)
    lp = lp.reshape(smap.score.shape)
    mask = case.get("mask_prior")
    if mask is not None:                       # no lesion outside the breast
        xi = np.clip(np.round(pts[:, 0]).astype(int), 0, mask.shape[1] - 1)
        yi = np.clip(np.round(pts[:, 1]).astype(int), 0, mask.shape[0] - 1)
        lp = np.where(mask[yi, xi].reshape(lp.shape), lp, -np.inf)
    return lp


def decode(log_post: np.ndarray, M_grid: np.ndarray, mode: str = "map",
           refine_radius: int = 2):
    """Turn a (unnormalized) log posterior over the grid into an image point.

    mode="map":  argmax, then a soft-argmax in a small window around it for
                 sub-cell precision. Robust when the posterior is multimodal.
    mode="mean": posterior mean. Minimizes expected squared error, but can
                 land between two modes.
    Returns (point_px, info) where info has the peak probability and entropy
    (useful as a confidence score: low peak / high entropy -> unsure).
    """
    H, W = log_post.shape
    lp = log_post - np.max(log_post)
    p = np.exp(lp)
    p /= p.sum()
    if mode == "mean":
        jj, ii = np.meshgrid(np.arange(W), np.arange(H))
        g = np.array([(p * jj).sum(), (p * ii).sum()])
    else:
        i0, j0 = np.unravel_index(np.argmax(lp), lp.shape)
        r = refine_radius
        i_lo, i_hi = max(0, i0 - r), min(H, i0 + r + 1)
        j_lo, j_hi = max(0, j0 - r), min(W, j0 + r + 1)
        w = p[i_lo:i_hi, j_lo:j_hi]
        jj, ii = np.meshgrid(np.arange(j_lo, j_hi), np.arange(i_lo, i_hi))
        g = np.array([(w * jj).sum(), (w * ii).sum()]) / w.sum()
    nz = p[p > 0]
    info = {"peak_prob": float(p.max()), "entropy": float(-(nz * np.log(nz)).sum())}
    return apply_affine(M_grid, g)[0], info


def predict(smap: ScoreMap, prior: AnatomicalPrior, case, spacing_mm, tau,
            mode="map", lp_map=None):
    """tau=None -> appearance only; tau=np.inf -> prior only."""
    if lp_map is None:
        lp_map = log_prior_map(smap, prior, case, spacing_mm)
    if tau is None:
        log_post = np.where(np.isfinite(lp_map), smap.score, -np.inf)
    elif np.isinf(tau):
        log_post = lp_map
    else:
        log_post = smap.score / tau + lp_map
    return decode(log_post, smap.M_grid, mode=mode)


# --------------------------------------------------------------------------- #
# 4. Choosing tau on validation, and metrics
# --------------------------------------------------------------------------- #
def summarize(err_mm: np.ndarray) -> dict:
    e = np.asarray(err_mm, float)
    return {"n": len(e), "mean": e.mean(), "median": np.median(e),
            "tmean30": np.minimum(e, 30.0).mean(),   # failure-sensitive but bounded
            "p90": np.percentile(e, 90),
            "<5mm": (e < 5).mean(), "<10mm": (e < 10).mean(), "<20mm": (e < 20).mean()}


def bootstrap_ci(err_mm, stat=np.median, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    e = np.asarray(err_mm, float)
    vals = [stat(rng.choice(e, len(e))) for _ in range(n_boot)]
    return np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def sweep_tau(cases, prior: AnatomicalPrior, taus, spacing_mm, mode="map"):
    """Every case needs c["smap"] (precomputed). Returns rows for:
    anatomical point, prior mode (bias-corrected), appearance only, and each tau."""
    rows = []

    def err(pts):
        return np.array([np.linalg.norm(np.asarray(p) - np.asarray(c["p_gt"])) * spacing_mm
                         for p, c in zip(pts, cases)])

    lps = [log_prior_map(c["smap"], prior, c, spacing_mm) for c in cases]
    rows.append(("anatomical point", summarize(err([c["p_anat"] for c in cases]))))
    rows.append(("prior mode (bias-corrected)", summarize(err(
        [prior.mode_px(c["p_anat"], c["centroid"], spacing_mm) for c in cases]))))
    rows.append(("appearance only", summarize(err(
        [predict(c["smap"], prior, c, spacing_mm, None, mode, lp)[0]
         for c, lp in zip(cases, lps)]))))
    for tau in taus:
        pts = [predict(c["smap"], prior, c, spacing_mm, tau, mode, lp)[0]
               for c, lp in zip(cases, lps)]
        rows.append((f"fused tau={tau:.4g}", summarize(err(pts))))
    return rows


def choose_tau(rows, key: str = "tmean30") -> float:
    """Pick tau on validation. Do NOT select on the median: it ignores the
    catastrophic failures (wrong look-alike structure far away), so it always
    favors trusting appearance too much. A truncated mean or the <10 mm rate
    penalizes those failures."""
    fused = [(n, s) for n, s in rows if n.startswith("fused")]
    best = min(fused, key=lambda r: r[1][key]) if key != "<10mm" else \
        max(fused, key=lambda r: r[1][key])
    return float(best[0].split("=")[1])


def print_table(rows):
    print(f"{'method':32s} {'n':>4s} {'mean':>6s} {'tmean30':>7s} {'median':>7s} {'p90':>6s} "
          f"{'<5mm':>6s} {'<10mm':>6s} {'<20mm':>6s}")
    for name, s in rows:
        print(f"{name:32s} {s['n']:4d} {s['mean']:6.2f} {s['tmean30']:7.2f} {s['median']:7.2f} {s['p90']:6.2f} "
              f"{s['<5mm']:6.2f} {s['<10mm']:6.2f} {s['<20mm']:6.2f}")
