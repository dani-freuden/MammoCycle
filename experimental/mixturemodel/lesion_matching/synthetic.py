"""
Synthetic 'mammogram-like' pairs with known correspondence, for unit-testing
the pipelines (and, in Step E, for self-supervised pretraining on single real
images: replace `make_texture` with a real image).

prior(x) = current(x + d(x)) * gain + other tissue + noise, where d is a small
affine + smooth elastic displacement. The lesion center in the prior is the
x that solves x + d(x) = p_cur.
"""
from __future__ import annotations

import cv2
import numpy as np


def make_texture(h, w, rng, scales_px=(2, 6, 15, 40), weights=(0.6, 1.0, 0.8, 1.2)):
    img = np.zeros((h, w), np.float32)
    for s, wt in zip(scales_px, weights):
        n = rng.standard_normal((h, w)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), s)
        img += wt * n / (n.std() + 1e-6)
    return img


def add_blob(img, center, sigma_px, amp):
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    g = np.exp(-((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2 * sigma_px ** 2))
    return img + amp * g.astype(np.float32)


def random_displacement(h, w, rng, spacing_mm, rot_deg=3.0, scale=0.05,
                        trans_mm=5.0, elastic_mm=4.0, grid=(5, 6)):
    """Displacement field d (h, w, 2) in px: affine part + smooth elastic part."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    c = np.array([w / 2, h / 2], np.float32)
    th = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
    s = 1 + rng.uniform(-scale, scale)
    A = s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
    t = rng.uniform(-trans_mm, trans_mm, 2) / spacing_mm
    X = np.stack([xx - c[0], yy - c[1]], -1)
    d_aff = X @ A.T + c + t - np.stack([xx, yy], -1)
    coarse = rng.standard_normal((grid[0], grid[1], 2)).astype(np.float32)
    d_el = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    d_el *= elastic_mm / spacing_mm
    return (d_aff + d_el).astype(np.float32)


def warp_with(img, d):
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(img, xx + d[..., 0], yy + d[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def bilinear(field, p):
    x, y = p
    h, w = field.shape[:2]
    x0, y0 = int(np.clip(np.floor(x), 0, w - 2)), int(np.clip(np.floor(y), 0, h - 2))
    fx, fy = x - x0, y - y0
    f = field
    return ((1 - fx) * (1 - fy) * f[y0, x0] + fx * (1 - fy) * f[y0, x0 + 1]
            + (1 - fx) * fy * f[y0 + 1, x0] + fx * fy * f[y0 + 1, x0 + 1])


def invert_point(d, p, n_iter=50):
    """Solve x + d(x) = p by fixed-point iteration."""
    x = np.asarray(p, np.float64).copy()
    for _ in range(n_iter):
        x = np.asarray(p, np.float64) - bilinear(d, x)
    return x


def make_case(rng, spacing_mm=0.2, h=600, w=500, lesion_sd_mm=3.0, lesion_amp=1.5,
              superpos=0.7, anat_mu=(-3.0, 0.0), anat_sd=(5.0, 9.0), p_outlier=0.1):
    from geometry import local_frame
    cur = make_texture(h, w, rng, scales_px=tuple(v * 0.2 / spacing_mm for v in (2, 6, 15, 40)))
    margin = 0.2 * min(h, w)
    p_cur = np.array([rng.uniform(margin, w - margin), rng.uniform(margin, h - margin)])
    sd_px = lesion_sd_mm / spacing_mm * rng.uniform(0.7, 1.4)
    cur = add_blob(cur, p_cur, sd_px, lesion_amp)
    d = random_displacement(h, w, rng, spacing_mm)
    prior = warp_with(cur, d)
    prior = 0.8 * prior + superpos * make_texture(
        h, w, rng, scales_px=tuple(v * 0.2 / spacing_mm for v in (2, 6, 15, 40)))       # tissue superposition
    prior += 0.3 * rng.standard_normal(prior.shape).astype(np.float32)
    p_gt = invert_point(d, p_cur)
    centroid = np.array([-0.5 * w, h / 2])   # far away -> stable local frame in this toy
    F = local_frame(p_gt, centroid)
    if rng.random() < p_outlier:
        e_mm = rng.uniform(-30, 30, 2)
    else:
        e_mm = np.array(anat_mu) + np.array(anat_sd) * rng.standard_normal(2)
    p_anat = p_gt + F @ (e_mm / spacing_mm)
    box = np.array([4 * sd_px, 4 * sd_px])
    return {"img_cur": cur, "img_prior": prior, "p_cur": p_cur, "box_wh": box,
            "p_gt": p_gt, "p_anat": p_anat, "centroid": centroid,
            "mask_prior": np.ones((h, w), bool), "disp": d}
