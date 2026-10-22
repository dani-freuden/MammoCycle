"""
Shared geometry for B5 (zero-shot search + prior) and Step D (learned matcher).

Conventions (used everywhere in this package)
---------------------------------------------
* Points are (x, y) = (column, row), in pixels, with the pixel-center-at-integer
  convention (pixel j has its center at x = j). This is what OpenCV uses.
* Images are assumed to be pre-resampled to one *working spacing* (mm/px), e.g.
  0.2 mm/px, with the chest wall on the same side for all images. Do the
  resampling once per image with `resample_to_spacing` (area interpolation,
  so no aliasing) and cache the result; crops are then taken at scale ~1.
* Every crop is described by a 2x3 affine matrix M that maps
  crop pixel (u, v) -> image pixel (x, y). Keeping M around means any
  location found in a crop (or in a score map over a crop) can be mapped back
  to the full image exactly, including after rotation/scale augmentation.
"""
from __future__ import annotations

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Resampling to a fixed physical spacing
# --------------------------------------------------------------------------- #
def resample_to_spacing(img: np.ndarray, spacing_mm: float, target_mm: float):
    """Resize a full image to `target_mm` per pixel. Returns (img, f) where
    f = spacing_mm / target_mm is the scale factor (new px per old px)."""
    f = spacing_mm / target_mm
    h, w = img.shape[:2]
    new_wh = (max(1, int(round(w * f))), max(1, int(round(h * f))))
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, new_wh, interpolation=interp), f


def rescale_points(pts: np.ndarray, f: float) -> np.ndarray:
    """Map points from the original image to an image resized by factor f
    with cv2.resize (OpenCV maps pixel centers: x' = (x + 0.5) * f - 0.5)."""
    return (np.asarray(pts, dtype=np.float64) + 0.5) * f - 0.5


# --------------------------------------------------------------------------- #
# Affine crops
# --------------------------------------------------------------------------- #
def crop_affine(center_xy, out_wh, scale: float = 1.0, angle_deg: float = 0.0) -> np.ndarray:
    """2x3 matrix mapping crop pixel (u, v) -> image pixel (x, y).

    The crop's center pixel ((W-1)/2, (H-1)/2) lands on `center_xy`.
    `scale` > 1 zooms in (one crop pixel covers 1/scale image pixels).
    `angle_deg` rotates the sampling grid (counter-clockwise in image coords).
    """
    w, h = out_wh
    th = np.deg2rad(angle_deg)
    s = 1.0 / scale
    R = s * np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th),  np.cos(th)]])
    c_crop = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    t = np.asarray(center_xy, dtype=np.float64) - R @ c_crop
    return np.hstack([R, t[:, None]])


def warp_crop(img: np.ndarray, M: np.ndarray, out_wh, border_value: float = 0.0) -> np.ndarray:
    """Sample `img` on the crop grid defined by M (crop px -> image px)."""
    return cv2.warpAffine(
        img, M.astype(np.float64), tuple(int(v) for v in out_wh),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


def to_h(M: np.ndarray) -> np.ndarray:
    """2x3 -> 3x3 homogeneous."""
    return np.vstack([M, [0.0, 0.0, 1.0]])


def compose(*Ms) -> np.ndarray:
    """compose(A, B) = A ∘ B (apply B first). Inputs/outputs are 2x3."""
    out = np.eye(3)
    for M in Ms:
        out = out @ to_h(M)
    return out[:2]


def invert_affine(M: np.ndarray) -> np.ndarray:
    return np.linalg.inv(to_h(M))[:2]


def apply_affine(M: np.ndarray, pts) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    return pts @ M[:, :2].T + M[:, 2]


# --------------------------------------------------------------------------- #
# Local anatomical frame at a point
# --------------------------------------------------------------------------- #
def local_frame(p_xy, centroid_xy) -> np.ndarray:
    """2x2 matrix F whose columns are the radial (away from the breast
    centroid) and tangential unit vectors at p. A displacement d in image
    coordinates has local coordinates F.T @ d = (d·e_r, d·e_t)."""
    v = np.asarray(p_xy, dtype=np.float64) - np.asarray(centroid_xy, dtype=np.float64)
    n = np.linalg.norm(v)
    e_r = v / n if n > 1e-6 else np.array([1.0, 0.0])
    e_t = np.array([-e_r[1], e_r[0]])
    return np.stack([e_r, e_t], axis=1)


def to_local_mm(d_img_px: np.ndarray, F: np.ndarray, spacing_mm: float) -> np.ndarray:
    """Image-frame displacement(s) in px -> local (radial, tangential) in mm."""
    return (np.atleast_2d(d_img_px) * spacing_mm) @ F
