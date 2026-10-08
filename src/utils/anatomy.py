import cv2
import numpy as np
from enum import Enum

import torch

from src.enums import View




class NippleStrategy(Enum):
    NAIVE = 'naive'
    JAS_CC = 'jas_cc'

class FrameKeys(Enum):
    NIPPLE = 'nipple'
    CENTROID = 'centroid'
    HEIGHT = 'height'
    U = 'u'
    V = 'v'
    D = 'd' 
    

def get_breast_mask(img, thres= 0.01):
    
    # normalize to 0-1 (it is now with max=img.max())
    img = img / img.max()
    img = img.astype(np.float32)
    
    mask = img > thres

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8)
    )

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = labels == largest

    return mask

def get_centroid(binary_img: np.ndarray):
    moments = cv2.moments(binary_img.astype(np.uint8))
    if moments["m00"] == 0:
        return None
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return cx, cy


def get_nipple_naive(mask,centroid):
    cx, cy = centroid
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    
    contour = max(contours, key=cv2.contourArea)[:, 0, :]  # [N, 2], x/y


    
    ys = np.where(mask)[0]
    breast_height = ys.max() - ys.min()

    window = 0.25 * breast_height

    candidates = contour[
        np.abs(contour[:, 1] - cy) < window
    ]

    nipple = candidates[np.argmin(candidates[:, 0])]
    x = contour[:, 0]
    y = contour[:, 1]

    x_norm = (x.max() - x) / (x.max() - x.min())
    y_penalty = np.abs(y - cy) / breast_height

    score = x_norm - 0.5 * y_penalty

    nipple = contour[np.argmax(score)]
    return nipple

def get_nipple_jas_cc(mask, centroid, chest_wall="right",
                  strip_width=3, search_band_fraction=0.20):
    """
    Approximation of the simplified Jas et al. nipple-search heuristic.

    mask: bool [H, W]
    centroid: (cx, cy)
    """

    H, W = mask.shape
    cx, cy = map(int, centroid)

    ys, xs = np.where(mask)
    breast_height = ys.max() - ys.min()

    # Limit nipple search to a band around the centroid.
    half_band = int(search_band_fraction * breast_height)

    y0 = max(0, cy - half_band)
    y1 = min(H, cy + half_band + 1)

    if chest_wall == "right":
        x_positions = range(cx, xs.min() - 1, -1)
    else:
        x_positions = range(cx, xs.max() + 1)

    last_foreground = None

    for x in x_positions:
        xa = max(0, x - strip_width // 2)
        xb = min(W, x + strip_width // 2 + 1)

        strip = mask[y0:y1, xa:xb]
        coords = np.argwhere(strip)

        if len(coords) > 0:
            # Convert local strip coords -> image coords
            yy = coords[:, 0] + y0
            xx = coords[:, 1] + xa
            last_foreground = (xx.mean(), yy.mean())

    return last_foreground



def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def get_boundary_radius(mask, centroid, direction, step=0.5):
    """
    Distance from centroid to breast boundary along `direction`.
    """
    c = np.asarray(centroid, dtype=float)
    direction = unit(direction)

    h, w = mask.shape

    r = 0.0
    last_inside_r = 0.0

    while True:
        p = c + r * direction
        x, y = np.round(p).astype(int)

        if x < 0 or x >= w or y < 0 or y >= h:
            break

        if not mask[y, x]:
            break

        last_inside_r = r
        r += step

    return last_inside_r


def to_anatomical_coords(point, mask, centroid, nipple):
    """
    Convert image point (x, y) to normalized anatomical coordinates:

        a = (rho * cos(theta), rho * sin(theta))

    where:
        theta = angle relative to centroid -> nipple axis
        rho   = fractional distance from centroid to breast boundary
                along the ray through the point
    """
    p = np.asarray(point, dtype=float)
    c = np.asarray(centroid, dtype=float)
    n = np.asarray(nipple, dtype=float)

    ref = unit(n - c)

    delta = p - c
    r = np.linalg.norm(delta)

    if r < 1e-6:
        return np.array([0.0, 0.0])

    direction = delta / r

    # signed angle relative to centroid -> nipple
    theta = np.arctan2(
        ref[0] * direction[1] - ref[1] * direction[0],
        np.dot(ref, direction),
    )

    boundary_r = get_boundary_radius(
        mask,
        centroid=c,
        direction=direction,
    )

    rho = r / max(boundary_r, 1e-6)

    return np.array([
        rho * np.cos(theta),
        rho * np.sin(theta),
    ])
    
    
def from_anatomical_coords(a, mask, centroid, nipple):
    """
    Convert normalized anatomical coordinates
        (rho*cos(theta), rho*sin(theta))
    back to image coordinates (x, y).
    """
    a = np.asarray(a, dtype=float)

    c = np.asarray(centroid, dtype=float)
    n = np.asarray(nipple, dtype=float)

    # Recover polar representation
    rho = np.linalg.norm(a)

    if rho < 1e-6:
        return c.copy()

    theta = np.arctan2(a[1], a[0])

    # centroid -> nipple is theta = 0
    ref = unit(n - c)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    direction = np.array([
        cos_t * ref[0] - sin_t * ref[1],
        sin_t * ref[0] + cos_t * ref[1],
    ])

    boundary_r = get_boundary_radius(
        mask,
        centroid=c,
        direction=direction,
    )

    return c + rho * boundary_r * direction

def map_prior_to_current(
    point,
    prior_mask,
    current_mask,
    prior_centroid,
    current_centroid,
    prior_nipple,
    current_nipple,
):
    a = to_anatomical_coords(
        point,
        prior_mask,
        prior_centroid,
        prior_nipple,
    )

    return from_anatomical_coords(
        a,
        current_mask,
        current_centroid,
        current_nipple,
    )
    
    
    

def boundary_radii(mask, centroid, num_angles, step=0.5):
    """get_boundary_radius in num_angles directions at once, direction k at angle 2 pi k / num_angles (y down).

    A polar resampling of the mask: row k is the ray in direction k, sampled every step pixels from the centroid.
    """
    h, w = mask.shape
    c = np.asarray(centroid, dtype=float)
    max_radius = max(np.hypot(x - c[0], y - c[1]) for x in (0, w) for y in (0, h)) + 1  # every ray leaves the image
    samples = int(np.ceil(max_radius / step))
    polar = cv2.warpPolar(mask.astype(np.uint8), (samples, num_angles), (float(c[0]), float(c[1])), max_radius,
                          cv2.WARP_POLAR_LINEAR | cv2.INTER_NEAREST | cv2.WARP_FILL_OUTLIERS)  # [num_angles, samples]
    first_outside = np.argmin(polar > 0, axis=1)

    return np.where(first_outside > 0, (first_outside - 1) * max_radius / samples, 0.0)


def anatomical_coords(points, mask, centroid, nipple, num_angles=720):
    """to_anatomical_coords for many points at once. points [N, 2] as (x, y) -> [N, 2].

    The boundary radius is interpolated between num_angles rays from the centroid instead of traced for every point.
    """
    c = np.asarray(centroid, dtype=float)
    ref = unit(np.asarray(nipple, dtype=float) - c)
    delta = np.asarray(points, dtype=float) - c

    angles = np.arange(num_angles) * 2 * np.pi / num_angles
    radius = np.interp(np.arctan2(delta[:, 1], delta[:, 0]), angles, boundary_radii(mask, c, num_angles),
                       period=2 * np.pi)

    # rho * (cos theta, sin theta) is the offset in the centroid -> nipple frame, divided by the boundary radius.
    along = delta @ ref
    across = ref[0] * delta[:, 1] - ref[1] * delta[:, 0]

    return np.stack([along, across], axis=-1) / np.maximum(radius, 1e-6)[:, None]


def anatomical_map(image, stride, threshold=0.01):
    """Anatomical coordinates of every stride x stride cell centre of a CC image [H, W] -> [2, H / stride, W / stride].

    CC only: the nipple search assumes a CC view with the chest wall on the right. Cell centres follow the tracker's
    convention (pixel i covers [i, i + 1)), half a pixel off the index coordinates used here.
    """
    mask = get_breast_mask(image, threshold)
    centroid = get_centroid(mask)
    nipple = get_nipple_jas_cc(mask, centroid)

    rows, columns = image.shape[0] // stride, image.shape[1] // stride
    y, x = np.mgrid[:rows, :columns]
    centers = (np.stack([x, y], axis=-1).reshape(-1, 2) + 0.5) * stride - 0.5

    return anatomical_coords(centers, mask, centroid, nipple).T.reshape(2, rows, columns).astype(np.float32)
