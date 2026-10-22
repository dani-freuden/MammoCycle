from dataclasses import dataclass

import cv2
import numpy as np


def breast_mask(image: np.ndarray) -> np.ndarray:
    """The breast in an 8-bit image: the largest contour of a triangle threshold, filled, as a boolean mask.

    The triangle method puts the threshold at the foot of the dominant histogram peak, the background, so it adapts
    when the background is not at zero. Otsu does not do here: on about a quarter of the EMBED images its threshold
    falls inside the breast and leaves out the fat and the skin.
    """
    _, binary = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 1, thickness=cv2.FILLED)

    return mask.astype(bool)


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

    The mask is mirrored across its right edge, the chest wall, so a ray toward the chest wall runs on into the
    mirrored breast and ends at the mirrored skin line, not at the image edge.
    """
    c = np.asarray(centroid, dtype=float)
    direction = unit(direction)

    h, w = mask.shape

    r = 0.0
    last_inside_r = 0.0

    while True:
        p = c + r * direction
        x, y = np.round(p).astype(int)

        if x < 0 or x >= 2 * w or y < 0 or y >= h:
            break

        if not mask[y, x if x < w else 2 * w - 1 - x]:
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

def boundary_radii(mask, centroid, num_angles, step=0.5):
    """get_boundary_radius in num_angles directions at once, direction k at angle 2 pi k / num_angles (y down).

    A polar resampling of the mask, mirrored across its right edge as in get_boundary_radius: row k is the ray in
    direction k, sampled every step pixels from the centroid.
    """
    mask = np.hstack([mask, mask[:, ::-1]])
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


def image_coords(a, centroid, nipple, radii):
    """The inverse of anatomical_coords: anatomical coordinates a [..., 2] -> image points (x, y) [..., 2].

    radii are boundary_radii(mask, centroid, num_angles), any num_angles.
    """
    a = np.asarray(a, dtype=float)
    c = np.asarray(centroid, dtype=float)
    ref = unit(np.asarray(nipple, dtype=float) - c)
    angle = np.arctan2(ref[1], ref[0]) + np.arctan2(a[..., 1], a[..., 0])  # in the image, y down
    radius = np.interp(angle, np.arange(len(radii)) * 2 * np.pi / len(radii), radii, period=2 * np.pi)
    distance = np.hypot(a[..., 0], a[..., 1]) * radius

    return c + distance[..., None] * np.stack([np.cos(angle), np.sin(angle)], axis=-1)


@dataclass(frozen=True)
class BreastAnatomy:
    """The frame of an 8-bit CC image's anatomical coordinates, chest wall on the right: its breast mask, centroid and
    jas_cc nipple."""
    mask: np.ndarray
    centroid: tuple
    nipple: tuple

    @classmethod
    def of(cls, image: np.ndarray) -> 'BreastAnatomy':
        mask = breast_mask(image)
        centroid = get_centroid(mask)
        return cls(mask, centroid, get_nipple_jas_cc(mask, centroid))

    def to_anatomical(self, point) -> np.ndarray:
        return to_anatomical_coords(point, self.mask, self.centroid, self.nipple)

    def from_anatomical(self, a) -> np.ndarray:
        return from_anatomical_coords(a, self.mask, self.centroid, self.nipple)

    def map_to(self, other: 'BreastAnatomy', point) -> np.ndarray:
        """Where a point (x, y) of this image lands in the other one, through anatomical coordinates."""
        return other.from_anatomical(self.to_anatomical(point))
