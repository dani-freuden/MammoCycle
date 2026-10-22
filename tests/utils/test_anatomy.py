"""The vectorized anatomical coordinates against the per-point functions, on a synthetic CC breast."""
import numpy as np

from src.utils.anatomy import (anatomical_coords, boundary_radii, get_breast_mask, get_centroid, get_nipple_jas_cc,
                               image_coords, to_anatomical_coords)


def cc_breast(height=512, width=320):
    """A half ellipse against the chest wall on the right edge, with some texture."""
    y, x = np.mgrid[:height, :width]
    inside = ((x - width) / 250) ** 2 + ((y - height / 2) / 200) ** 2 < 1
    return (inside * (0.3 + 0.5 * np.random.default_rng(0).random((height, width)))).astype(np.float32)


def landmarks(image):
    mask = get_breast_mask(image)
    centroid = get_centroid(mask)
    return mask, centroid, get_nipple_jas_cc(mask, centroid)


def test_coordinates_agree_with_the_per_point_function():
    mask, centroid, nipple = landmarks(cc_breast())
    points = np.random.default_rng(1).random((100, 2)) * [320, 512]
    expected = np.stack([to_anatomical_coords(point, mask, centroid, nipple) for point in points])

    assert np.allclose(anatomical_coords(points, mask, centroid, nipple), expected, atol=0.01)


def test_nipple_and_centroid_have_their_coordinates():
    mask, centroid, nipple = landmarks(cc_breast())
    coords = anatomical_coords(np.array([nipple, centroid]), mask, centroid, nipple)

    assert np.allclose(coords, [[1, 0], [0, 0]], atol=0.02)


def test_image_coords_inverts_the_coordinates():
    mask, centroid, nipple = landmarks(cc_breast())
    points = np.random.default_rng(2).random((100, 2)) * [320, 512]
    coords = anatomical_coords(points, mask, centroid, nipple)

    assert np.allclose(image_coords(coords, centroid, nipple, boundary_radii(mask, centroid, 720)), points, atol=1e-6)


def test_rays_into_the_chest_wall_end_at_the_mirrored_skin_line():
    mask, centroid, nipple = landmarks(cc_breast())
    cx, cy = centroid
    width = mask.shape[1]
    skin = np.argmax(mask[cy])  # the skin line on the centroid's row, on the nipple side
    edge = (width - 1, cy)  # the chest wall on the same row, inside the image
    rho = (edge[0] - cx) / (2 * width - 1 - skin - cx)  # mirrored across the right edge, the skin is at 2w - 1 - skin

    assert np.isclose(np.hypot(*to_anatomical_coords(edge, mask, centroid, nipple)), rho, atol=0.01)
    assert np.isclose(np.hypot(*anatomical_coords(np.array([edge]), mask, centroid, nipple)[0]), rho, atol=0.01)
