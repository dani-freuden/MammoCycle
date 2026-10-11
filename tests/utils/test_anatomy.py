"""The vectorized anatomical coordinates against the per-point functions, on a synthetic CC breast."""
import numpy as np

from src.utils.anatomy import (anatomical_coords, anatomical_map, get_breast_mask, get_centroid, get_nipple_jas_cc,
                               to_anatomical_coords)


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


def test_map_holds_the_coordinates_of_the_cell_centres():
    image = cc_breast()
    mask, centroid, nipple = landmarks(image)
    anatomy = anatomical_map(image, stride=8)

    assert anatomy.shape == (2, 64, 40) and anatomy.dtype == np.float32
    for row, column in [(0, 0), (10, 30), (32, 20), (50, 39), (63, 5)]:
        center = ((column + 0.5) * 8 - 0.5, (row + 0.5) * 8 - 0.5)  # tracker convention, in index coordinates
        assert np.allclose(anatomy[:, row, column], to_anatomical_coords(center, mask, centroid, nipple), atol=0.01)
