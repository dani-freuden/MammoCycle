"""Boxes as (ymin, xmin, ymax, xmax) and points as (x, y), in image pixels."""
import numpy as np


OUTLINE_STEP = 8  # PNG pixels between the points mapped along each edge of a box


def centre(box) -> np.ndarray:
    """The centre (x, y) of a box."""
    return np.array([box[1] + box[3], box[0] + box[2]]) / 2


def iou(a, b) -> float:
    """Intersection over union of two boxes."""
    height = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    width = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda box: (box[2] - box[0]) * (box[3] - box[1])

    return height * width / (area(a) + area(b) - height * width)


def centre_distance(a, b) -> float:
    """Distance between the centres of two boxes, in their pixels."""
    return np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)


def box_outline(box, step: float = OUTLINE_STEP) -> list[tuple[float, float]]:
    """Points (x, y) around a box, clockwise from the top left corner, at most step apart, each corner once."""
    ymin, xmin, ymax, xmax = box
    xs = np.linspace(xmin, xmax, int(np.ceil((xmax - xmin) / step)) + 1)
    ys = np.linspace(ymin, ymax, int(np.ceil((ymax - ymin) / step)) + 1)

    return ([(x, ymin) for x in xs] + [(xmax, y) for y in ys[1:]]
            + [(x, ymax) for x in xs[-2::-1]] + [(xmin, y) for y in ys[-2:0:-1]])
