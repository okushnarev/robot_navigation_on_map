import math
from dataclasses import dataclass

import cv2 as cv

mm_in_px = 1180 / 528


def mm_to_px(dist_mm: float) -> int:
    return math.ceil(dist_mm / mm_in_px)


def px_to_mm(dist_px: int) -> float:
    return dist_px * mm_in_px


print(mm_to_px(450))


@dataclass
class Patch:
    center: tuple
    container: list


def draw_filled_square(img, center, side, color=(255, 0, 0)) -> Patch:
    """
    Draws a filled square on an image with the specified center, side length, and color.

    :param img: The image on which to draw the square.
    :param center: A tuple (x, y) representing the center of the square.
    :param side: The side length of the square.
    :param color: The color of the square in BGR format. Default is blue (255, 0, 0).

    :return: The portion of the original image that was replaced by the square.
    """

    assert side > 0 and side % 2 != 0, 'Value of square side should be odd and positive'

    # Calculate the top-left and bottom-right corners of the square
    x, y = center
    half_side = side // 2
    top_left = (x - half_side, y - half_side)
    bottom_right = (x + half_side, y + half_side)

    # Extract the original patch
    original_patch = img[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1].copy()

    # Draw the filled square on the image
    cv.rectangle(img, top_left, bottom_right, color, thickness=cv.FILLED)

    # Create and return the Patch object
    return Patch(center, original_patch)


def undo_patch(img, patch):
    """
    Places the original patch back onto the image at the specified center.

    :param img: The image on which to place the patch.
    :param patch: An instance of the Patch class containing the center and the original patch.
    """
    half_side = patch.container.shape[0] // 2
    top_left = (patch.center[0] - half_side, patch.center[1] - half_side)
    bottom_right = (patch.center[0] + half_side, patch.center[1] + half_side)

    # Place the patch back onto the image
    img[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1] = patch.container
