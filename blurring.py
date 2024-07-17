import cv2
import numpy as np

from extract import extract
from masking import mask


def blur(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Blurs the license plate using the given points.
    :param image: The image to blur.
    :param points: The normalized points of the license plate.
    :return: The blurred image.
    """
    # Extract the license plate
    license_plate = extract(image, points)
    # Blur the license plate
    box_filter = np.ones((20, 20), np.float32) / 400
    blurred = cv2.filter2D(license_plate, -1, box_filter)
    # Mask the license plate
    masked_plate = mask(image, points, blurred)
    return masked_plate
