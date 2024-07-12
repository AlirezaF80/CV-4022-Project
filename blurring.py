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


if __name__ == '__main__':
    img_path = '1.jpg'
    points = np.array([
        (0.33375111849641503, 0.44584873554048715),
        (0.7320580100203828, 0.442254517965083),
        (0.7308046545311376, 0.5506102212267183),
        (0.3347763347763347, 0.5519230769230771)
    ])
    img = cv2.imread(img_path)
    blurred_img = blur(img, points)
    cv2.imshow('Blurred Img', blurred_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
