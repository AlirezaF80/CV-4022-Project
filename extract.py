import cv2
import numpy as np


def extract(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Extract the license plate from the image using homography of points
    :param image: Image to be extracted
    :param points: Normalized Points of the license plate
    :return: Extracted image with aspect ratio of 4.5
    """
    points_pixel = points * np.array([image.shape[1], image.shape[0]])

    width = int(np.linalg.norm(points_pixel[0] - points_pixel[1]))
    height = int(width / 4.5)
    # Find homography matrix
    homography_matrix, _ = cv2.findHomography(
        points_pixel, np.array([(0, 0), (width, 0), (width, height), (0, height)], dtype=np.float32))
    # Warp the image
    warped_image = cv2.warpPerspective(image, homography_matrix, (width, height))
    return warped_image
