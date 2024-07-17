import numpy as np
import cv2


def mask(image: np.ndarray, points: np.ndarray, cover: np.ndarray) -> np.ndarray:
    """
    Cover the license plate with the cover image, using homography of points
    :param image: Image to be covered
    :param points: Normalized Points of the license plate
    :param cover: Cover image
    :return: Covered image
    """
    points_pixel = points * np.array([image.shape[1], image.shape[0]])

    width, height = cover.shape[1], cover.shape[0]
    # Find homography matrix
    homography_matrix, _ = cv2.findHomography(
        np.array([(0, 0), (width, 0), (width, height), (0, height)], dtype=np.float32), points_pixel)
    # Warp the cover image
    warped_cover = cv2.warpPerspective(cover, homography_matrix, (image.shape[1], image.shape[0]))
    # Create a mask for the cover image
    warped_cover_mask = cv2.warpPerspective(np.ones_like(cover, dtype=np.uint8) * -1, homography_matrix,
                                            (image.shape[1], image.shape[0]))
    # Blend the cover image with the original image using the mask
    blended_image = np.where(warped_cover_mask == -1, warped_cover, image)
    return blended_image
