import numpy as np
import cv2


def mask(image: np.ndarray, points: np.ndarray, cover: np.ndarray) -> np.ndarray:
    """
    Cover the license plate with the cover image, using homography of points
    :param image: Image to be covered
    :param points: Points of the license plate
    :param cover: Cover image
    :return: Covered image
    """
    width, height = cover.shape[1], cover.shape[0]
    # Find homography matrix
    homography_matrix, _ = cv2.findHomography(
        np.array([(0, 0), (width, 0), (width, height), (0, height)], dtype=np.float32), points)
    # Warp the cover image
    warped_cover = cv2.warpPerspective(cover, homography_matrix, (image.shape[1], image.shape[0]))
    # Create a mask for the cover image
    warped_cover_mask = cv2.warpPerspective(np.ones_like(cover, dtype=np.uint8) * -1, homography_matrix,
                                            (image.shape[1], image.shape[0]))
    # Blend the cover image with the original image using the mask
    blended_image = np.where(warped_cover_mask == -1, warped_cover, image)
    return blended_image


if __name__ == '__main__':
    cover_img_path = 'kntu.jpg'
    img_path = '1.jpg'
    points = np.array([
        (0.33375111849641503, 0.44584873554048715),
        (0.7320580100203828, 0.442254517965083),
        (0.7308046545311376, 0.5506102212267183),
        (0.3347763347763347, 0.5519230769230771)
    ])
    cover_img = cv2.imread(cover_img_path)
    img = cv2.imread(img_path)
    points_abs = np.array([(x * img.shape[1], y * img.shape[0]) for x, y in points])
    masked_img = mask(img, points_abs, cover_img)
    cv2.imshow('Masked Img', masked_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
