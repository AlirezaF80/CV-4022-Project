import os

import cv2
import numpy as np

input_shape = (224, 224, 3)


def augment(image: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aug_image, aug_points = image.copy(), points.copy()

    # Slight shifts
    max_shift = 0.2
    max_shift_x = min(1 - points[:, 0].max(), max_shift)
    min_shift_x = max(0 - points[:, 0].min(), -max_shift)
    shift_x = np.random.uniform(min_shift_x, max_shift_x) * image.shape[1]
    max_shift_y = min(1 - points[:, 1].max(), max_shift)
    min_shift_y = max(0 - points[:, 1].min(), -max_shift)
    shift_y = np.random.uniform(min_shift_y, max_shift_y) * image.shape[0]
    M_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    aug_image = cv2.warpAffine(aug_image, M_shift, (aug_image.shape[1], aug_image.shape[0]))
    aug_points[:, 0] += shift_x / image.shape[1]
    aug_points[:, 1] += shift_y / image.shape[0]

    # Blurs
    ksize = np.random.choice([3, 5, 7, 11])
    aug_image = cv2.GaussianBlur(aug_image, (ksize, ksize), 0)

    # Noise Injection
    noise = np.random.randn(*aug_image.shape) * 25  # Adjust noise level
    aug_image = np.clip(aug_image + noise, 0, 255).astype(np.uint8)

    # Crops
    crop_fraction = 0.1  # crop up to 10% of the image dimensions
    crop_x = int(crop_fraction * image.shape[1])
    crop_y = int(crop_fraction * image.shape[0])
    x1, y1 = np.random.randint(0, crop_x), np.random.randint(0, crop_y)
    x2, y2 = aug_image.shape[1] - np.random.randint(0, crop_x), aug_image.shape[0] - np.random.randint(0, crop_y)
    aug_image, aug_points = crop_image(aug_image, aug_points, x1, x2, y1, y2)

    # Rotations
    angle = np.random.uniform(-15, 15)
    M_rot = cv2.getRotationMatrix2D((aug_image.shape[1] / 2, aug_image.shape[0] / 2), angle, 1)
    aug_image = cv2.warpAffine(aug_image, M_rot, (aug_image.shape[1], aug_image.shape[0]))
    ones = np.ones(shape=(len(aug_points), 1))
    aug_points_homogeneous = np.hstack([aug_points * [aug_image.shape[1], aug_image.shape[0]], ones])
    aug_points = M_rot.dot(aug_points_homogeneous.T).T
    aug_points[:, 0] /= aug_image.shape[1]
    aug_points[:, 1] /= aug_image.shape[0]

    # Contrast adjustments
    alpha = np.random.uniform(0.8, 1.2)  # contrast control
    beta = np.random.uniform(-20, 20)  # brightness control
    aug_image = cv2.convertScaleAbs(aug_image, alpha=alpha, beta=beta)

    # Ensure points are valid
    is_points_in_bound = np.all(aug_points >= 0) and np.all(aug_points <= 1)
    if not is_points_in_bound or np.any(np.isnan(aug_points)):
        return None, None
    return aug_image, aug_points


def crop_image(img, points, x1, x2, y1, y2):
    crop_img = img[y1:y2, x1:x2].copy()
    cropped_points = points.copy()
    cropped_points[:, 0] = (points[:, 0] * img.shape[1] - x1) / (x2 - x1)
    cropped_points[:, 1] = (points[:, 1] * img.shape[0] - y1) / (y2 - y1)
    return crop_img, cropped_points


def resize(image: np.ndarray, points: np.ndarray, target_size=input_shape[0]) -> tuple[np.ndarray, np.ndarray]:
    width, height = image.shape[1], image.shape[0]
    square_size = min(width, height)
    aspect_ratio = width / height
    # Unnormalize points
    unnorm_x_min = points[:, 0].min() * width
    unnorm_x_max = points[:, 0].max() * width
    unnorm_y_min = points[:, 1].min() * height
    unnorm_y_max = points[:, 1].max() * height
    # Crop the image, making sure the bounding box stays inside the image
    if unnorm_x_max - unnorm_x_min > square_size:
        crop_x1, crop_x_width = unnorm_x_min, unnorm_x_max - unnorm_x_min
        crop_y1, crop_y_width = 0, square_size
    elif unnorm_y_max - unnorm_y_min > square_size:
        crop_x1, crop_x_width = 0, square_size
        crop_y1, crop_y_width = unnorm_y_min, unnorm_y_max - unnorm_y_min
    else:
        # A square centered around the bounding box
        crop_x1 = (unnorm_x_min + unnorm_x_max) / 2 - square_size / 2 if aspect_ratio > 1 else 0
        crop_y1 = (unnorm_y_min + unnorm_y_max) / 2 - square_size / 2 if aspect_ratio < 1 else 0
        crop_x_width, crop_y_width = square_size, square_size
        if crop_x1 + crop_x_width > width:
            crop_x1 = width - square_size
        elif crop_x1 < 0:
            crop_x1 = 0
        if crop_y1 + crop_y_width > height:
            crop_y1 = height - square_size
        elif crop_y1 < 0:
            crop_y1 = 0

    crop_x1, crop_y1 = int(crop_x1), int(crop_y1)
    crop_x_width, crop_y_width = int(crop_x_width), int(crop_y_width)

    image, points = crop_image(image, points, crop_x1, crop_x1 + crop_x_width, crop_y1, crop_y1 + crop_y_width)
    image = cv2.resize(image, (target_size, target_size))
    return image, points


def load(dir_name: str, augment_prob=0.2) -> tuple[np.ndarray, np.ndarray]:
    image_dir = os.path.join(dir_name, 'images')
    label_dir = os.path.join(dir_name, 'labels')

    images = []
    labels = []

    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            image_id = label_file.split('.')[0]
            image_path = os.path.join(image_dir, f'{image_id}.jpg')
            label_path = os.path.join(label_dir, label_file)

            if os.path.exists(image_path):
                # Read image
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Read label
                with open(label_path, 'r') as file:
                    label_data = file.readline().strip().split()
                    points = np.array([float(coord) for coord in label_data[1:]]).reshape(-1, 2)

                # Resize image and adjust points
                resized_image, resized_points = resize(image, points)
                images.append(resized_image)
                labels.append(resized_points)

                if np.random.rand() < augment_prob:
                    augmented_image, augmented_points = augment(image, points)
                    if augmented_image is None:
                        continue
                    augmented_image, augmented_points = resize(augmented_image, augmented_points)
                    images.append(augmented_image)
                    labels.append(augmented_points)

    images = np.array(images)
    labels = np.array(labels)

    # Normalize images
    images = np.clip(images.astype(np.float32) / 255, 0, 1)

    return images, labels
