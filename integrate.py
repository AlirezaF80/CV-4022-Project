import cv2
import numpy as np

from classifier import classifier_input_shape, classifier_model, convert_output_to_label
from extract import extract
from regressor import input_shape, model


def read_plate(image: np.ndarray) -> str:
    pass


def read_single(image: np.ndarray):
    image = cv2.resize(image, (input_shape[0], input_shape[0]))
    image = np.clip(image / 255.0, 0, 1)
    pred = model.predict(np.array([image]))[0].reshape(-1, 2)
    image = extract(image, pred)
    image = cv2.resize(image, (classifier_input_shape[1], classifier_input_shape[0]))
    pred = classifier_model.predict(np.array([image]))[0]
    return convert_output_to_label(pred)


if __name__ == '__main__':
    pass
