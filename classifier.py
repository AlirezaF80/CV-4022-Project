import re

import cv2
import datasets
import numpy as np
import tensorflow.keras as keras
from PIL import Image
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Reshape
from tensorflow.keras.models import Model

classifier_input_shape = (64, 288, 3)


def create_classifier_model(input_shape=classifier_input_shape):
    # Input layer
    inputs = Input(shape=input_shape)

    # Convolutional layers
    x = Conv2D(32, (7, 7), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (7, 7), activation='relu', padding='same')(x)
    x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Reshape to get 8 feature maps for 8 characters
    feature_maps = Reshape((8, -1))(x)

    # Create a list to hold the outputs for each character
    outputs = []

    for i in range(8):
        # Extract feature vector for the ith character
        feature_vector = feature_maps[:, i, :]

        # Fully connected layers for the ith character
        fc = Dense(512, activation='relu')(feature_vector)
        fc = Dense(256, activation='relu')(fc)
        if i == 2:  # Third character is a persian letter
            output = Dense(32, activation='softmax')(fc)
        else:  # All other characters are digits
            output = Dense(10, activation='softmax')(fc)

        outputs.append(output)

    # Combine all outputs
    final_output = keras.layers.Concatenate()(outputs)

    # Define the model
    model = Model(inputs, final_output)

    return model


# Create the classifier model
classifier_model = create_classifier_model()


def preprocess_image(image, target_size):
    """
    Load and preprocess an image
    :param image: PIL image
    :param target_size: Target size for the image
    :return: Preprocessed image array
    """
    # Ensure the image is of type PIL.Image
    if isinstance(image, Image.Image):
        # Resize the image to the target size
        image = np.array(image)
        image = cv2.resize(image, target_size)
        # Normalize the image array to [0, 1]
        image = image / 255.0
    else:
        raise ValueError("Input image is not of type PIL.Image")

    return image


def prepare_dataset(dataset, target_size=(classifier_input_shape[1], classifier_input_shape[0])):
    """
    Prepare the dataset by converting labels and preprocessing images
    :param dataset: Dataset to be prepared
    :param target_size: Target size for the images
    :return: Tuple of preprocessed images and labels
    """
    images = []
    labels = []

    for example in dataset:
        image_path = example['image_path']
        label_str = example['label']

        image = preprocess_image(image_path, target_size)
        label = convert_label_to_output(label_str)

        images.append(image)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


valid_license_plate_label_pattern = re.compile(r'^\d{2}[آ-ی]\d{5}$')


def is_valid_label(example):
    """
    Check if the label follows the desired pattern.
    :param example: A dictionary containing the image_path and label.
    :return: True if the label is valid, False otherwise.
    """
    return bool(valid_license_plate_label_pattern.match(example['label']))


def convert_label_to_output(label_str):
    # Persian letters to index mapping (example, you need to complete this mapping)
    persian_letter_to_index = {
        'ا': 0, 'ب': 1, 'پ': 2, 'ت': 3, 'ث': 4, 'ج': 5, 'چ': 6, 'ح': 7,
        'خ': 8, 'د': 9, 'ذ': 10, 'ر': 11, 'ز': 12, 'ژ': 13, 'س': 14, 'ش': 15,
        'ص': 16, 'ض': 17, 'ط': 18, 'ظ': 19, 'ع': 20, 'غ': 21, 'ف': 22, 'ق': 23,
        'ک': 24, 'گ': 25, 'ل': 26, 'م': 27, 'ن': 28, 'و': 29, 'ه': 30, 'ی': 31
    }

    # Initialize the output array
    output = []

    # Convert each character to one-hot encoding
    for i, char in enumerate(label_str):
        if i == 2:  # Persian letter
            index = persian_letter_to_index[char]
            one_hot = to_categorical(index, num_classes=32)
        else:  # Digits
            index = int(char)
            one_hot = to_categorical(index, num_classes=10)

        output.append(one_hot)

    # Concatenate all one-hot encodings
    output = np.concatenate(output)

    return output


index_to_persian_letter = {
    0: 'ا', 1: 'ب', 2: 'پ', 3: 'ت', 4: 'ث', 5: 'ج', 6: 'چ', 7: 'ح',
    8: 'خ', 9: 'د', 10: 'ذ', 11: 'ر', 12: 'ز', 13: 'ژ', 14: 'س', 15: 'ش',
    16: 'ص', 17: 'ض', 18: 'ط', 19: 'ظ', 20: 'ع', 21: 'غ', 22: 'ف', 23: 'ق',
    24: 'ک', 25: 'گ', 26: 'ل', 27: 'م', 28: 'ن', 29: 'و', 30: 'ه', 31: 'ی'
}


def convert_output_to_label(output):
    label = ""

    # Process each character
    k = 0
    for i in range(8):
        if i != 2:
            index = np.argmax(output[k:k + 10])
            label += str(index)
            k += 10
        if i == 2:  # Persian letter
            index = np.argmax(output[k:k + 32])
            label += index_to_persian_letter[index]
            k += 32

    return label


if __name__ == '__main__':
    # Load the datasets
    dataset_train = datasets.load_dataset('hezarai/persian-license-plate-v1', split='train')
    dataset_test = datasets.load_dataset('hezarai/persian-license-plate-v1', split='test')
    dataset_valid = datasets.load_dataset('hezarai/persian-license-plate-v1', split='validation')

    # Filter the datasets
    dataset_train = dataset_train.filter(is_valid_label)
    dataset_test = dataset_test.filter(is_valid_label)
    dataset_valid = dataset_valid.filter(is_valid_label)

    # Prepare the datasets
    X_train, y_train = prepare_dataset(dataset_train)
    X_test, y_test = prepare_dataset(dataset_test)
    X_val, y_val = prepare_dataset(dataset_valid)

    classifier_model.summary()
    classifier_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = classifier_model.fit(X_train, y_train,
                                   validation_data=(X_val, y_val),
                                   epochs=5,
                                   batch_size=64)

    # Evaluate the model
    test_loss, test_accuracy = classifier_model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
