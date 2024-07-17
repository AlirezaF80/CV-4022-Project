import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model

from loader import load

input_shape = (256, 256, 3)


def create_model() -> Model:
    inputs = keras.layers.Input(shape=input_shape)

    x = keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
    x = keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((4, 4))(x)

    x = keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    outputs = keras.layers.Dense(8)(x)
    model = Model(inputs, outputs)
    return model


if __name__ == '__main__':
    DATASET_PATH = './dataset/'
    BEST_MODEL_PATH = './best_model.keras'
    MODEL_PATH = './model.keras'

    model = create_model()
    model.summary()

    X, y = load(DATASET_PATH)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    callbacks = [
        keras.callbacks.ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    ]
    history = model.fit(X, y, epochs=200, batch_size=32, validation_split=0.2)
    model.save(MODEL_PATH)