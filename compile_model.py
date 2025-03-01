import get_data as gt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization, LeakyReLU, SeparableConv2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import os
import string
import tkinter as tk
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tkinter import messagebox
import cv2

# Letters A-Z in a list
letters = list(string.ascii_uppercase)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=False, fill_mode="nearest"
)

MODEL_FILE = "model_kisy.h5"

def load_or_train_model():


    if os.path.exists(MODEL_FILE):
        print("Loading preexisting model...")
        model = tf.keras.models.load_model(MODEL_FILE)
    else:
        print("No preexisting model found. Training a new model...")

            # Load Data
        data = gt.load_data()
        features = np.array(data[0])
        labels = np.array(data[1], dtype=int)

            # Train-Test Split
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

            # Normalize Pixels (0-1 Range)
        x_train, x_test = x_train / 255.0, x_test / 255.0

            # Reshape for CNN
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

            # Convert Labels to One-Hot Encoding
        y_train = to_categorical(y_train, num_classes=26)
        y_test = to_categorical(y_test, num_classes=26)


        # Apply Data Augmentation
        datagen.fit(x_train)

        model = Sequential()

        # CNN Model
        model.add(SeparableConv2D(64, (3, 3), activation=None, padding="same", input_shape=(28, 28, 1)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(SeparableConv2D(128, (3, 3), activation=None, padding="same",
                                  depthwise_regularizer=l2(0.001), pointwise_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(SeparableConv2D(256, (3, 3), activation=None, padding="same",
                                  depthwise_regularizer=l2(0.001), pointwise_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))  # Regularization

        model.add(Flatten())
        model.add(Dense(128, activation="relu", kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(26, activation="softmax"))

        model.compile(optimizer=Adam(learning_rate=0.0005), loss="categorical_crossentropy", metrics=["accuracy"])

        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=0.00001)
        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=32),
            epochs=50,
            callbacks=[reduce_lr, early_stop],
            validation_data=(x_test, y_test)
        )

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
        print(f"\nTest Accuracy: {test_acc} and loss {test_loss}")

        model.save(MODEL_FILE)
        print("Model saved successfully!")

    return model


def test_model(model):
    # Load Data
    data = gt.load_data()
    features = np.array(data[0])
    labels = np.array(data[1], dtype=int)

    # Train-Test Split
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Normalize Pixels (0-1 Range)
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape for CNN
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Convert Labels to One-Hot Encoding
    y_train = to_categorical(y_train, num_classes=26)
    y_test = to_categorical(y_test, num_classes=26)

    idx = np.random.randint(0, len(x_test))  # Pick a random test sample
    test_img = x_test[idx].reshape(28, 28)  # Remove extra dimension
    test_label = np.argmax(y_test[idx])  # Get correct label
   
    # Predict the Test Image
    img_input = x_test[idx].reshape(1, 28, 28, 1)
    predictions = model.predict(img_input)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    # Display the Image & Prediction
    plt.figure(figsize=(4, 4))
    plt.imshow(test_img, cmap="gray")
    plt.title(f"True: {letters[test_label]}, Predicted: {letters[predicted_class]} ({confidence:.2f})")
    plt.axis("off")
    plt.show()
    
    print(f"Predicted Letter: {letters[predicted_class]}, Confidence: {confidence:.2f}")
    print(f"Actual Label: {letters[test_label]}")
