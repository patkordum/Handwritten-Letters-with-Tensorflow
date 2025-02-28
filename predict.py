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

# Classes A to Z
letters = list(string.ascii_uppercase)

# 📌 Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=False, fill_mode="nearest"
)

# --- Load or Train Model ---
MODEL_FILE = "model_kisy.h5"

def load_or_train_model():

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

    if os.path.exists(MODEL_FILE):
        print("Loading preexisting model...")
        model = tf.keras.models.load_model(MODEL_FILE)
    else:
        print("No preexisting model found. Training a new model...")



        # Apply Data Augmentation
        datagen.fit(x_train)

        model = Sequential()

        # 📌 CNN Model
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

    return model, x_test, y_test

# --- Load Model & Get Test Data ---
model, x_test, y_test = load_or_train_model()

# --- Step 1: Test with One Image Before Tkinter Opens ---
def test_model():
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

test_model()  # Run Before Opening Tkinter

# --- Step 2: Create the GUI for Drawing Digits ---
class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Recognizer")

        # Canvas settings
        self.canvas_width = 280  # Scaled-up canvas size
        self.canvas_height = 280
        self.canvas = tk.Canvas(root, bg='white', width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        # Draw event
        self.canvas.bind("<B1-Motion>", self.paint)

        # Buttons
        self.predict_button = tk.Button(root, text="Predict", command=self.predict)
        self.predict_button.pack()

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        # PIL Image for canvas
        self.image = Image.new("L", (28, 28), color=0)  # 28x28 pixel grayscale image
        self.draw = ImageDraw.Draw(self.image)

        # Load the trained model
        self.model = tf.keras.models.load_model(MODEL_FILE)

    def paint(self, event):
        x, y = event.x, event.y
        brush_size = 10
        scaled_x = x * (28 / self.canvas_width)
        scaled_y = y * (28 / self.canvas_height)
        self.canvas.create_oval(x, y, x + brush_size, y + brush_size, fill='black')
        self.draw.ellipse([scaled_x, scaled_y, scaled_x + 1, scaled_y + 1], fill=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (28, 28), color=0)
        self.draw = ImageDraw.Draw(self.image)


    def predict(self):
        # Convert the drawn image to a NumPy array
        img_resized = self.image.resize((28, 28))  # Resize to 28x28
        img_array = np.array(img_resized, dtype=np.uint8)  # Convert to NumPy array

        # Invert colors (if needed) - Training data might be black-on-white
        img_array = cv2.bitwise_not(img_array)

        # Apply thresholding to make it match the dataset (binary black/white)
        _, img_array = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY)

        # Normalize to [0, 1] like the training images
        img_array = img_array / 255.0

        # Reshape for the model
        img_array = img_array.reshape(1, 28, 28, 1)

        # Make a prediction
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

        # Show result
        print(f"Predicted Letter: {letters[predicted_class]}, Confidence: {confidence:.2f}")
        messagebox.showinfo("Prediction Result", f"Predicted Letter: {letters[predicted_class]} \nConfidence: {confidence:.2f}")

        # Create Probability Plot
        plt.figure(figsize=(5, 3))
        plt.bar(letters, predictions.flatten(), color='blue')
        plt.xlabel("Letters")
        plt.ylabel("Probability")
        plt.title("Probability Distribution")
        plt.show()


# --- Step 3: Launch the Application ---
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
