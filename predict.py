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
import compile_model as cm

# Classes A to Z
letters = list(string.ascii_uppercase)


# Load Model file or train a new model 
model = cm.load_or_train_model()

# File Name for Model 
MODEL_FILE = "model_kisy.h5"

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
