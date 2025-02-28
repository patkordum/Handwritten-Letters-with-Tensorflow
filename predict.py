import get_data as gt
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
from sklearn.model_selection import train_test_split
import os
import string
import tkinter as tk
from tkinter import messagebox  # Import messagebox
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Classes A to Z
letters = list(string.ascii_uppercase)

# --- Step 1: Load or Train the MNIST Digit Classifier ---
MODEL_FILE = "model_kisy.h5"

def load_or_train_model():
    if os.path.exists(MODEL_FILE):
        print("Loading preexisting model...")
        model = tf.keras.models.load_model(MODEL_FILE)
    else:
        print("No preexisting model found. Training a new model...")

        # Load data
        data = gt.load_data()
        features = np.array(data[0])  # Ensure NumPy array format
        labels = np.array(data[1],dtype=int)
        print(labels[0])

        # Split dataset (80% training, 20% testing)
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        # Normalize Pixel values
        x_train, x_test=x_train/255, x_test/255

        # Verify the split
        print(f"Training Data: {x_train.shape}, Labels: {y_train.shape}")
        print(f"Testing Data: {x_test.shape}, Labels: {y_test.shape}")

        model = Sequential([
            Input(shape=(28, 28)),  
            Flatten(),              
            Dense(256, activation='relu'),  # 1st Hidden Layer
            Dense(128, activation='relu'),   # 2nd Hidden Layer
            Dense(64, activation='relu'),   # 2nd Hidden Layer
            Dense(26, activation='softmax') # Output Layer: 26 classes (A-Z)
        ])

        # Compile the model
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        # Show the model summary
        model.summary()

        model.fit(x_train, y_train, epochs=50,validation_data=(x_test, y_test))
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
        print(f"\nTest Accuracy: {test_acc}")

        model.save(MODEL_FILE)
        print("Model saved successfully!")

    return model

# Load or train the model
model = load_or_train_model()

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
        # Scale coordinates to the 28x28 image size
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
        img_resized = self.image.resize((28, 28))
        img_resized = np.array(self.image) / 255.0  # Normalize to 0-1 range
        img_resized = img_resized.reshape(1, 28, 28)
        img_resized = np.clip(img_resized * 2, 0, 1)

        # Make a prediction
        predictions = self.model.predict(img_resized)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)
        print(predictions)
        
        # Create Probability Plot
        plt.clf()
        plt.bar(letters, predictions.flatten())
        plt.xlabel("Letters")
        plt.ylabel("Probability")
        plt.title("Probability Distribution")
        plt.show()
        

        # Display the result
        print(f"Predicted Digit: {predicted_class} = {letters[int(predicted_class)]}, Confidence: {confidence:.2f}")
        messagebox.showinfo("Prediction Result", f"Predicted Digit: {predicted_class} = {letters[int(predicted_class)]} \nConfidence: {confidence:.2f}")

# --- Step 3: Launch the Application ---
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()

