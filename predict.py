import tensorflow as tf
import numpy as np
import string
import tkinter as tk
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import cv2
import compile_model as cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Set a modern Matplotlib style
mplstyle.use('seaborn-v0_8-deep')  

# Classes A to Z
letters = list(string.ascii_uppercase)

# Load Model file or train a new model 
model = cm.load_or_train_model()

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwriting Recognition")
        self.root.geometry("1000x500")  # Reduced height to fit content properly
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)  # Handle window close event

        # Create Frames
        self.left_frame = tk.Frame(root, padx=40, pady=30, bg="#f0f0f0")  # Light gray background
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_frame = tk.Frame(root, padx=40, pady=30, bg="#f0f0f0")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Canvas for Drawing (Increased Size)
        self.canvas_width = 320
        self.canvas_height = 320
        self.canvas = tk.Canvas(self.left_frame, bg='white', width=self.canvas_width, height=self.canvas_height, relief=tk.SOLID, borderwidth=2)
        self.canvas.pack(pady=5)

        # Smooth Drawing Variables
        self.last_x, self.last_y = None, None
        self.has_drawing = False  # Track if user has drawn

        # Draw event (Smooth Line Instead of Dots)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last_position)

        # Bigger "Clear" Button
        self.clear_button = tk.Button(self.left_frame, text="Clear", command=self.clear_canvas, 
                                      font=("Arial", 20, "bold"), bg="#d9534f", fg="white", 
                                      relief=tk.FLAT, padx=20, pady=5)
        self.clear_button.pack(pady=15)

        # PIL Image for canvas
        self.image = Image.new("L", (28, 28), color=0)
        self.draw = ImageDraw.Draw(self.image)

        # Load the trained model
        self.model = tf.keras.models.load_model("model_kisy.h5")

        # Matplotlib Figure for Probabilities
        self.fig, self.ax = plt.subplots(figsize=(7, 3.5))  # Slightly adjusted height
        self.ax.set_xlabel("Letters", fontsize=12, labelpad=10)
        self.ax.set_ylabel("Probability", fontsize=12)
        self.ax.set_title("Probability Distribution", fontsize=14, pad=15)
        self.ax.set_ylim(0, 1)  # No negative probabilities
        self.ax.set_xticks(range(len(letters)))
        self.ax.set_xticklabels(letters, rotation=90, fontsize=10)  # Make x-labels readable
        self.bar_container = self.ax.bar(letters, np.zeros(len(letters)), color='#007acc', edgecolor='black')  # Modern blue color
        self.fig.subplots_adjust(bottom=0.3)  # More space below "Letters"
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas_plot.get_tk_widget().pack(pady=5)

        # Prediction Label
        self.prediction_label = tk.Label(self.right_frame, text="Predicted Letter: None", font=("Arial", 16), fg="#333", bg="#f0f0f0")
        self.prediction_label.pack(pady=10)

        # Start Live Update
        self.update_prediction()

    def paint(self, event):
        """ Draws smooth lines between points instead of separate dots. """
        self.has_drawing = True  # User started drawing

        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, fill='black', width=10, capstyle=tk.ROUND, smooth=True)
            scaled_x1 = self.last_x * (28 / self.canvas_width)
            scaled_y1 = self.last_y * (28 / self.canvas_height)
            scaled_x2 = event.x * (28 / self.canvas_width)
            scaled_y2 = event.y * (28 / self.canvas_height)
            self.draw.line([scaled_x1, scaled_y1, scaled_x2, scaled_y2], fill=255, width=3)

        self.last_x, self.last_y = event.x, event.y

    def reset_last_position(self, event):
        """ Reset the last drawn position when the mouse is released. """
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        """ Clears the canvas and resets the chart and prediction. """
        self.canvas.delete("all")
        self.image = Image.new("L", (28, 28), color=0)
        self.draw = ImageDraw.Draw(self.image)

        # Reset probability chart and ensure it remains empty
        self.has_drawing = False
        for bar in self.bar_container:
            bar.set_height(0)
        self.canvas_plot.draw()

        # Reset prediction label
        self.prediction_label.config(text="Predicted Letter: None")

    def update_prediction(self):
        """ Continuously updates the prediction every 10ms if there is a drawing. """
        if not self.has_drawing:
            # Ensure the chart remains empty when no drawing is present
            for bar in self.bar_container:
                bar.set_height(0)
            self.canvas_plot.draw()
            self.prediction_label.config(text="Predicted Letter: None")
        else:
            # Convert the drawn image to a NumPy array
            img_resized = self.image.resize((28, 28))
            img_array = np.array(img_resized, dtype=np.uint8)

            # Invert colors
            img_array = cv2.bitwise_not(img_array)

            # Apply thresholding
            _, img_array = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY)

            # Normalize
            img_array = img_array / 255.0

            # Reshape for model
            img_array = img_array.reshape(1, 28, 28, 1)

            # Make a prediction
            predictions = self.model.predict(img_array)
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions)

            # Update Label with Prediction
            self.prediction_label.config(text=f"Predicted Letter: {letters[predicted_class]} ({confidence:.2f})")

            # Update Bar Chart
            for bar, prob in zip(self.bar_container, predictions.flatten()):
                bar.set_height(prob)
            self.ax.set_ylim(0, 1)  # Ensure bars are visible within range
            self.canvas_plot.draw()

        # Call this function every 10ms for live updates
        self.root.after(10, self.update_prediction)

    def on_closing(self):
        """ Properly closes the application when the window is closed. """
        plt.close(self.fig)  # Close matplotlib figure
        self.root.destroy()  # Destroy Tkinter window

# Run the Application
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
