import numpy as np
import cv2
import os

# Define max number of features per folder
max_features = 8000

def load_data():
    features = []
    labels = []
    
    # List all valid single-character folders inside "Letters"
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of get_data.py
    print(base_dir)
    folder_list = sorted([f for f in os.listdir(base_dir) if len(f) == 1 and os.path.isdir(os.path.join(base_dir, f))])
    
    if not folder_list:
        print("Error: No valid letter folders found in 'Letters' directory.")
        return None, None

    for index, folder in enumerate(folder_list):
        folder_path = os.path.join(base_dir, folder)
        label = index  # Store index as label

        # Get file list
        file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        if not file_list:
            print(f"Warning: No files found in '{folder_path}', skipping folder.")
            continue  

        # Lists to store images and labels for current folder
        image_list = []
        label_list = []

        for filename in file_list[:max_features]:  # Limit number of files processed
            image_path = os.path.join(folder_path, filename)

            # Read image in grayscale mode
            image = cv2.imread(image_path, 0)  # 0 = grayscale

            if image is None:
                print(f"Warning: Skipping '{filename}', failed to load.")
                continue  

            # Resize the image to 28x28
            resized_img = cv2.resize(image, (28, 28))

            # Append to lists
            image_list.append(resized_img)
            label_list.append([label])  

        # Skip folders where no valid images were loaded
        if not image_list:
            print(f"Warning: No valid images loaded for folder '{folder}', skipping.")
            continue

        # Convert lists to NumPy arrays
        image_dataset = np.array(image_list, dtype=np.float32)  # Shape: (N, 28, 28)
        label_dataset = np.array(label_list, dtype=str)  # Shape: (N, 1)

        # Append datasets
        features.append(image_dataset)
        labels.append(label_dataset)

    if not features:
        print("Error: No valid data loaded.")
        return None, None

    # Concatenate all data into single NumPy arrays
    features = np.concatenate(features, axis=0)  # Shape: (Total_N, 28, 28)
    labels = np.concatenate(labels, axis=0)  # Shape: (Total_N, 1)
    print(f"Loaded {features.shape}")
    return features, labels

