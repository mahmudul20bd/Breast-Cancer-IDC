import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ==================================
# 🔹 Variables and Constants
# ==================================
MODEL_PATH = 'model/final_model.h5'
DATASET_PATH = 'dataset'
IMAGE_SIZE = 50

# --- User Configuration ---
# Which class of images do you want to test?
# 0 = No Cancer, 1 = Cancer
CHOOSE_CLASS = 0  # Change this to 0 or 1 and run the script
# -------------------------

# ==================================
# 🔹 New function to find a random image from the dataset
# ==================================
def get_random_image_path(dataset_path, class_label):
    """
    Finds the path of a random image from a nested folder structure.
    """
    class_str = str(class_label)
    all_image_paths = []

    # Finds all folders inside the dataset folder (e.g., 9227, 9228)
    for patient_folder in os.listdir(dataset_path):
        patient_path = os.path.join(dataset_path, patient_folder)
        
        if os.path.isdir(patient_path):
            # Finds the class_label (0 or 1) folder inside each patient's folder
            class_path = os.path.join(patient_path, class_str)
            
            if os.path.exists(class_path):
                # Adds the path of all images in that folder to a list
                for img_file in os.listdir(class_path):
                    if img_file.endswith('.png'):
                        all_image_paths.append(os.path.join(class_path, img_file))

    if not all_image_paths:
        return None  # If no image is found

    # Returns a random path from the list
    return random.choice(all_image_paths)


# ==================================
# 🔹 Main Execution Block
# ==================================
# Get the path of a random image
IMAGE_PATH = get_random_image_path(DATASET_PATH, CHOOSE_CLASS)

if not IMAGE_PATH:
    print(f"Error: No images were found in the subdirectories for class '{os.path.join(DATASET_PATH, str(CHOOSE_CLASS))}'.")
    exit()

print(f"Randomly selected '{IMAGE_PATH}' for testing.")

# Load the model
try:
    model = load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"Error: The model could not be loaded.")
    exit()

# Prepare the image
try:
    img = cv2.imread(IMAGE_PATH)
    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img_normalized = img_resized / 255.0
    img_for_prediction = np.expand_dims(img_normalized, axis=0)
except Exception as e:
    print(f"Error: The image could not be processed: {e}")
    exit()

# Make a prediction
prediction = model.predict(img_for_prediction)
prediction_value = prediction[0][0]

# Determine the result
if prediction_value > 0.5:
    result_text = f"Cancer Detected (Confidence: {prediction_value * 100:.2f}%)"
    result_color = 'red'
else:
    result_text = f"No Cancer Detected (Confidence: {(1 - prediction_value) * 100:.2f}%)"
    result_color = 'green'

print(f"\nResult: {result_text}")
print(f"Actual Class Was: {'Cancer' if CHOOSE_CLASS == 1 else 'No Cancer'}")

# Display the image with the result
original_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(6, 6))
plt.imshow(original_img_rgb)
plt.title(result_text, color='white', backgroundcolor=result_color, pad=10)
plt.axis('off')
plt.show()