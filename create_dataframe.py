# === create_dataframe.py ===
# This script scans the nested directory and creates a CSV file
# containing the filepath and label for each image.

import os
import pandas as pd

# Configuration
DATASET_PATH = "dataset"
CSV_PATH = "image_data.csv"

def create_image_dataframe(folder):
    """
    Scans all subdirectories of the given folder to find image paths and their labels.
    """
    filepaths = []
    labels = []
    print(f"Scanning directory '{folder}' to create a map of all images...")

    for root, dirs, files in os.walk(folder):
        # The label is the name of the parent directory of the image file ('0' or '1')
        if os.path.basename(root) in ['0', '1']:
            label = int(os.path.basename(root))

            for file in files:
                if file.endswith('.png'):
                    # Create the full path for the image and add it to our list
                    filepath = os.path.join(root, file)
                    filepaths.append(filepath)
                    labels.append(label)

    print(f"Scan complete. Found {len(filepaths)} images.")

    # Create a pandas DataFrame
    df = pd.DataFrame({
        'filepath': filepaths,
        'label': labels
    })

    return df

if __name__ == '__main__':
    image_df = create_image_dataframe(DATASET_PATH)

    # Save the DataFrame to a CSV file
    if not image_df.empty:
        image_df.to_csv(CSV_PATH, index=False)
        print(f"Successfully created image map and saved to '{CSV_PATH}'")
    else:
        print("No images found. Could not create CSV file.")