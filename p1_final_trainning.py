# === Final Training Script (Complete & Fast Version) ===
# This version uses a smaller, random subset of the full dataset
# for significantly faster training cycles.
# It includes all necessary code blocks from start to finish.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# --- Configuration ---
CSV_PATH = "image_data.csv"
MODEL_PATH = "model/final_model.h5"
IMAGE_SIZE = 50
EPOCHS = 30
BATCH_SIZE = 64

# --- NEW: Set the number of samples for a faster run ---
# Change this number as needed. For the full dataset, set it to None.
NUM_SAMPLES = 30000
# ---------------------------------------------------------

# --- 1. Load the DataFrame ---
try:
    df = pd.read_csv(CSV_PATH)
    # The 'label' column must be a string type for flow_from_dataframe
    df['label'] = df['label'].astype(str)
    print(f"Successfully loaded image data map from '{CSV_PATH}'. Total images found: {len(df)}")
except FileNotFoundError:
    print(f"Error: '{CSV_PATH}' not found. Please run 'create_dataframe.py' first.")
    exit()

# --- Use a smaller subset if NUM_SAMPLES is set ---
if NUM_SAMPLES and len(df) > NUM_SAMPLES:
    print(f"Using a random subset of {NUM_SAMPLES} images for faster training.")
    df = df.sample(n=NUM_SAMPLES, random_state=42)
else:
    print("Dataset is smaller than NUM_SAMPLES, using the full dataset.")


# --- 2. Split the DataFrame into Training and Validation sets ---
train_df, validation_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
print(f"Training with {len(train_df)} samples and validating with {len(validation_df)} samples.")

# --- 3. Setup ImageDataGenerators ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1./255)

# --- 4. Create Generators from DataFrame ---
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepath',
    y_col='label',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=validation_df,
    x_col='filepath',
    y_col='label',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# --- 5. Load Existing Model or Create New One ---
if os.path.exists(MODEL_PATH):
    print(f"Loading existing model to resume training: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
else:
    print("No saved model found. Creating a new model from scratch.")
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

# --- 6. Compile the Model ---
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("\n--- Model Architecture ---")
model.summary()

# --- 7. Callbacks ---
if not os.path.exists('model'): os.makedirs('model')
checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)

# --- 8. Train the Model ---
print("\n--- Starting/Resuming Model Training ---")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# --- 9. Final Evaluation and Reporting ---
print("\n--- Final Model Evaluation on the Best Saved Model ---")
# Load the best performing model saved by ModelCheckpoint
best_model = load_model(MODEL_PATH)
loss, acc = best_model.evaluate(validation_generator, verbose=0)
print(f"✅ Validation Accuracy on best model: {acc * 100:.2f}%")
print(f"✅ Validation Loss on best model: {loss:.4f}")

# --- 10. Plotting and Saving Graphs ---
print("\nPlotting performance graphs...")
plt.figure(figsize=(14, 6))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("final_model_performance.png")
plt.show()

# --- 11. Confusion Matrix and Classification Report ---
print("\nGenerating Classification Report and Confusion Matrix...")
# To get predictions, we reset the generator to ensure order
validation_generator.reset()
y_pred_proba = best_model.predict(validation_generator)
y_pred_classes = (y_pred_proba > 0.5).astype("int32")

# We need the true labels in the correct order
true_labels = validation_generator.classes
# Slicing predictions to match number of labels
y_pred_classes = y_pred_classes[:len(true_labels)]

cm = confusion_matrix(true_labels, y_pred_classes)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Cancer (0)', 'Cancer (1)'],
            yticklabels=['No Cancer (0)', 'Cancer (1)'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig("final_confusion_matrix.png")
plt.show()

print("\n--- Classification Report ---\n")
print(classification_report(true_labels, y_pred_classes, target_names=['No Cancer (0)', 'Cancer (1)']))