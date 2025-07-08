import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import cv2
from glob import glob
import matplotlib.pyplot as plt

# Config
IMG_SIZE = 184  # Match the input size expected by face_detector.py
DATASET_PATH = "dataset"
CATEGORIES = ["face", "no_face"]

# Load & preprocess images
def load_images():
    data = []
    labels = []

    print(f"[INFO] Loading images from {DATASET_PATH}")

    for label_index, category in enumerate(CATEGORIES):
        folder = os.path.join(DATASET_PATH, category)
        image_files = glob(os.path.join(folder, "*.png"))
        print(f"[INFO] Found {len(image_files)} images in {category} category")

        for filename in image_files:
            try:
                # Load image
                img = cv2.imread(filename)
                if img is None:
                    print(f"[WARNING] Could not load {filename}")
                    continue

                # Convert to grayscale (to match face_detector.py)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Resize to match expected input size
                resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

                # Normalize pixel values
                normalized = resized / 255.0

                # Flatten to match face_detector.py input format
                flattened = normalized.flatten()

                data.append(flattened)
                labels.append(label_index)

            except Exception as e:
                print(f"[ERROR] Error processing {filename}: {e}")
                continue

    print(f"[INFO] Successfully loaded {len(data)} images")
    return np.array(data), np.array(labels)

print("[INFO] Loading and preprocessing images...")
X, y = load_images()

if len(X) == 0:
    print("[ERROR] No images loaded! Check your dataset path.")
    exit()

print(f"[INFO] Dataset shape: {X.shape}")
print(f"[INFO] Labels shape: {y.shape}")
print(f"[INFO] Face samples: {np.sum(y == 0)}, No-face samples: {np.sum(y == 1)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"[INFO] Training set: {X_train.shape[0]} samples")
print(f"[INFO] Test set: {X_test.shape[0]} samples")

# Build improved model for flattened input
model = models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(IMG_SIZE * IMG_SIZE,)),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification: 0=face, 1=no_face
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("[INFO] Model architecture:")
model.summary()

print("[INFO] Training model...")
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"[INFO] Test accuracy: {test_accuracy:.4f}")

# Save model
model.save("face_detector_model.h5")
print("[INFO] Model saved as face_detector_model.h5")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
print("[INFO] Training history saved as training_history.png")
