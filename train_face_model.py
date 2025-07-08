import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
from glob import glob
import matplotlib.pyplot as plt

# Config
IMG_SIZE = 184  # Match the input size expected by face_detector.py
DATASET_PATH = "dataset"
CATEGORIES = ["face", "no_face"]

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load & preprocess images for CNN (keep 2D structure)
def load_images_cnn():
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

                # Keep 2D structure for CNN (add channel dimension)
                cnn_input = np.expand_dims(normalized, axis=-1)

                data.append(cnn_input)
                labels.append(label_index)

            except Exception as e:
                print(f"[ERROR] Error processing {filename}: {e}")
                continue

    print(f"[INFO] Successfully loaded {len(data)} images")
    return np.array(data), np.array(labels)

# Create data augmentation generator
def create_data_augmentation():
    """Create data augmentation for training"""
    return ImageDataGenerator(
        rotation_range=15,          # Rotate images up to 15 degrees
        width_shift_range=0.1,      # Shift images horizontally
        height_shift_range=0.1,     # Shift images vertically
        brightness_range=[0.8, 1.2], # Adjust brightness
        zoom_range=0.1,             # Zoom in/out slightly
        horizontal_flip=False,       # Don't flip faces horizontally
        fill_mode='nearest'         # Fill missing pixels
    )

print("[INFO] Loading and preprocessing images...")
X, y = load_images_cnn()

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

# Create data augmentation
datagen = create_data_augmentation()
datagen.fit(X_train)

# Build simpler, more effective CNN model
def create_cnn_model():
    """Create a simpler CNN model that works better with limited data"""
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (5, 5), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second convolutional block
        layers.Conv2D(64, (5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # Binary classification: 0=face, 1=no_face
    ])

    return model

model = create_cnn_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

print("[INFO] Enhanced CNN Model architecture:")
model.summary()

# Setup simpler callbacks
callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        'best_face_detector_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("[INFO] Training simplified CNN model...")

# Train without data augmentation first to test basic functionality
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=25,
    validation_data=(X_test, y_test),
    callbacks=callbacks_list,
    verbose=1
)

# Evaluate model
print("\n[INFO] Evaluating model on test set...")
test_results = model.evaluate(X_test, y_test, verbose=0)
test_loss, test_accuracy, test_precision, test_recall = test_results

print(f"[INFO] Test Results:")
print(f"  - Accuracy: {test_accuracy:.4f}")
print(f"  - Precision: {test_precision:.4f}")
print(f"  - Recall: {test_recall:.4f}")
if test_precision + test_recall > 0:
    f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
    print(f"  - F1-Score: {f1_score:.4f}")
else:
    print(f"  - F1-Score: 0.0000 (no positive predictions)")

# Save final model
model.save("face_detector_model.h5")
print("[INFO] Final model saved as face_detector_model.h5")

# Also save the best model if it exists
if os.path.exists('best_face_detector_model.h5'):
    print("[INFO] Best model saved as best_face_detector_model.h5")

# Plot enhanced training history
plt.figure(figsize=(15, 10))

# Accuracy plot
plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(2, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Precision plot
plt.subplot(2, 2, 3)
plt.plot(history.history['precision'], label='Training Precision')
plt.plot(history.history['val_precision'], label='Validation Precision')
plt.title('Model Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)

# Recall plot
plt.subplot(2, 2, 4)
plt.plot(history.history['recall'], label='Training Recall')
plt.plot(history.history['val_recall'], label='Validation Recall')
plt.title('Model Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("[INFO] Enhanced training history saved as training_history.png")

# Print training summary
print(f"\n[INFO] Training completed!")
print(f"[INFO] Total epochs trained: {len(history.history['accuracy'])}")
print(f"[INFO] Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"[INFO] Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
