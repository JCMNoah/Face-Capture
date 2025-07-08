import numpy as np
import cv2
import tensorflow as tf

model = tf.keras.models.load_model('face_detector_model.h5')

# Dummy test input
dummy_img = np.zeros((184, 184), dtype=np.uint8)
dummy_input = dummy_img.flatten().astype(np.float32) / 255.0
dummy_input = np.expand_dims(dummy_input, axis=0)

pred = model.predict(dummy_input)
print("Prediction:", pred)
