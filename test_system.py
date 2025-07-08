#!/usr/bin/env python3
"""
Test script for the face overlay system
Validates model performance and system functionality
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

def test_model_performance():
    """Test the trained model on the dataset"""
    print("[INFO] Testing model performance...")
    
    # Check if model exists
    if not os.path.exists('face_detector_model.h5'):
        print("[ERROR] Model not found. Please train the model first.")
        return False
    
    # Load model
    try:
        model = tf.keras.models.load_model('face_detector_model.h5')
        print("[INFO] Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False
    
    # Load test data
    print("[INFO] Loading test data...")
    
    data = []
    labels = []
    categories = ["face", "no_face"]
    
    for label_index, category in enumerate(categories):
        folder = os.path.join("dataset", category)
        if not os.path.exists(folder):
            print(f"[ERROR] Dataset folder not found: {folder}")
            return False
        
        image_files = glob(os.path.join(folder, "*.png"))
        print(f"[INFO] Found {len(image_files)} images in {category} category")
        
        for filename in image_files:
            try:
                # Load and preprocess image (same as training)
                img = cv2.imread(filename)
                if img is None:
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (184, 184))
                normalized = resized / 255.0
                flattened = normalized.flatten()
                
                data.append(flattened)
                labels.append(label_index)
                
            except Exception as e:
                print(f"[WARNING] Error processing {filename}: {e}")
                continue
    
    if len(data) == 0:
        print("[ERROR] No test data loaded")
        return False
    
    X_test = np.array(data)
    y_test = np.array(labels)
    
    print(f"[INFO] Test data shape: {X_test.shape}")
    print(f"[INFO] Test labels shape: {y_test.shape}")
    
    # Make predictions
    print("[INFO] Making predictions...")
    predictions = model.predict(X_test, verbose=0)
    y_pred = (predictions > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[INFO] Model accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\n[INFO] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=categories))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("[INFO] Confusion matrix saved as confusion_matrix.png")
    
    # Prediction distribution
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(predictions[y_test == 0], bins=20, alpha=0.7, label='Face', color='blue')
    plt.hist(predictions[y_test == 1], bins=20, alpha=0.7, label='No Face', color='red')
    plt.xlabel('Prediction Score')
    plt.ylabel('Count')
    plt.title('Prediction Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(predictions)), predictions, 
                c=['blue' if label == 0 else 'red' for label in y_test], alpha=0.6)
    plt.axhline(y=0.5, color='black', linestyle='--', label='Threshold')
    plt.xlabel('Sample Index')
    plt.ylabel('Prediction Score')
    plt.title('Predictions vs True Labels')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('prediction_analysis.png')
    print("[INFO] Prediction analysis saved as prediction_analysis.png")
    
    return True

def test_webcam():
    """Test webcam face detection"""
    print("[INFO] Testing webcam face detection...")
    print("Press 'q' to quit, 's' to save a test image")
    
    import mediapipe as mp
    
    # Initialize MediaPipe
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Failed to access webcam")
        return False
    
    save_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from webcam")
            break
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        
        # Draw detections
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)
            
            # Show confidence
            confidence = results.detections[0].score[0]
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Webcam Face Detection Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"webcam_test_{save_count}.png"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Saved test image: {filename}")
            save_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    return True

def test_screen_capture():
    """Test screen capture and face detection"""
    print("[INFO] Testing screen capture...")
    print("Press ESC to quit")
    
    import mss
    import mediapipe as mp
    
    # Initialize MediaPipe
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    
    # Setup screen capture
    sct = mss.mss()
    monitor = sct.monitors[1]  # Primary monitor
    
    while True:
        # Capture screen
        screenshot = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        
        # Resize for faster processing
        height, width = frame.shape[:2]
        if width > 1920:
            scale = 1920 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        
        # Draw detections
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)
            print(f"[INFO] Detected {len(results.detections)} face(s)")
        
        # Resize for display
        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("Screen Capture Face Detection Test", display_frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break
    
    cv2.destroyAllWindows()
    return True

def test_config():
    """Test configuration loading"""
    print("[INFO] Testing configuration...")
    
    config_file = 'face_overlay_config.json'
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print("[INFO] Configuration loaded successfully:")
            print(json.dumps(config, indent=2))
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load config: {e}")
            return False
    else:
        print("[WARNING] No configuration file found")
        return False

def main():
    """Run all tests"""
    print("=== Face Overlay System Tests ===\n")
    
    tests = [
        ("Configuration", test_config),
        ("Model Performance", test_model_performance),
        ("Webcam Face Detection", test_webcam),
        ("Screen Capture", test_screen_capture)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} Test ---")
        try:
            result = test_func()
            results[test_name] = result
            status = "PASSED" if result else "FAILED"
            print(f"[{status}] {test_name} test completed")
        except KeyboardInterrupt:
            print(f"\n[INTERRUPTED] {test_name} test interrupted by user")
            results[test_name] = False
        except Exception as e:
            print(f"[ERROR] {test_name} test failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n=== Test Summary ===")
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status} {test_name}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    return 0 if passed == total else 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
