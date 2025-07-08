#!/usr/bin/env python3
"""
Face Overlay for OBS Streaming - Optimized 60fps Version
High-performance face detection and overlay with threading and caching
"""

import cv2
import numpy as np
import mediapipe as mp
import pyvirtualcam
import mss
import tensorflow as tf
import time
import json
import os
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import gc

# --- CONFIG ---
CONFIG_FILE = "face_overlay_config.json"

def load_config():
    """Load configuration from file or create default"""
    default_config = {
        "game_region": {"top": 200, "left": 700, "width": 1000, "height": 1000},
        "input_size": 184,
        "webcam_face_size": 200,
        "fps": 60,
        "face_detection_confidence": 0.5,
        "face_threshold": 0.5,
        "overlay_alpha": 1.0,
        "mirror_webcam": True,
        "show_debug": True,
        "disable_game_background": True,
        "use_transparency": True,
        "webcam_fps": 60,
        "webcam_resolution": [1920, 1080],
        "high_quality_mode": True,
        "optimize_performance": True
    }
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            print(f"[INFO] Loaded config from {CONFIG_FILE}")
            return config
        except Exception as e:
            print(f"[WARNING] Error loading config: {e}. Using defaults.")
    
    with open(CONFIG_FILE, 'w') as f:
        json.dump(default_config, f, indent=2)
    print(f"[INFO] Created default config file: {CONFIG_FILE}")
    return default_config

# Load configuration
config = load_config()
GAME_REGION = config["game_region"]
INPUT_SIZE = config["input_size"]
WEBCAM_FACE_SIZE = config["webcam_face_size"]
FPS = config["fps"]
FACE_THRESHOLD = config["face_threshold"]
FACE_DETECTION_CONFIDENCE = config["face_detection_confidence"]

print(f"[INFO] Target FPS: {FPS}")
print(f"[INFO] Game region: {GAME_REGION}")

# Load trained model with optimization
try:
    model = tf.keras.models.load_model('face_detector_model.h5')
    # Optimize model for inference
    model.compile(optimizer='adam', loss='binary_crossentropy')
    print("[INFO] Model loaded and optimized")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit()

# Initialize MediaPipe globally for use in debug windows
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=FACE_DETECTION_CONFIDENCE
)

# Global variables for thread communication
webcam_frame = None
game_screen = None
character_detected = False
face_position = None
webcam_face_ready = None
webcam_mask_ready = None
detection_confidence = 0.0

# Debug variables for main thread display
debug_game_img = None
debug_ml_pred = 0.5
debug_ml_detected = False
debug_final_detected = False
debug_detection_history = []

# Enhanced debugging variables
debug_processed_img = None
debug_confidence_map = None
debug_mediapipe_detections = None
debug_cnn_detections = []
debug_processing_times = {
    'ml_inference': 0.0,
    'mediapipe': 0.0,
    'total_detection': 0.0,
    'frame_processing': 0.0
}
debug_detection_stats = {
    'total_frames': 0,
    'ml_detections': 0,
    'mediapipe_detections': 0,
    'final_detections': 0,
    'false_positives': 0,
    'false_negatives': 0
}

# Thread-safe locks
webcam_lock = threading.Lock()
game_lock = threading.Lock()
detection_lock = threading.Lock()

# Performance optimization: Pre-allocate arrays
def create_preallocated_arrays():
    """Pre-allocate arrays to avoid memory allocation during runtime"""
    global temp_game_gray, temp_game_resized, temp_webcam_rgb, temp_face_crop
    
    temp_game_gray = np.zeros((GAME_REGION['height'], GAME_REGION['width']), dtype=np.uint8)
    temp_game_resized = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
    temp_webcam_rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
    temp_face_crop = np.zeros((WEBCAM_FACE_SIZE, WEBCAM_FACE_SIZE, 3), dtype=np.uint8)

def webcam_capture_thread():
    """Dedicated thread for webcam capture"""
    global webcam_frame
    
    # Initialize webcam in thread
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Failed to access webcam in thread")
        return
    
    # Configure webcam for optimal performance
    if config.get("high_quality_mode", True):
        webcam_resolution = config.get("webcam_resolution", [1920, 1080])
        webcam_fps = config.get("webcam_fps", 60)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_resolution[1])
        cap.set(cv2.CAP_PROP_FPS, webcam_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    print("[INFO] Webcam capture thread started")
    
    while True:
        ret, frame = cap.read()
        if ret:
            with webcam_lock:
                webcam_frame = frame.copy()
        time.sleep(1/60)  # 60fps webcam capture
    
    cap.release()

def game_screen_capture_thread():
    """Dedicated thread for game screen capture"""
    global game_screen
    
    sct = mss.mss()
    print("[INFO] Game screen capture thread started")
    
    while True:
        try:
            # Capture screen
            screen = np.array(sct.grab(GAME_REGION))
            screen_bgr = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
            
            with game_lock:
                game_screen = screen_bgr.copy()
                
        except Exception as e:
            print(f"[WARNING] Screen capture error: {e}")
        
        time.sleep(1/60)  # 60fps screen capture

def detection_thread():
    """Dedicated thread for AI detection with hybrid validation and smoothing"""
    global character_detected, face_position, detection_confidence, face_detection
    global debug_game_img, debug_ml_pred, debug_ml_detected, debug_final_detected, debug_detection_history
    global debug_mediapipe_detections, debug_processing_times, debug_processed_img, debug_cnn_detections

    # Detection smoothing
    detection_history = []
    smoothing_frames = config.get("detection_smoothing", 3)
    use_hybrid = config.get("use_hybrid_detection", True)

    print(f"[INFO] Detection thread started - Hybrid: {use_hybrid}, Smoothing: {smoothing_frames}")

    while True:
        try:
            # Start timing for total detection
            detection_start_time = time.time()

            # Get current game screen
            with game_lock:
                if game_screen is not None:
                    current_game = game_screen.copy()
                else:
                    time.sleep(0.1)
                    continue

            # Run CNN ML detection (back to simple, fast approach)
            ml_start_time = time.time()

            gray = cv2.cvtColor(current_game, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE))
            normalized = resized / 255.0

            # Prepare input for CNN model (keep 2D structure + channel dimension)
            cnn_input = np.expand_dims(normalized, axis=-1)  # Add channel dimension
            input_for_model = np.expand_dims(cnn_input, axis=0)  # Add batch dimension

            pred = model.predict(input_for_model, verbose=0)[0][0]
            ml_detected = pred < FACE_THRESHOLD

            # Smart face localization without slow sliding windows
            cnn_detections = []

            if ml_detected:
                # Use a fast 4-quadrant approach to find face location
                h, w = gray.shape

                # Test 4 main regions: top-left, top-right, bottom-left, bottom-right
                # Plus center region - total of 5 quick tests
                regions = [
                    (0, 0, w//2, h//2),                    # Top-left
                    (w//2, 0, w//2, h//2),                # Top-right
                    (0, h//2, w//2, h//2),                # Bottom-left
                    (w//2, h//2, w//2, h//2),             # Bottom-right
                    (w//4, h//4, w//2, h//2)              # Center
                ]

                best_confidence = pred
                best_region = None

                for i, (rx, ry, rw, rh) in enumerate(regions):
                    # Extract region
                    region = gray[ry:ry+rh, rx:rx+rw]

                    # Resize and normalize
                    resized_region = cv2.resize(region, (INPUT_SIZE, INPUT_SIZE))
                    normalized = resized_region / 255.0

                    # Prepare input for CNN model
                    cnn_input = np.expand_dims(normalized, axis=-1)
                    input_for_model = np.expand_dims(cnn_input, axis=0)

                    # Get prediction for this region
                    region_pred = model.predict(input_for_model, verbose=0)[0][0]

                    # If this region has better confidence, use it
                    if region_pred < best_confidence:
                        best_confidence = region_pred
                        best_region = (rx, ry, rw, rh)

                # Use the best region found, or fall back to center if none are better
                if best_region:
                    x, y, w, h = best_region
                else:
                    # Fallback to center region
                    w, h = gray.shape[1] // 2, gray.shape[0] // 2
                    x, y = (gray.shape[1] - w) // 2, (gray.shape[0] - h) // 2

                confidence = 1.0 - best_confidence
                cnn_detections.append({
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'confidence': confidence, 'raw_pred': best_confidence
                })

                # Update prediction with best found
                pred = best_confidence

            ml_inference_time = time.time() - ml_start_time

            # Store debug information
            with detection_lock:
                debug_processed_img = gray.copy()  # Store full image instead of resized
                debug_ml_pred = pred
                debug_ml_detected = ml_detected
                debug_cnn_detections = cnn_detections.copy()
                debug_processing_times['ml_inference'] = ml_inference_time

            # Detection logic: CNN-only or Hybrid mode
            mediapipe_start_time = time.time()
            final_detected = False
            mediapipe_detections = None

            # Check if CNN-only mode is enabled
            cnn_only_mode = config.get("cnn_only_mode", False)

            if cnn_only_mode:
                # CNN-only mode: Use only the ML model
                final_detected = ml_detected
                if config.get("show_debug", True):
                    if ml_detected:
                        print(f"[DETECTION] CNN-only mode: FACE DETECTED ({pred:.3f})")
                    else:
                        print(f"[DETECTION] CNN-only mode: NO FACE ({pred:.3f})")
            elif use_hybrid:
                # Always check MediaPipe first
                rgb_game = cv2.cvtColor(current_game, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_game)
                mediapipe_detections = results.detections if results.detections else []

                if results.detections:
                    # MediaPipe found face(s), check confidence
                    best_confidence = max([det.score[0] for det in results.detections])
                    mp_confident = best_confidence >= config.get("mediapipe_confidence", 0.3)

                    if mp_confident:
                        # MediaPipe is confident, use ML as secondary validation
                        if ml_detected:
                            # Both agree - highest confidence
                            final_detected = True
                            if config.get("show_debug", True):
                                print(f"[DETECTION] Both ML ({pred:.3f}) and MediaPipe ({best_confidence:.2f}) agree - FACE DETECTED")
                        else:
                            # MediaPipe confident but ML disagrees - trust MediaPipe for character faces
                            final_detected = True
                            if config.get("show_debug", True):
                                print(f"[DETECTION] MediaPipe confident ({best_confidence:.2f}), ML uncertain ({pred:.3f}) - TRUSTING MEDIAPIPE")
                    else:
                        # MediaPipe found face but low confidence
                        if ml_detected:
                            # ML agrees with low-confidence MediaPipe
                            final_detected = True
                            if config.get("show_debug", True):
                                print(f"[DETECTION] ML supports ({pred:.3f}) low-confidence MediaPipe ({best_confidence:.2f}) - FACE DETECTED")
                        else:
                            # Both uncertain
                            final_detected = False
                            if config.get("show_debug", True):
                                print(f"[DETECTION] Both uncertain - ML ({pred:.3f}), MediaPipe ({best_confidence:.2f}) - NO FACE")
                else:
                    # MediaPipe found no faces
                    if ml_detected:
                        # ML detected but MediaPipe didn't - likely false positive
                        final_detected = False
                        if config.get("show_debug", True):
                            print(f"[DETECTION] ML detected ({pred:.3f}) but MediaPipe found nothing - FALSE POSITIVE")
                    else:
                        # Both agree no face
                        final_detected = False
                        if config.get("show_debug", True):
                            print(f"[DETECTION] Both agree no face - ML ({pred:.3f}), MediaPipe (no detections)")
            else:
                # Use only ML detection (with debug info)
                final_detected = ml_detected
                if config.get("show_debug", True):
                    if ml_detected:
                        print(f"[DETECTION] ML-only mode: FACE DETECTED ({pred:.3f})")
                    else:
                        print(f"[DETECTION] ML-only mode: NO FACE ({pred:.3f})")

            # Calculate timing
            mediapipe_time = time.time() - mediapipe_start_time
            total_detection_time = time.time() - detection_start_time

            # Apply detection smoothing
            detection_history.append(final_detected)
            if len(detection_history) > smoothing_frames:
                detection_history.pop(0)

            # Smooth decision: majority vote
            smooth_detected = sum(detection_history) > len(detection_history) // 2
            is_character_detected = smooth_detected

            # Store debug info for main thread (don't create windows in detection thread)
            if config.get("show_debug_windows", False):
                # Store debug data for main thread to display
                with detection_lock:
                    # Store the debug information for main thread
                    debug_game_img = current_game.copy()
                    debug_ml_pred = pred
                    debug_ml_detected = ml_detected
                    debug_final_detected = is_character_detected
                    debug_detection_history = detection_history.copy()
                    debug_mediapipe_detections = mediapipe_detections
                    debug_processing_times['mediapipe'] = mediapipe_time
                    debug_processing_times['total_detection'] = total_detection_time
            
            # Face position detection (CNN-only or MediaPipe)
            current_face_position = None
            if is_character_detected:
                if cnn_only_mode and cnn_detections:
                    # Use CNN detection for position
                    detection = cnn_detections[0]  # Use the first/best detection
                    x, y, w, h = detection['x'], detection['y'], detection['w'], detection['h']

                    center_x = x + w // 2
                    center_y = y + h // 2
                    confidence = detection['confidence']

                    current_face_position = (center_x, center_y, w, h, confidence)
                else:
                    # Use MediaPipe for position (hybrid mode)
                    rgb_game = cv2.cvtColor(current_game, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(rgb_game)

                    if results.detections:
                        detection = results.detections[0]
                        bbox = detection.location_data.relative_bounding_box

                        h, w = current_game.shape[:2]
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)

                        center_x = x + width // 2
                        center_y = y + height // 2
                        confidence = detection.score[0]

                        current_face_position = (center_x, center_y, width, height, confidence)
            
            # Update global variables
            with detection_lock:
                character_detected = is_character_detected
                face_position = current_face_position
                detection_confidence = pred
                
        except Exception as e:
            print(f"[WARNING] Detection error: {e}")
        
        # Run detection every 0.2 seconds (5fps) for better performance
        time.sleep(0.2)

def webcam_face_processing_thread():
    """Dedicated thread for webcam face processing"""
    global webcam_face_ready, webcam_mask_ready, face_detection
    
    print("[INFO] Webcam face processing thread started")
    
    while True:
        try:
            # Get current webcam frame
            with webcam_lock:
                if webcam_frame is not None:
                    current_webcam = webcam_frame.copy()
                else:
                    time.sleep(0.1)
                    continue
            
            # Process webcam face
            face_crop, mask = create_face_overlay_optimized(
                current_webcam,
                WEBCAM_FACE_SIZE,
                face_detection,
                mirror=config["mirror_webcam"]
            )
            
            # Update processed face
            with webcam_lock:
                webcam_face_ready = face_crop
                webcam_mask_ready = mask
                
        except Exception as e:
            print(f"[WARNING] Webcam processing error: {e}")
        
        # Process webcam at 30fps (enough for smooth overlay)
        time.sleep(1/30)

def create_face_overlay_optimized(webcam_frame, face_size, face_detection, mirror=True):
    """Optimized face extraction with minimal memory allocation"""
    h, w = webcam_frame.shape[:2]
    
    # Use smaller resolution for face detection
    if w > 640:
        scale_factor = 640 / w
        detection_width = int(w * scale_factor)
        detection_height = int(h * scale_factor)
        detection_frame = cv2.resize(webcam_frame, (detection_width, detection_height))
    else:
        detection_frame = webcam_frame
        scale_factor = 1.0
    
    rgb_webcam = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_webcam)
    
    if not results.detections:
        return None, None
    
    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box
    
    # Scale coordinates back to original resolution
    det_h, det_w = detection_frame.shape[:2]
    x = int(bbox.xmin * det_w / scale_factor)
    y = int(bbox.ymin * det_h / scale_factor)
    width = int(bbox.width * det_w / scale_factor)
    height = int(bbox.height * det_h / scale_factor)
    
    # Ensure cropping within bounds
    x, y = max(0, x), max(0, y)
    x2, y2 = min(x + width, w), min(y + height, h)
    
    if x2 <= x or y2 <= y:
        return None, None
    
    # Crop face
    face_crop = webcam_frame[y:y2, x:x2]
    
    # Resize with high quality
    face_crop = cv2.resize(face_crop, (face_size, face_size), interpolation=cv2.INTER_LINEAR)
    
    # Mirror if requested
    if mirror:
        face_crop = cv2.flip(face_crop, 1)
    
    # Create circular mask (reuse pre-allocated array if possible)
    mask = np.zeros((face_size, face_size), dtype=np.uint8)
    cv2.circle(mask, (face_size // 2, face_size // 2), face_size // 2, 255, -1)
    mask_3ch = cv2.merge([mask, mask, mask])
    
    return face_crop, mask_3ch

def create_debug_windows():
    """Create comprehensive debugging visualization"""
    global debug_processed_img, debug_ml_pred, debug_ml_detected, debug_final_detected
    global debug_mediapipe_detections, debug_processing_times, debug_detection_stats
    global debug_game_img, game_screen

    try:
        # Create main debug window with multiple panels
        debug_height = 800
        debug_width = 1200
        debug_canvas = np.zeros((debug_height, debug_width, 3), dtype=np.uint8)

        # Panel 1: Original game screen (top-left)
        panel_w, panel_h = 300, 200
        if game_screen is not None:
            game_resized = cv2.resize(game_screen, (panel_w, panel_h))
            debug_canvas[10:10+panel_h, 10:10+panel_w] = game_resized
            cv2.putText(debug_canvas, "Game Screen", (10, 10+panel_h+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Panel 2: CNN detections with bounding boxes (top-center)
        if debug_processed_img is not None:
            # Convert grayscale to BGR for display
            ml_img_bgr = cv2.cvtColor(debug_processed_img, cv2.COLOR_GRAY2BGR)

            # Draw CNN detection boxes
            if debug_cnn_detections:
                for detection in debug_cnn_detections:
                    x, y, w, h = detection['x'], detection['y'], detection['w'], detection['h']
                    confidence = detection['confidence']

                    # Draw bounding box (green for high confidence, yellow for medium, red for low)
                    if confidence > 0.7:
                        color = (0, 255, 0)  # Green
                    elif confidence > 0.5:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (0, 0, 255)  # Red

                    cv2.rectangle(ml_img_bgr, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(ml_img_bgr, f"{confidence:.2f}", (x, y-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            ml_resized = cv2.resize(ml_img_bgr, (panel_w, panel_h))
            debug_canvas[10:10+panel_h, 320:320+panel_w] = ml_resized

            cnn_count = len(debug_cnn_detections) if debug_cnn_detections else 0
            cv2.putText(debug_canvas, f"CNN Detections ({cnn_count})",
                       (320, 10+panel_h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (0, 255, 0) if debug_ml_detected else (0, 0, 255), 2)

        # Panel 3: MediaPipe detections (top-right)
        if game_screen is not None:
            mp_img = game_screen.copy()
            mp_resized = cv2.resize(mp_img, (panel_w, panel_h))

            # Draw MediaPipe detections
            if debug_mediapipe_detections:
                for detection in debug_mediapipe_detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * panel_w)
                    y = int(bbox.ymin * panel_h)
                    w = int(bbox.width * panel_w)
                    h = int(bbox.height * panel_h)
                    confidence = detection.score[0]

                    cv2.rectangle(mp_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(mp_resized, f"{confidence:.2f}", (x, y-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            debug_canvas[10:10+panel_h, 630:630+panel_w] = mp_resized
            mp_count = len(debug_mediapipe_detections) if debug_mediapipe_detections else 0
            cv2.putText(debug_canvas, f"MediaPipe ({mp_count} faces)",
                       (630, 10+panel_h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (0, 255, 0) if mp_count > 0 else (0, 0, 255), 2)

        cv2.imshow("Face Detection Debug", debug_canvas)

    except Exception as e:
        print(f"[DEBUG] Error creating debug window: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main loop optimized for 60fps"""
    global webcam_frame, game_screen, character_detected, face_position
    global webcam_face_ready, webcam_mask_ready, detection_confidence
    global debug_detection_stats, debug_processing_times
    
    print("[INFO] Starting optimized face overlay system...")
    print("[INFO] Press 'd' key to toggle debug windows")
    print("[INFO] Press Ctrl+C to stop")
    
    # Pre-allocate arrays
    create_preallocated_arrays()
    
    # Start background threads
    webcam_thread = threading.Thread(target=webcam_capture_thread, daemon=True)
    game_thread = threading.Thread(target=game_screen_capture_thread, daemon=True)
    detection_thread_obj = threading.Thread(target=detection_thread, daemon=True)
    face_processing_thread = threading.Thread(target=webcam_face_processing_thread, daemon=True)
    
    webcam_thread.start()
    game_thread.start()
    detection_thread_obj.start()
    face_processing_thread.start()
    
    # Wait for threads to initialize
    time.sleep(2)

    # Wait for webcam to be ready
    print("[INFO] Waiting for webcam to be ready...")
    while webcam_frame is None:
        time.sleep(0.1)
    print("[INFO] Webcam ready!")

    print("[INFO] All threads started, beginning main loop...")
    
    # Performance tracking
    frame_count = 0
    start_time = time.time()
    
    # Main rendering loop
    try:
        with pyvirtualcam.Camera(width=GAME_REGION['width'], height=GAME_REGION['height'], fps=FPS) as cam:
            print(f"[INFO] Virtual camera started: {cam.device}")
            
            while True:
                loop_start = time.time()
                
                # Get current game screen (non-blocking)
                with game_lock:
                    if game_screen is not None:
                        current_game = game_screen.copy()
                    else:
                        continue
                
                # Get current detection status (non-blocking)
                with detection_lock:
                    current_character_detected = character_detected
                    current_face_position = face_position
                    current_confidence = detection_confidence
                
                # Create output frame
                use_transparency = config.get("use_transparency", False)
                disable_background = config.get("disable_game_background", False)
                
                if use_transparency:
                    overlay = np.zeros((current_game.shape[0], current_game.shape[1], 4), dtype=np.uint8)
                elif disable_background:
                    overlay = np.zeros_like(current_game)
                else:
                    overlay = current_game.copy()
                
                # Apply face overlay if character detected
                if current_character_detected:
                    # Get processed webcam face (non-blocking)
                    with webcam_lock:
                        face_crop = webcam_face_ready
                        mask = webcam_mask_ready
                    
                    if face_crop is not None and mask is not None:
                        # Determine overlay position and size
                        if current_face_position:
                            center_x, center_y, char_width, char_height, confidence = current_face_position
                            # Scale overlay based on character face size
                            scale_factor = max(char_width, char_height) / WEBCAM_FACE_SIZE
                            overlay_size = int(WEBCAM_FACE_SIZE * max(0.8, min(2.0, scale_factor)))
                            
                            if overlay_size != WEBCAM_FACE_SIZE:
                                face_crop = cv2.resize(face_crop, (overlay_size, overlay_size))
                                mask = cv2.resize(mask, (overlay_size, overlay_size))
                        else:
                            # Use center as fallback
                            center_x = GAME_REGION['width'] // 2
                            center_y = GAME_REGION['height'] // 2
                            overlay_size = WEBCAM_FACE_SIZE
                        
                        # Calculate position
                        top_left_x = max(0, min(center_x - overlay_size // 2, GAME_REGION['width'] - overlay_size))
                        top_left_y = max(0, min(center_y - overlay_size // 2, GAME_REGION['height'] - overlay_size))
                        
                        # Apply overlay (optimized)
                        try:
                            if use_transparency and overlay.shape[2] == 4:
                                # RGBA overlay
                                face_rgba = cv2.cvtColor(face_crop, cv2.COLOR_BGR2BGRA)
                                mask_alpha = mask[:, :, 0]
                                face_rgba[:, :, 3] = mask_alpha
                                overlay[top_left_y:top_left_y + overlay_size, 
                                       top_left_x:top_left_x + overlay_size] = face_rgba
                            else:
                                # RGB overlay (fastest)
                                roi = overlay[top_left_y:top_left_y + overlay_size, 
                                            top_left_x:top_left_x + overlay_size]
                                if roi.shape[:2] == face_crop.shape[:2]:
                                    if disable_background:
                                        fg = cv2.bitwise_and(face_crop, mask)
                                        overlay[top_left_y:top_left_y + overlay_size, 
                                               top_left_x:top_left_x + overlay_size] = fg
                                    else:
                                        fg = cv2.bitwise_and(face_crop, mask)
                                        bg = cv2.bitwise_and(roi, cv2.bitwise_not(mask))
                                        overlay[top_left_y:top_left_y + overlay_size, 
                                               top_left_x:top_left_x + overlay_size] = cv2.add(bg, fg)
                        except Exception as e:
                            if config["show_debug"]:
                                print(f"[WARNING] Overlay error: {e}")
                
                # Convert to RGB for virtual camera
                if use_transparency and overlay.shape[2] == 4:
                    alpha_channel = overlay[:, :, 3]
                    has_content = np.any(alpha_channel > 0)
                    
                    if has_content:
                        rgb_channels = overlay[:, :, :3]
                        alpha_normalized = alpha_channel.astype(np.float32) / 255.0
                        green_bg = np.full_like(rgb_channels, [0, 255, 0], dtype=np.uint8)
                        alpha_3ch = np.stack([alpha_normalized] * 3, axis=2)
                        blended = (alpha_3ch * rgb_channels.astype(np.float32) + 
                                  (1 - alpha_3ch) * green_bg.astype(np.float32))
                        overlay_rgb = cv2.cvtColor(blended.astype(np.uint8), cv2.COLOR_BGR2RGB)
                    else:
                        overlay_rgb = np.full((overlay.shape[0], overlay.shape[1], 3), [0, 255, 0], dtype=np.uint8)
                else:
                    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                
                # Enhanced debug windows
                if config.get("show_debug_windows", False) and webcam_frame is not None:
                    try:
                        # Update detection statistics
                        with detection_lock:
                            debug_detection_stats['total_frames'] += 1
                            if debug_ml_detected:
                                debug_detection_stats['ml_detections'] += 1
                            if debug_mediapipe_detections and len(debug_mediapipe_detections) > 0:
                                debug_detection_stats['mediapipe_detections'] += 1
                            if debug_final_detected:
                                debug_detection_stats['final_detections'] += 1

                        # Create comprehensive debug visualization
                        create_debug_windows()
                    except Exception as e:
                        print(f"[DEBUG] Error in debug windows: {e}")
                    # Show webcam with face detection
                    webcam_debug = webcam_frame.copy()

                    # Detect face in webcam for debug
                    h, w = webcam_frame.shape[:2]
                    if w > 1280:
                        scale_factor = 1280 / w
                        detection_width = int(w * scale_factor)
                        detection_height = int(h * scale_factor)
                        detection_frame = cv2.resize(webcam_frame, (detection_width, detection_height))
                    else:
                        detection_frame = webcam_frame
                        scale_factor = 1.0

                    rgb_webcam = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
                    webcam_results = face_detection.process(rgb_webcam)

                    if webcam_results.detections:
                        for detection in webcam_results.detections:
                            bbox = detection.location_data.relative_bounding_box
                            det_h, det_w = detection_frame.shape[:2]
                            x = int(bbox.xmin * det_w / scale_factor)
                            y = int(bbox.ymin * det_h / scale_factor)
                            width = int(bbox.width * det_w / scale_factor)
                            height = int(bbox.height * det_h / scale_factor)

                            # Draw webcam face detection box
                            cv2.rectangle(webcam_debug, (x, y), (x + width, y + height), (0, 255, 0), 2)
                            confidence = detection.score[0]
                            cv2.putText(webcam_debug, f"Face: {confidence:.2f}",
                                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        webcam_text = f"Webcam: {len(webcam_results.detections)} face(s) detected"
                        webcam_color = (0, 255, 0)
                    else:
                        webcam_text = "Webcam: No face detected"
                        webcam_color = (0, 0, 255)

                    cv2.putText(webcam_debug, webcam_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, webcam_color, 2)

                    # Show webcam debug window
                    webcam_resized = cv2.resize(webcam_debug, (640, 480))
                    cv2.imshow("Webcam Debug", webcam_resized)

                    # Show final output
                    output_debug = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                    cv2.putText(output_debug, f"Character: {'DETECTED' if character_detected else 'NOT DETECTED'}",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if character_detected else (0, 0, 255), 2)
                    output_resized = cv2.resize(output_debug, (640, 480))
                    cv2.imshow("Final Output", output_resized)

                # Create a simple control window for keyboard input
                control_img = np.zeros((120, 450, 3), dtype=np.uint8)
                cv2.putText(control_img, "Face Detection Control", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(control_img, "Press 'd' debug, 'c' CNN-only, 'q' quit", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                debug_status = "ON" if config.get("show_debug_windows", False) else "OFF"
                cv2.putText(control_img, f"Debug: {debug_status}", (10, 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0) if config.get("show_debug_windows", False) else (0, 0, 255), 1)

                cnn_only = "ON" if config.get("cnn_only_mode", False) else "OFF"
                cv2.putText(control_img, f"CNN-Only: {cnn_only}", (10, 95),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0) if config.get("cnn_only_mode", False) else (0, 0, 255), 1)

                cv2.imshow("Control Panel", control_img)

                # Check for keyboard input (non-blocking)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('d'):
                    config["show_debug_windows"] = not config.get("show_debug_windows", False)
                    status = "ENABLED" if config["show_debug_windows"] else "DISABLED"
                    print(f"[INFO] Debug windows {status}")
                    if not config["show_debug_windows"]:
                        cv2.destroyWindow("Face Detection Debug")
                        cv2.destroyWindow("Webcam Debug")
                        cv2.destroyWindow("Final Output")
                elif key == ord('c'):
                    # Toggle CNN-only mode
                    config["cnn_only_mode"] = not config.get("cnn_only_mode", False)
                    status = "ENABLED" if config["cnn_only_mode"] else "DISABLED"
                    print(f"[INFO] CNN-only mode {status}")
                    if config["cnn_only_mode"]:
                        print("[INFO] MediaPipe disabled - using only CNN for detection")
                    else:
                        print("[INFO] Hybrid mode enabled - using both CNN and MediaPipe")
                elif key == ord('s'):
                    # Save current frame for analysis
                    if debug_game_img is not None:
                        timestamp = int(time.time())
                        filename = f"debug_frame_{timestamp}.png"
                        cv2.imwrite(filename, debug_game_img)
                        print(f"[DEBUG] Saved frame to {filename}")
                        print(f"[DEBUG] ML Prediction: {debug_ml_pred:.3f}, Detected: {debug_ml_detected}")
                        print(f"[DEBUG] Final Detection: {debug_final_detected}")
                elif key == ord('r'):
                    # Reset statistics
                    with detection_lock:
                        debug_detection_stats.update({
                            'total_frames': 0,
                            'ml_detections': 0,
                            'mediapipe_detections': 0,
                            'final_detections': 0,
                            'false_positives': 0,
                            'false_negatives': 0
                        })
                    print("[DEBUG] Statistics reset")
                elif key == ord('q'):
                    print("[INFO] Quit requested")
                    break

                # Send to virtual camera
                cam.send(overlay_rgb)

                # Performance tracking
                frame_count += 1
                if frame_count % 120 == 0:  # Every 2 seconds at 60fps
                    perf_current_time = time.time()
                    actual_fps = 120 / (perf_current_time - start_time)
                    if config["show_debug"]:
                        print(f"[PERF] {actual_fps:.1f} FPS | Face: {current_character_detected} | Confidence: {current_confidence:.3f}")
                    start_time = perf_current_time
                
                # Maintain target framerate
                elapsed = time.time() - loop_start
                target_interval = 1.0 / FPS
                sleep_time = max(0, target_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Garbage collection every 300 frames to prevent memory leaks
                if frame_count % 300 == 0:
                    gc.collect()
                    
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    except Exception as e:
        print(f"[ERROR] System error: {e}")
    finally:
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete")

if __name__ == "__main__":
    main()