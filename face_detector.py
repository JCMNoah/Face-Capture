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

    # Detection smoothing
    detection_history = []
    smoothing_frames = config.get("detection_smoothing", 3)
    use_hybrid = config.get("use_hybrid_detection", True)

    print(f"[INFO] Detection thread started - Hybrid: {use_hybrid}, Smoothing: {smoothing_frames}")

    while True:
        try:
            # Get current game screen
            with game_lock:
                if game_screen is not None:
                    current_game = game_screen.copy()
                else:
                    time.sleep(0.1)
                    continue

            # Run ML detection (expensive operation)
            gray = cv2.cvtColor(current_game, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE))
            normalized = resized / 255.0
            flattened = normalized.flatten()
            input_for_model = np.expand_dims(flattened, axis=0)

            pred = model.predict(input_for_model, verbose=0)[0][0]
            ml_detected = pred < FACE_THRESHOLD

            # Improved hybrid detection: Use MediaPipe as primary, ML as validation
            final_detected = False
            if use_hybrid:
                # Always check MediaPipe first
                rgb_game = cv2.cvtColor(current_game, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_game)

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
                    global debug_game_img, debug_ml_pred, debug_ml_detected, debug_final_detected, debug_detection_history
                    debug_game_img = current_game.copy()
                    debug_ml_pred = pred
                    debug_ml_detected = ml_detected
                    debug_final_detected = is_character_detected
                    debug_detection_history = detection_history.copy()
            
            # Run MediaPipe face position detection
            current_face_position = None
            if is_character_detected:
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

def main():
    """Main loop optimized for 60fps"""
    global webcam_frame, game_screen, character_detected, face_position
    global webcam_face_ready, webcam_mask_ready, detection_confidence
    
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
                
                # Debug windows for webcam and output
                if config.get("show_debug_windows", False) and webcam_frame is not None:
                    # Create Face Detection Debug window (moved from detection thread)
                    if debug_game_img is not None:
                        debug_img = debug_game_img.copy()

                        # Add ML prediction info
                        ml_color = (0, 255, 0) if debug_ml_detected else (0, 0, 255)
                        cv2.putText(debug_img, f"ML Pred: {debug_ml_pred:.3f} ({'FACE' if debug_ml_detected else 'NO FACE'})",
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ml_color, 2)

                        # Add MediaPipe detection info
                        rgb_game = cv2.cvtColor(debug_game_img, cv2.COLOR_BGR2RGB)
                        mp_results = face_detection.process(rgb_game)

                        if mp_results.detections:
                            for detection in mp_results.detections:
                                bbox = detection.location_data.relative_bounding_box
                                h, w = debug_game_img.shape[:2]
                                x = int(bbox.xmin * w)
                                y = int(bbox.ymin * h)
                                width = int(bbox.width * w)
                                height = int(bbox.height * h)

                                # Draw MediaPipe detection box
                                cv2.rectangle(debug_img, (x, y), (x + width, y + height), (255, 0, 0), 2)
                                confidence = detection.score[0]
                                cv2.putText(debug_img, f"MP: {confidence:.2f}",
                                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                            mp_text = f"MediaPipe: DETECTED ({len(mp_results.detections)} faces)"
                            mp_color = (255, 0, 0)
                        else:
                            mp_text = "MediaPipe: NO FACE"
                            mp_color = (0, 0, 255)

                        cv2.putText(debug_img, mp_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mp_color, 2)

                        # Add final decision
                        final_color = (0, 255, 0) if debug_final_detected else (0, 0, 255)
                        final_text = f"FINAL: {'FACE DETECTED' if debug_final_detected else 'NO FACE'}"
                        cv2.putText(debug_img, final_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, final_color, 2)

                        # Add smoothing info
                        history_text = f"History: {debug_detection_history} -> {debug_final_detected}"
                        cv2.putText(debug_img, history_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        # Show debug window
                        debug_resized = cv2.resize(debug_img, (640, 480))
                        cv2.imshow("Face Detection Debug", debug_resized)
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