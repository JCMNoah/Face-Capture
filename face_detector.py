#!/usr/bin/env python3
"""
Face Overlay for OBS Streaming - Clean 60fps Version
Detects character faces and overlays webcam face in real-time
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
    
    # Save default config
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

# Load trained model
try:
    model = tf.keras.models.load_model('face_detector_model.h5')
    print("[INFO] Model loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    print("[INFO] Please train the model first using: python main.py train")
    exit()

# Initialize webcam
print("[INFO] Initializing webcam...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[ERROR] Failed to access webcam")
    exit()

# Configure webcam for high quality
if config.get("high_quality_mode", True):
    webcam_resolution = config.get("webcam_resolution", [1920, 1080])
    webcam_fps = config.get("webcam_fps", 60)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_resolution[1])
    cap.set(cv2.CAP_PROP_FPS, webcam_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for low latency
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    print(f"[INFO] Webcam configured: {webcam_resolution[0]}x{webcam_resolution[1]} @ {webcam_fps} FPS")

# Test webcam
ret, frame = cap.read()
if not ret:
    print("[ERROR] Failed to read from webcam")
    exit()

frame_height, frame_width = frame.shape[:2]
print(f"[INFO] Webcam active: {frame_width}x{frame_height}")

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1, 
    min_detection_confidence=FACE_DETECTION_CONFIDENCE
)

# Setup screen capture
sct = mss.mss()

def select_game_region_interactive():
    """Interactive region selection with live preview"""
    print("\n=== Interactive Game Region Selection ===")
    print("1. A screenshot will appear")
    print("2. Click and drag to select the region where character faces appear")
    print("3. Press ENTER or SPACE to confirm")
    print("4. Press ESC to cancel and use current region")
    print("5. The selected region will be highlighted in real-time")

    try:
        # Capture full screen
        full_monitor = sct.monitors[1]
        screenshot = np.array(sct.grab(full_monitor))
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        # Resize if too large for display
        height, width = screenshot.shape[:2]
        if width > 1920 or height > 1080:
            scale = min(1920/width, 1080/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            display_img = cv2.resize(screenshot, (new_width, new_height))
            scale_factor = scale
        else:
            display_img = screenshot.copy()
            scale_factor = 1.0

        print("Select the game region and press ENTER or SPACE to confirm")
        roi = cv2.selectROI("Select Game Region - Press ENTER to confirm", display_img, False)
        cv2.destroyAllWindows()

        if roi[2] == 0 or roi[3] == 0:
            print("No region selected. Keeping current region.")
            return None

        # Adjust coordinates back to original scale
        roi_scaled = [int(x/scale_factor) for x in roi]

        new_region = {
            "top": full_monitor["top"] + roi_scaled[1],
            "left": full_monitor["left"] + roi_scaled[0],
            "width": roi_scaled[2],
            "height": roi_scaled[3]
        }

        print(f"New region selected: {new_region}")
        return new_region

    except Exception as e:
        print(f"Error during region selection: {e}")
        return None

def detect_character_face(game_img):
    """Detect if character face is present using ML model"""
    gray = cv2.cvtColor(game_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE))
    normalized = resized / 255.0
    flattened = normalized.flatten()
    input_for_model = np.expand_dims(flattened, axis=0)
    
    pred = model.predict(input_for_model, verbose=0)[0][0]
    return pred < FACE_THRESHOLD, pred

def detect_character_face_position(game_img):
    """Detect exact position of character face using MediaPipe"""
    rgb_game = cv2.cvtColor(game_img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_game)
    
    if results.detections:
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        h, w = game_img.shape[:2]
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        center_x = x + width // 2
        center_y = y + height // 2
        confidence = detection.score[0]
        
        return (center_x, center_y, width, height, confidence)
    
    return None

def create_face_overlay(webcam_frame, face_size, mirror=True):
    """Extract and prepare face from webcam for overlay"""
    h, w = webcam_frame.shape[:2]
    
    # For high resolution, downsample for face detection
    if w > 1280:
        scale_factor = 1280 / w
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
    
    # Crop face from original high-resolution frame
    face_crop = webcam_frame[y:y2, x:x2]
    
    # Resize with high quality
    face_crop = cv2.resize(face_crop, (face_size, face_size), interpolation=cv2.INTER_LANCZOS4)
    
    # Mirror if requested
    if mirror:
        face_crop = cv2.flip(face_crop, 1)
    
    # Create circular mask
    mask = np.zeros((face_size, face_size), dtype=np.uint8)
    cv2.circle(mask, (face_size // 2, face_size // 2), face_size // 2, 255, -1)
    mask_3ch = cv2.merge([mask, mask, mask])
    
    return face_crop, mask_3ch

# Performance tracking
frame_count = 0
start_time = time.time()
detection_count = 0

# Detection optimization - cache results
last_detection_time = 0
detection_interval = 0.1  # Run detection every 0.1 seconds (10 FPS)
cached_character_detected = False
cached_face_position = None
cached_pred = 0.5

# Frame processing optimization
process_every_n_frames = 2  # Process every 2nd frame for 30fps effective rate
frame_skip_counter = 0

print("[INFO] Starting face overlay system...")
print(f"[INFO] Virtual camera will run at {FPS} FPS")
print(f"[INFO] Detection will run every {detection_interval} seconds for performance")
print("[INFO] Press 'r' key to reselect game region")
print("[INFO] Press Ctrl+C to stop")

# Ask if user wants to select region now
response = input("\nWould you like to select the game region now? (y/n): ").lower().strip()
if response == 'y':
    new_region = select_game_region_interactive()
    if new_region:
        GAME_REGION = new_region
        config["game_region"] = new_region
        # Save updated config
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[INFO] Game region updated and saved: {GAME_REGION}")
    else:
        print(f"[INFO] Using current region: {GAME_REGION}")

# Main loop
try:
    with pyvirtualcam.Camera(width=GAME_REGION['width'], height=GAME_REGION['height'], fps=FPS) as cam:
        print(f"[INFO] Virtual camera started: {cam.device}")
        
        while True:
            loop_start = time.time()
            current_time = time.time()

            # Frame skipping for performance
            frame_skip_counter += 1
            should_process = frame_skip_counter >= process_every_n_frames
            if should_process:
                frame_skip_counter = 0

            # Capture game screen
            game_img = np.array(sct.grab(GAME_REGION))
            game_img = cv2.cvtColor(game_img, cv2.COLOR_BGRA2BGR)

            # Capture webcam frame
            ret, webcam_frame = cap.read()
            if not ret:
                print("[WARNING] Failed to grab webcam frame")
                continue

            # Run detection only periodically for performance
            if current_time - last_detection_time >= detection_interval:
                character_detected, pred = detect_character_face(game_img)
                face_position = detect_character_face_position(game_img)

                # Cache results
                cached_character_detected = character_detected
                cached_face_position = face_position
                cached_pred = pred
                last_detection_time = current_time
                detection_count += 1
            else:
                # Use cached results
                character_detected = cached_character_detected
                face_position = cached_face_position
                pred = cached_pred
            
            # Create output frame
            use_transparency = config.get("use_transparency", False)
            disable_background = config.get("disable_game_background", False)
            
            if use_transparency:
                overlay = np.zeros((game_img.shape[0], game_img.shape[1], 4), dtype=np.uint8)
            elif disable_background:
                overlay = np.zeros_like(game_img)
            else:
                overlay = game_img.copy()
            
            if character_detected:
                # Create webcam face overlay
                face_crop, mask = create_face_overlay(
                    webcam_frame, 
                    WEBCAM_FACE_SIZE, 
                    mirror=config["mirror_webcam"]
                )
                
                if face_crop is not None and mask is not None:
                    # Determine overlay position and size
                    if face_position:
                        center_x, center_y, char_width, char_height, confidence = face_position
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
                    
                    # Apply overlay
                    try:
                        if use_transparency and overlay.shape[2] == 4:
                            # RGBA overlay
                            face_rgba = cv2.cvtColor(face_crop, cv2.COLOR_BGR2BGRA)
                            mask_alpha = mask[:, :, 0]
                            face_rgba[:, :, 3] = mask_alpha
                            overlay[top_left_y:top_left_y + overlay_size, 
                                   top_left_x:top_left_x + overlay_size] = face_rgba
                        else:
                            # RGB overlay
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
            
            # Send to virtual camera
            cam.send(overlay_rgb)

            # Check for keyboard input (non-blocking)
            if cv2.waitKey(1) & 0xFF == ord('r'):
                print("\n[INFO] Reselecting game region...")
                new_region = select_game_region_interactive()
                if new_region:
                    GAME_REGION = new_region
                    config["game_region"] = new_region
                    # Save updated config
                    with open(CONFIG_FILE, 'w') as f:
                        json.dump(config, f, indent=2)
                    print(f"[INFO] Game region updated: {GAME_REGION}")
                else:
                    print("[INFO] Region selection cancelled")

            # Performance tracking
            frame_count += 1
            if frame_count % 60 == 0:
                perf_current_time = time.time()
                actual_fps = 60 / (perf_current_time - start_time)
                detection_fps = detection_count / (perf_current_time - start_time)
                if config["show_debug"]:
                    print(f"[PERF] Output: {actual_fps:.1f} FPS, Detection: {detection_fps:.1f} FPS, Face: {character_detected}, Pred: {pred:.3f}")
                start_time = perf_current_time
                detection_count = 0
            
            # Maintain target framerate
            elapsed = time.time() - loop_start
            target_interval = 1.0 / FPS
            sleep_time = max(0, target_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\n[INFO] Stopping...")
except Exception as e:
    print(f"[ERROR] System error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Cleanup complete")
