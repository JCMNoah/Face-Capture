#!/usr/bin/env python3
"""
Face Overlay for OBS Streaming
Main application script that provides a user-friendly interface for the face overlay system.

Usage:
    python main.py [command]
    
Commands:
    collect     - Collect training data (face/no-face screenshots)
    train       - Train the face detection model
    run         - Run the face overlay system
    test        - Test face detection on screen
    config      - Configure settings
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'cv2', 'numpy', 'mediapipe', 'pyvirtualcam', 
        'mss', 'tensorflow', 'sklearn', 'matplotlib'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"[ERROR] Missing required packages: {', '.join(missing)}")
        print("Please install them using:")
        print("pip install opencv-python numpy mediapipe pyvirtualcam mss tensorflow scikit-learn matplotlib")
        return False
    
    return True

def check_files():
    """Check if required files exist"""
    required_files = [
        'facecollector.py',
        'train_face_model.py', 
        'face_detector.py',
        'template_match_test.py'
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)
    
    if missing:
        print(f"[ERROR] Missing required files: {', '.join(missing)}")
        return False
    
    return True

def collect_data():
    """Run the data collection script"""
    print("[INFO] Starting data collection...")
    print("This will help you collect face/no-face training data.")
    print("Instructions:")
    print("- Select the region where character faces appear")
    print("- Press 'f' when a character face is visible")
    print("- Press 'n' when no character face is visible")
    print("- Press 'r' to reselect region")
    print("- Press 'q' to quit")
    print()
    
    try:
        subprocess.run([sys.executable, 'facecollector.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Data collection failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n[INFO] Data collection interrupted by user")
    
    return True

def train_model():
    """Train the face detection model"""
    print("[INFO] Starting model training...")
    
    # Check if dataset exists
    if not os.path.exists('dataset/face') or not os.path.exists('dataset/no_face'):
        print("[ERROR] Dataset not found. Please collect data first using 'python main.py collect'")
        return False
    
    # Count samples
    face_count = len([f for f in os.listdir('dataset/face') if f.endswith('.png')])
    no_face_count = len([f for f in os.listdir('dataset/no_face') if f.endswith('.png')])
    
    print(f"[INFO] Found {face_count} face samples and {no_face_count} no-face samples")
    
    if face_count < 10 or no_face_count < 10:
        print("[WARNING] You should have at least 10 samples of each type for good results")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    try:
        subprocess.run([sys.executable, 'train_face_model.py'], check=True)
        print("[SUCCESS] Model training completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Model training failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
        return False

def run_overlay():
    """Run the face overlay system"""
    print("[INFO] Starting face overlay system...")
    
    # Check if model exists
    if not os.path.exists('face_detector_model.h5'):
        print("[ERROR] Trained model not found. Please train the model first using 'python main.py train'")
        return False
    
    print("Instructions:")
    print("- Make sure your game/application is running")
    print("- The system will detect character faces and overlay your webcam face")
    print("- Use the virtual camera in OBS or your streaming software")
    print("- Press Ctrl+C to stop")
    print()
    
    try:
        subprocess.run([sys.executable, 'face_detector.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Face overlay failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n[INFO] Face overlay stopped by user")
    
    return True

def test_detection():
    """Test face detection on screen"""
    print("[INFO] Starting face detection test...")
    print("This will show detected faces on your screen in real-time.")
    print("Press ESC to exit.")
    print()
    
    try:
        subprocess.run([sys.executable, 'template_match_test.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Face detection test failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n[INFO] Test interrupted by user")
    
    return True

def configure():
    """Configure system settings"""
    config_file = 'face_overlay_config.json'
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        print("[INFO] No config file found. Creating default configuration.")
        config = {
            "game_region": {"top": 100, "left": 100, "width": 1000, "height": 1000},
            "input_size": 184,
            "webcam_face_size": 200,
            "fps": 30,
            "face_detection_confidence": 0.7,
            "face_threshold": 0.5,
            "overlay_alpha": 1.0,
            "mirror_webcam": True,
            "show_debug": True
        }
    
    print("\nCurrent Configuration:")
    print(json.dumps(config, indent=2))
    print()
    
    print("Configuration options:")
    print("1. Game region (screen area to capture)")
    print("2. Face detection confidence (0.1-1.0)")
    print("3. Face threshold (0.0-1.0)")
    print("4. Webcam face size (pixels)")
    print("5. Mirror webcam (true/false)")
    print("6. Show debug info (true/false)")
    print("7. Transparent when no face (true/false)")
    print("8. Disable game background (true/false)")
    print("9. Use hybrid detection (true/false)")
    print("10. MediaPipe confidence (0.1-1.0)")
    print("11. Detection smoothing frames (1-10)")
    print("12. Use transparency (true/false)")
    print("13. Frame skip for ML detection (1-5)")
    print("14. Webcam FPS (15-60)")
    print("15. Optimize performance (true/false)")
    print("16. Prioritize camera smoothness (true/false)")
    print("17. Detection interval in seconds (0.05-1.0)")
    print("18. Cache detection time in seconds (1.0-5.0)")
    print("19. High quality mode - 1080p60 (true/false)")
    print("20. Webcam resolution [width, height]")
    print("21. Save and exit")
    
    while True:
        try:
            choice = input("\nSelect option (1-21): ").strip()

            if choice == '1':
                print("Current game region:", config["game_region"])
                print("Run the face overlay to interactively select a new region.")

            elif choice == '2':
                new_val = float(input(f"Face detection confidence ({config['face_detection_confidence']}): "))
                if 0.1 <= new_val <= 1.0:
                    config['face_detection_confidence'] = new_val
                else:
                    print("Value must be between 0.1 and 1.0")

            elif choice == '3':
                new_val = float(input(f"Face threshold ({config['face_threshold']}): "))
                if 0.0 <= new_val <= 1.0:
                    config['face_threshold'] = new_val
                else:
                    print("Value must be between 0.0 and 1.0")

            elif choice == '4':
                new_val = int(input(f"Webcam face size ({config['webcam_face_size']}): "))
                if 50 <= new_val <= 500:
                    config['webcam_face_size'] = new_val
                else:
                    print("Value must be between 50 and 500")

            elif choice == '5':
                new_val = input(f"Mirror webcam ({config['mirror_webcam']}) [true/false]: ").lower()
                if new_val in ['true', 'false']:
                    config['mirror_webcam'] = new_val == 'true'
                else:
                    print("Value must be 'true' or 'false'")

            elif choice == '6':
                new_val = input(f"Show debug info ({config['show_debug']}) [true/false]: ").lower()
                if new_val in ['true', 'false']:
                    config['show_debug'] = new_val == 'true'
                else:
                    print("Value must be 'true' or 'false'")

            elif choice == '7':
                current_val = config.get('transparent_when_no_face', True)
                new_val = input(f"Transparent when no face ({current_val}) [true/false]: ").lower()
                if new_val in ['true', 'false']:
                    config['transparent_when_no_face'] = new_val == 'true'
                else:
                    print("Value must be 'true' or 'false'")

            elif choice == '8':
                current_val = config.get('disable_game_background', False)
                new_val = input(f"Disable game background ({current_val}) [true/false]: ").lower()
                if new_val in ['true', 'false']:
                    config['disable_game_background'] = new_val == 'true'
                else:
                    print("Value must be 'true' or 'false'")

            elif choice == '9':
                current_val = config.get('use_hybrid_detection', True)
                new_val = input(f"Use hybrid detection ({current_val}) [true/false]: ").lower()
                if new_val in ['true', 'false']:
                    config['use_hybrid_detection'] = new_val == 'true'
                else:
                    print("Value must be 'true' or 'false'")

            elif choice == '10':
                current_val = config.get('mediapipe_confidence', 0.3)
                new_val = float(input(f"MediaPipe confidence ({current_val}): "))
                if 0.1 <= new_val <= 1.0:
                    config['mediapipe_confidence'] = new_val
                else:
                    print("Value must be between 0.1 and 1.0")

            elif choice == '11':
                current_val = config.get('detection_smoothing', 3)
                new_val = int(input(f"Detection smoothing frames ({current_val}): "))
                if 1 <= new_val <= 10:
                    config['detection_smoothing'] = new_val
                else:
                    print("Value must be between 1 and 10")

            elif choice == '12':
                current_val = config.get('use_transparency', False)
                new_val = input(f"Use transparency ({current_val}) [true/false]: ").lower()
                if new_val in ['true', 'false']:
                    config['use_transparency'] = new_val == 'true'
                else:
                    print("Value must be 'true' or 'false'")

            elif choice == '13':
                current_val = config.get('skip_frames', 1)
                new_val = int(input(f"Frame skip for ML detection ({current_val}): "))
                if 1 <= new_val <= 5:
                    config['skip_frames'] = new_val
                    print(f"ML detection will run every {new_val} frame(s)")
                else:
                    print("Value must be between 1 and 5")

            elif choice == '14':
                current_val = config.get('webcam_fps', 30)
                new_val = int(input(f"Webcam FPS ({current_val}): "))
                if 15 <= new_val <= 60:
                    config['webcam_fps'] = new_val
                else:
                    print("Value must be between 15 and 60")

            elif choice == '15':
                current_val = config.get('optimize_performance', True)
                new_val = input(f"Optimize performance ({current_val}) [true/false]: ").lower()
                if new_val in ['true', 'false']:
                    config['optimize_performance'] = new_val == 'true'
                else:
                    print("Value must be 'true' or 'false'")

            elif choice == '16':
                current_val = config.get('prioritize_camera_smoothness', False)
                new_val = input(f"Prioritize camera smoothness ({current_val}) [true/false]: ").lower()
                if new_val in ['true', 'false']:
                    config['prioritize_camera_smoothness'] = new_val == 'true'
                    if new_val == 'true':
                        print("Camera smoothness prioritized - detection will run less frequently")
                    else:
                        print("Detection accuracy prioritized - may impact camera smoothness")
                else:
                    print("Value must be 'true' or 'false'")

            elif choice == '17':
                current_val = config.get('detection_interval', 0.1)
                new_val = float(input(f"Detection interval in seconds ({current_val}): "))
                if 0.05 <= new_val <= 1.0:
                    config['detection_interval'] = new_val
                    print(f"ML detection will run every {new_val} seconds")
                else:
                    print("Value must be between 0.05 and 1.0")

            elif choice == '18':
                current_val = config.get('cache_detection_time', 2.0)
                new_val = float(input(f"Cache detection time in seconds ({current_val}): "))
                if 1.0 <= new_val <= 5.0:
                    config['cache_detection_time'] = new_val
                    print(f"Detection results will be cached for {new_val} seconds")
                else:
                    print("Value must be between 1.0 and 5.0")

            elif choice == '19':
                current_val = config.get('high_quality_mode', False)
                new_val = input(f"High quality mode - 1080p60 ({current_val}) [true/false]: ").lower()
                if new_val in ['true', 'false']:
                    config['high_quality_mode'] = new_val == 'true'
                    if new_val == 'true':
                        print("High quality mode enabled - will use full webcam resolution")
                        print("Note: Requires more processing power")
                    else:
                        print("Performance mode enabled - will use 640x480 for better performance")
                else:
                    print("Value must be 'true' or 'false'")

            elif choice == '20':
                current_val = config.get('webcam_resolution', [1920, 1080])
                print(f"Current resolution: {current_val[0]}x{current_val[1]}")
                print("Common resolutions:")
                print("  1920x1080 (1080p)")
                print("  1280x720 (720p)")
                print("  640x480 (480p)")
                try:
                    width = int(input("Enter width: "))
                    height = int(input("Enter height: "))
                    if 320 <= width <= 3840 and 240 <= height <= 2160:
                        config['webcam_resolution'] = [width, height]
                        print(f"Webcam resolution set to {width}x{height}")
                    else:
                        print("Resolution must be between 320x240 and 3840x2160")
                except ValueError:
                    print("Invalid input. Please enter numbers only.")

            elif choice == '21':
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"[INFO] Configuration saved to {config_file}")
                break

            else:
                print("Invalid choice. Please select 1-21.")
                
        except ValueError:
            print("Invalid input. Please try again.")
        except KeyboardInterrupt:
            print("\n[INFO] Configuration cancelled")
            break

def main():
    parser = argparse.ArgumentParser(
        description="Face Overlay for OBS Streaming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'command', 
        nargs='?',
        choices=['collect', 'train', 'run', 'test', 'config'],
        help='Command to execute'
    )
    
    args = parser.parse_args()
    
    print("=== Face Overlay for OBS Streaming ===")
    print()
    
    # Check dependencies and files
    if not check_dependencies():
        return 1
    
    if not check_files():
        return 1
    
    if args.command is None:
        print("Available commands:")
        print("  collect  - Collect training data")
        print("  train    - Train the face detection model")
        print("  run      - Run the face overlay system")
        print("  test     - Test face detection")
        print("  config   - Configure settings")
        print()
        print("Usage: python main.py [command]")
        return 0
    
    # Execute command
    if args.command == 'collect':
        success = collect_data()
    elif args.command == 'train':
        success = train_model()
    elif args.command == 'run':
        success = run_overlay()
    elif args.command == 'test':
        success = test_detection()
    elif args.command == 'config':
        configure()
        success = True
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
