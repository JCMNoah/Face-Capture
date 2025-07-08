import os
import cv2
import numpy as np
import mss
from glob import glob

# Create dataset folders if they don't exist
os.makedirs("dataset/face", exist_ok=True)
os.makedirs("dataset/no_face", exist_ok=True)

sct = mss.mss()
full_monitor = sct.monitors[1]

def count_existing_files():
    """Count existing files to continue numbering"""
    face_files = glob("dataset/face/face_*.png")
    no_face_files = glob("dataset/no_face/no_face_*.png")

    # Extract numbers from filenames and find the highest
    face_count = 0
    for f in face_files:
        try:
            num = int(os.path.basename(f).split('_')[1].split('.')[0])
            face_count = max(face_count, num + 1)
        except:
            continue

    no_face_count = 0
    for f in no_face_files:
        try:
            num = int(os.path.basename(f).split('_')[2].split('.')[0])
            no_face_count = max(no_face_count, num + 1)
        except:
            continue

    return face_count, no_face_count

def select_roi():
    print("\n=== Region Selection ===")
    print("1. A screenshot window will appear")
    print("2. Click and drag to select the region where character faces appear")
    print("3. Press ENTER or SPACE to confirm selection")
    print("4. Press ESC to cancel")

    try:
        img = np.array(sct.grab(full_monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Resize image if too large for display
        height, width = img.shape[:2]
        if width > 1920 or height > 1080:
            scale = min(1920/width, 1080/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            display_img = cv2.resize(img, (new_width, new_height))
            scale_factor = scale
        else:
            display_img = img.copy()
            scale_factor = 1.0

        print("Select the face region ROI and press ENTER or SPACE")
        roi = cv2.selectROI("Select Face Region", display_img, False)
        cv2.destroyAllWindows()

        if roi[2] == 0 or roi[3] == 0:
            print("No region selected. Using default center region.")
            return {
                "top": full_monitor["top"] + height//4,
                "left": full_monitor["left"] + width//4,
                "width": width//2,
                "height": height//2
            }

        # Adjust coordinates back to original scale
        roi_scaled = [int(x/scale_factor) for x in roi]

        return {
            "top": full_monitor["top"] + roi_scaled[1],
            "left": full_monitor["left"] + roi_scaled[0],
            "width": roi_scaled[2],
            "height": roi_scaled[3]
        }
    except Exception as e:
        print(f"Error during region selection: {e}")
        print("Using default center region.")
        return {
            "top": full_monitor["top"] + 200,
            "left": full_monitor["left"] + 200,
            "width": 800,
            "height": 600
        }

# Count existing files to continue numbering
count_face, count_no_face = count_existing_files()
print(f"\n=== Data Collection ===")
print(f"Existing files: {len(glob('dataset/face/face_*.png'))} face, {len(glob('dataset/no_face/no_face_*.png'))} no_face")
print(f"Next file numbers: face_{count_face}, no_face_{count_no_face}")

# Initial region selection
monitor = select_roi()
print(f"Selected region: {monitor}")

print("\n=== Controls ===")
print("Press 'f' to save as FACE")
print("Press 'n' to save as NO_FACE")
print("Press 'r' to re-select region")
print("Press 'q' to quit")
print("Press 's' to show current stats")
print("\nCapture window will appear...")

try:
    while True:
        try:
            # Capture the selected region
            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Show current frame
            display_frame = frame.copy()

            # Add text overlay with instructions
            cv2.putText(display_frame, f"Face: {count_face} | No-Face: {count_no_face}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "f=Face, n=No-Face, r=Region, s=Stats, q=Quit",
                       (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Screenshot Collector - Face Region", display_frame)
            key = cv2.waitKey(1) & 0xFF  # Check for key press every 1ms

            if key == ord('f'):
                filename = f"dataset/face/face_{count_face}.png"
                success = cv2.imwrite(filename, frame)
                if success:
                    print(f"✓ Saved FACE screenshot: {filename}")
                    count_face += 1
                else:
                    print(f"✗ Failed to save {filename}")

            elif key == ord('n'):
                filename = f"dataset/no_face/no_face_{count_no_face}.png"
                success = cv2.imwrite(filename, frame)
                if success:
                    print(f"✓ Saved NO_FACE screenshot: {filename}")
                    count_no_face += 1
                else:
                    print(f"✗ Failed to save {filename}")

            elif key == ord('r'):
                cv2.destroyAllWindows()
                print("\nReselecting region...")
                monitor = select_roi()
                print(f"New region: {monitor}")

            elif key == ord('s'):
                total_face = len(glob('dataset/face/face_*.png'))
                total_no_face = len(glob('dataset/no_face/no_face_*.png'))
                print(f"\n=== Current Stats ===")
                print(f"Total FACE samples: {total_face}")
                print(f"Total NO_FACE samples: {total_no_face}")
                print(f"This session: {count_face - count_existing_files()[0]} face, {count_no_face - count_existing_files()[1]} no_face")
                print(f"Recommended: 100+ of each type for good results")

            elif key == ord('q'):
                print("\n=== Final Stats ===")
                total_face = len(glob('dataset/face/face_*.png'))
                total_no_face = len(glob('dataset/no_face/no_face_*.png'))
                print(f"Total FACE samples: {total_face}")
                print(f"Total NO_FACE samples: {total_no_face}")
                print("Exiting data collection.")
                break

        except Exception as e:
            print(f"Error during capture: {e}")
            print("Continuing...")
            continue

except KeyboardInterrupt:
    print("\nInterrupted by user.")
finally:
    cv2.destroyAllWindows()
    print("Data collection finished.")
