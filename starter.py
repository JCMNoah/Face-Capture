import cv2
import numpy as np
import mediapipe as mp
import pyvirtualcam

# Init MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
ret, frame = cap.read()
if not ret:
    print("Failed to access webcam")
    exit()

height, width = frame.shape[:2]

with pyvirtualcam.Camera(width=width, height=height, fps=30) as cam:
    print(f"Virtual cam started: {cam.device}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)

        overlay_frame = np.zeros_like(frame)

        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * width)
            y = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)

            # Ensure cropping within frame bounds
            x, y = max(0, x), max(0, y)
            x2, y2 = min(x + w, width), min(y + h, height)

            face_crop = frame[y:y2, x:x2]

            # Optional resize (so it's consistent regardless of camera distance)
            target_w, target_h = 200, 200
            face_crop = cv2.resize(face_crop, (target_w, target_h))

            # Optional mirror
            face_crop = cv2.flip(face_crop, 1)

            # Create circular mask
            mask = np.zeros((target_h, target_w), dtype=np.uint8)
            cv2.circle(mask, (target_w // 2, target_h // 2), min(target_w, target_h) // 2, 255, -1)
            mask_3ch = cv2.merge([mask, mask, mask])

            # Center coordinates
            top_left_y = (height - target_h) // 2
            top_left_x = (width - target_w) // 2
            bottom_right_y = top_left_y + target_h
            bottom_right_x = top_left_x + target_w

            # Blend into black overlay frame
            roi = overlay_frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            fg = cv2.bitwise_and(face_crop, mask_3ch)
            bg = cv2.bitwise_and(roi, 255 - mask_3ch)
            overlay_frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = cv2.add(bg, fg)

        # Send to virtual camera
        cam.send(cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB))
        cam.sleep_until_next_frame()
