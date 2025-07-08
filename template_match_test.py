import cv2
import numpy as np
import mediapipe as mp
import mss

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

sct = mss.mss()
monitor = sct.monitors[1]  # Change if you want a specific monitor or region

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while True:
        # Capture screen
        screenshot = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detection.process(rgb_frame)

        if results.detections:
            print("Face detected")
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)
        else:
            print("No face detected")

        cv2.imshow("Game Screen Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

cv2.destroyAllWindows()
