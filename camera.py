import cv2
import pyvirtualcam

cap = cv2.VideoCapture(0)
with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
    print(f'Using virtual camera: {cam.device}')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # (You can add face detection and cropping here)

        cam.send(frame)
        cam.sleep_until_next_frame()
