import cv2
import face_recognition
import os
import numpy as np
import pickle
import time
from picamera2 import picamera2 

# ... (Rest of your SimpleFacerec class code)

# Initialize SimpleFacerec
sfr = SimpleFacerec(threshold=0.5) 
sfr.load_encoding_images("/Users/vladpavlovich/Desktop/FaceImages/Original Images/Original Images/")

# Initialize picamera2
picam2 = picamera2.Picamera2()
camera_config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)})
picam2.configure(camera_config)
picam2.start()

while True:
    frame = picam2.capture_array("main") # Capture directly as NumPy array

    try:
        # Detect Faces
        face_locations, face_names = sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            # Display the results
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    except Exception as e:
        print(f"Error processing frame: {e}")

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('n'):
        # ... (Your code for adding new faces)

# cv2.destroyAllWindows()
picam2.stop()
