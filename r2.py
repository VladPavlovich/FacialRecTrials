import cv2
import face_recognition
import os
import numpy as np
import pickle
from picamera2.array import PiRGBArray
from picamera2 import PiCamera
import time

class SimpleFacerec:
    def __init__(self, threshold=0.8):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25
        self.threshold = threshold

    def load_encoding_images(self, images_path, encodings_file="encodings.pkl"):
        # Load encodings if they exist
        if os.path.exists(encodings_file):
            with open(encodings_file, 'rb') as f:
                self.known_face_encodings, self.known_face_names = pickle.load(f)
            print(f"Loaded encodings from {encodings_file}")
            return

        # Otherwise, load images and create encodings
        images_path = os.path.abspath(images_path)
        print(f"Loading images from {images_path}")

        for dirpath, dnames, fnames in os.walk(images_path):
            for f in fnames:
                if f.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(dirpath, f)
                    print(f"Loading image {img_path}")

                    # Read the image file using OpenCV
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Failed to load image {img_path}")
                        continue

                    # Print image properties for diagnosis
                    print(f"Image properties: shape={img.shape}, dtype={img.dtype}")

                    try:
                        # Ensure image is 8-bit
                        if img.dtype != np.uint8:
                            img = img.astype(np.uint8)

                        # Encode the face
                        img_encoding = face_recognition.face_encodings(img)
                        if img_encoding:
                            img_encoding = img_encoding[0]
                            self.known_face_encodings.append(img_encoding)
                            self.known_face_names.append(os.path.basename(dirpath))  # Use folder name as person's name
                            print(f"Encoded image {img_path}")
                        else:
                            print(f"No faces found in image {img_path}")
                    except Exception as e:
                        print(f"Could not process image {img_path}: {e}")

        # Save the encodings to a file
        with open(encodings_file, 'wb') as f:
            pickle.dump((self.known_face_encodings, self.known_face_names), f)
        print(f"Saved encodings to {encodings_file}")

    def save_encodings(self, encodings_file="encodings.pkl"):
        # Save the encodings to a file
        with open(encodings_file, 'wb') as f:
            pickle.dump((self.known_face_encodings, self.known_face_names), f)
        print(f"Saved encodings to {encodings_file}")

    def detect_known_faces(self, frame):
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)

        # Convert the image from BGR to RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=self.threshold)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index] and face_distances[best_match_index] < self.threshold:
                name = self.known_face_names[best_match_index]

            face_names.append(name)

        # Adjust face locations back to the original frame size
        face_locations = np.array(face_locations) / self.frame_resizing
        face_locations = face_locations.astype(int)

        return face_locations, face_names


# Initialize SimpleFacerec with a specific threshold value
sfr = SimpleFacerec(threshold=0.5)  # Adjust the threshold value here
sfr.load_encoding_images("/Users/vladpavlovich/Desktop/FaceImages/Original Images/Original Images/")

# Initialize the camera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
raw_capture = PiRGBArray(camera, size=(640, 480))

# Allow the camera to warm up
time.sleep(0.1)

for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
    image = frame.array

    try:
        # Detect Faces
        face_locations, face_names = sfr.detect_known_faces(image)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            # Display the results
            cv2.putText(image, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    except Exception as e:
        print(f"Error processing frame: {e}")

    cv2.imshow("Frame", image)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('n'):
        # Capture current frame for new face encoding
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame)

        if face_encodings:
            # Assume only one face to add
            new_encoding = face_encodings[0]
            # Prompt for the name in the console
            name = input("Enter name: ")
            sfr.known_face_encodings.append(new_encoding)
            sfr.known_face_names.append(name)
            sfr.save_encodings()  # Save the new encoding

    # Clear the stream in preparation for the next frame
    raw_capture.truncate(0)

cv2.destroyAllWindows()
