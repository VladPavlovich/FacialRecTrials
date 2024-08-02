import cv2
import face_recognition
import os
import numpy as np
import pickle
import time


class SimpleFacerec:
    def __init__(self, threshold=0.9):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25
        self.threshold = threshold

    def load_encodings(self, encodings_file="encodings.pkl"):
        # Load encodings if they exist
        if os.path.exists(encodings_file):
            with open(encodings_file, 'rb') as f:
                self.known_face_encodings, self.known_face_names = pickle.load(f)
            print(f"Loaded encodings from {encodings_file}")
            print(f"Number of known face encodings: {len(self.known_face_encodings)}")
            print(f"Number of known face names: {len(self.known_face_names)}")
        else:
            print(f"Encodings file {encodings_file} not found. Please provide a valid file.")
            raise FileNotFoundError

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

            if np.any(matches):  # Check if there are any matches
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
sfr = SimpleFacerec(threshold=0.9)
sfr.load_encodings("encodings3.pkl")


# Function to process an image file
def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Unable to load image {image_path}")
        return

    start_time = time.time()

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    if face_locations.size == 0:  # Check if face_locations is empty
        print("No faces detected in the image.")
    else:
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            # Display the results
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            end_time = time.time()
            print(f"Recognized: {name}, time elapsed: {end_time - start_time}")  # Print the recognized name to the terminal

    # Show the image with the annotations
    cv2.imshow("Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
image_path = "/Users/vladpavlovich/Desktop/Replicated Faces/Tom Cruise/1.jpg"
process_image(image_path)
