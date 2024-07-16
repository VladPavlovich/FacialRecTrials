import cv2
import face_recognition
import os
import numpy as np
import pickle
from picamera2 import Picamera2, Preview
import time
import tty, sys, termios

class SimpleFacerec:
    def __init__(self, threshold=0.8):
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
sfr = SimpleFacerec(threshold=0.5)
sfr.load_encodings("encodings.pkl")

# Initialize the camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

time.sleep(2)  # Allow the camera to warm up

filedescriptors = termios.tcgetattr(sys.stdin)
tty.setcbreak(sys.stdin)
x = 0
typedkey = ""
print("!~Starting up camera~!")

while typedkey == "":
    frame = picam2.capture_array()
    start_time = time.time()

   
  
    try:
        # Detect Faces
        face_locations, face_names = sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            # Display the results
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            end_time = time.time()
            print(f"Recognized: {name}, time elapsed: {end_time - start_time}")  # Print the recognized name to the terminal
            
    except Exception as e:
        print(f"Error processing frame: {e}")

    # cv2.imshow("Frame", frame)
    x=sys.stdin.read(1)[0]    

    key = cv2.waitKey(1)
    if x == ('q'):
        break
    elif x == ('n'):
        
        # Capture current frame for new face encoding
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame)

        if face_encodings:
            # Assume only one face to add
            new_encoding = face_encodings[0]
            # Prompt for the name in the console
            name = input("Enter name: ")
            if name != "":
                start_time = time.time()
                sfr.known_face_encodings.append(new_encoding)
                sfr.known_face_names.append(name)
                sfr.save_encodings()  # Save the new encoding
                end_time = time.time()
                print(f"Saved new person: {name}, time elapsed: {end_time - start_time}") 
            else:
                print("oops, didn't add a name for the encoding")

      
    print("You pressed", x)
    if x == "r":
        typedkey = "o"

termios.tcsetattr(sys.stdin, termios.TCSADRAIN, filedescriptors)
picam2.close()
print("Camera has been turned off")

cv2.destroyAllWindows()
