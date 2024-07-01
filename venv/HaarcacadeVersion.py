import face_recognition
import cv2
import os
import numpy as np
import pickle

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25
        self.haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

    def detect_known_faces(self, frame):
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)

        # Convert the image to grayscale for the Haar Cascade
        gray_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using Haar Cascade
        faces = self.haar_cascade.detectMultiScale(gray_small_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        face_locations = []
        for (x, y, w, h) in faces:
            face_locations.append((y, x + w, y + h, x))

        # Convert the image from BGR to RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            face_names.append(name)

        # Adjust face locations back to the original frame size
        face_locations = np.array(face_locations) / self.frame_resizing
        face_locations = face_locations.astype(int)

        return face_locations, face_names

if __name__ == "__main__":
    # Encode faces from the specified folder
    sfr = SimpleFacerec()
    sfr.load_encoding_images("/Users/vladpavlovich/Downloads/FaceDataSet/Original Images/Original Images/")

    # Load Camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

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

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
