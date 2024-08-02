import os
import cv2
import face_recognition
import pickle

class FaceEncoder:
    def __init__(self, threshold=0.8):
        self.known_face_encodings = []
        self.known_face_names = []
        self.threshold = threshold

    def encode_faces(self, images_path, encodings_file="encodings3.pkl"):
        images_path = os.path.abspath(images_path)
        print(f"Loading images from {images_path}")

        for dirpath, _, fnames in os.walk(images_path):
            for f in fnames:
                if f.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(dirpath, f)
                    print(f"Loading image {img_path}")

                    # Read the image file using OpenCV
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Failed to load image {img_path}")
                        continue

                    try:
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

# Directory containing images organized by person
images_directory = "/Users/vladpavlovich/Desktop/FaceImages/Original Images/Original Images/"

# Initialize FaceEncoder and encode images
face_encoder = FaceEncoder()
face_encoder.encode_faces(images_directory)
