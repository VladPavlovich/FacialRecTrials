import cv2
import face_recognition
import os
import numpy as np
import pickle
from picamera2 import Picamera2, Preview
import time
import paho.mqtt.client as mqtt

class SimpleFacerec:
    def __init__(self, threshold=0.8):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25
        self.threshold = threshold

    def load_encoding_images(self, images_path, encodings_file="encodings.pkl"):
        if os.path.exists(encodings_file):
            with open(encodings_file, "rb") as f:
                self.known_face_encodings, self.known_face_names = pickle.load(f)
            print(f"Loaded encodings from {encodings_file}")
            return

        images_path = os.path.abspath(images_path)
        print(f"Loading images from {images_path}")

        for dirpath, dnames, fnames in os.walk(images_path):
            for f in fnames:
                if f.endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(dirpath, f)
                    print(f"Loading image {img_path}")

                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Failed to load image {img_path}")
                        continue

                    print(f"Image properties: shape={img.shape}, dtype={img.dtype}")

                    try:
                        if img.dtype != np.uint8:
                            img = img.astype(np.uint8)

                        img_encoding = face_recognition.face_encodings(img)
                        if img_encoding:
                            img_encoding = img_encoding[0]
                            self.known_face_encodings.append(img_encoding)
                            self.known_face_names.append(os.path.basename(dirpath))
                            print(f"Encoded image {img_path}")
                        else:
                            print(f"No faces found in image {img_path}")
                    except Exception as e:
                        print(f"Could not process image {img_path}: {e}")

        with open(encodings_file, "wb") as f:
            pickle.dump((self.known_face_encodings, self.known_face_names), f)
        print(f"Saved encodings to {encodings_file}")

    def save_encodings(self, encodings_file="encodings.pkl"):
        with open(encodings_file, "wb") as f:
            pickle.dump((self.known_face_encodings, self.known_face_names), f)
        print(f"Saved encodings to {encodings_file}")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(
            frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing
        )
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, tolerance=self.threshold
            )
            name = "Unknown"

            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)
            if (
                matches[best_match_index]
                and face_distances[best_match_index] < self.threshold
            ):
                name = self.known_face_names[best_match_index]

            face_names.append(name)

        face_locations = np.array(face_locations) / self.frame_resizing
        face_locations = face_locations.astype(int)

        return face_locations, face_names


# Initialize SimpleFacerec with a specific threshold value
sfr = SimpleFacerec(threshold=0.5)
sfr.load_encoding_images(
    "/Users/vladpavlovich/Desktop/FaceImages/Original Images/Original Images/"
)

# Initialize the camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

time.sleep(2)  # Allow the camera to warm up

# MQTT settings
broker_address = "broker.hivemq.com"  # Use your broker address
port = 1883
topic = "face_recognition/names"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print("Failed to connect, return code %d\n", rc)

mqtt_client = mqtt.Client("RaspberryPi")
mqtt_client.on_connect = on_connect

mqtt_client.connect(broker_address, port, 60)
mqtt_client.loop_start()

while True:
    frame = picam2.capture_array()

    try:
        face_locations, face_names = sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            cv2.putText(
                frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            print(f"Recognized: {name}")  # Print the recognized name to the terminal

            # Publish the recognized name to the MQTT broker
            mqtt_client.publish(topic, name)
    except Exception as e:
        print(f"Error processing frame: {e}")

    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break
    elif key & 0xFF == ord("n"):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame)

        if face_encodings:
            new_encoding = face_encodings[0]
            name = input("Enter name: ")
            sfr.known_face_encodings.append(new_encoding)
            sfr.known_face_names.append(name)
            sfr.save_encodings()

cv2.destroyAllWindows()
mqtt_client.loop_stop()
mqtt_client.disconnect()
