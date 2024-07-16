import cv2
import face_recognition
import os
import numpy as np
import pickle
from picamera2 import Picamera2, Preview
import time
import tty, sys, termios
import requests
import pyaudio
import wave
import speech_recognition as sr

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

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index] and face_distances[best_match_index] < self.threshold:
                name = self.known_face_names[best_match_index]

            face_names.append(name)

        return face_locations, face_names

def send_name_to_api(name):
    url = 'http://3.138.158.57:8000/names'
    payload = {'name': name}
    try:
        print("Attempting to send name to API...")
        response = requests.post(url, json=payload)
        # Check if the request was successful
        if response.status_code == 200:
            print("Successfully sent name to API.")
        else:
            print(f"Failed to send name to API, status code: {response.status_code}")
        print("Response from API:", response.text)
    except requests.RequestException as e:
        print(f"Exception occurred when sending data to API: {e}")

def record_audio(duration=60, output_file="output.wav"):
    # Record audio for a given duration and save to a file
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 44100
    filename = output_file

    p = pyaudio.PyAudio()
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    print('Recording...')

    frames = []

    for i in range(0, int(fs / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print('Finished recording.')

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        transcription = recognizer.recognize_sphinx(audio)
        print("Transcription: " + transcription)
        return transcription
    except sr.UnknownValueError:
        print("Sphinx could not understand audio")
        return ""
    except sr.RequestError as e:
        print("Sphinx error; {0}".format(e))
        return ""

def send_transcription_to_api(transcription):
    url = 'http://3.138.158.57:8000/transcriptions'
    payload = {'transcription': transcription}
    try:
        print("Attempting to send transcription to API...")
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Successfully sent transcription to API.")
        else:
            print(f"Failed to send transcription to API, status code: {response.status_code}")
        print("Response from API:", response.text)
    except requests.RequestException as e:
        print(f"Exception occurred when sending data to API: {e}")

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
            if name != "Unknown":
                send_name_to_api(name)  # Send the recognized name to the API
            else:
                # Record and transcribe audio when an unknown person is detected
                audio_file = "output.wav"
                record_audio(output_file=audio_file)
                transcription = transcribe_audio(audio_file)
                if transcription:  # Check if transcription is not empty
                    send_transcription_to_api(transcription)

    except Exception as e:
        print(f"Error processing frame: {e}")

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('n'):
        # Capture current frame for new face encoding
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame)

        if face_encodings:
            # Assume only one face to add
            new_encoding = face_encodings[0]
            # Prompt for the name in the console
            name = input("Enter name: ")
            sfr.known_face_encodings.append(new_encoding)
            sfr.known_face_names.append(name)
            sfr.save_encodings()  # Save the new encoding

    x = sys.stdin.read(1)[0]
    print("You pressed", x)
    if x == "r":
        print("If condition is met")
    if x == "r":
        typedkey = "o"

termios.tcsetattr(sys.stdin, termios.TCSADRAIN, filedescriptors)
picam2.close()
print("Camera has been turned off")

cv2.destroyAllWindows()
