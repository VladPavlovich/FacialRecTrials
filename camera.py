from picamera2 import Picamera2, Preview
from time import sleep

picam2 = Picamera2()
picam2.start_preview(Preview.QTGL)
picam2.start()
sleep(5)

picam2.start_and_capture_files("images{:d}.jpg", num_files=3, delay=0.5)

picam2.start_and_record_video("new_video.mp4", duration=5, show_preview=True)


picam2.close()
