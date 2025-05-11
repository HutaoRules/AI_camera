import cv2
import time
from kafka import KafkaProducer
import base64

producer = KafkaProducer(bootstrap_servers='localhost:9092')

cap = cv2.VideoCapture("video_demo.mp4")  # video đầu vào giả lập

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    _, buffer = cv2.imencode('.jpg', frame)
    producer.send('camera-frames', base64.b64encode(buffer))
    time.sleep(1/30)  # giả lập 30fps

cap.release()
producer.close()
