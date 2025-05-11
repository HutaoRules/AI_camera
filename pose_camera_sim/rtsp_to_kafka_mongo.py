import cv2
import base64
import time
import datetime
from kafka import KafkaProducer
from pymongo import MongoClient, ASCENDING

producer = KafkaProducer(bootstrap_servers='localhost:9092')
mongo = MongoClient('mongodb://localhost:27017/')
db = mongo["video_storage"]
collection = db["video_chunks"]

# TTL Index - MongoDB sẽ xóa video sau 1 ngày
collection.create_index([("createdAt", ASCENDING)], expireAfterSeconds=86400)

cap = cv2.VideoCapture("rtsp://your_camera_stream")

chunk_frames = []
chunk_start_time = time.time()
chunk_duration = 10  # 10s mỗi chunk

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Gửi frame để xử lý keypoint real-time
    _, buffer = cv2.imencode('.jpg', frame)
    b64_frame = base64.b64encode(buffer).decode('utf-8')
    producer.send('camera-frames', b64_frame.encode('utf-8'))

    chunk_frames.append(buffer)

    # Mỗi 10s thì lưu vào MongoDB
    if time.time() - chunk_start_time >= chunk_duration:
        video_id = str(datetime.datetime.utcnow().timestamp())
        collection.insert_one({
            "video_id": video_id,
            "frames": [base64.b64encode(f).decode('utf-8') for f in chunk_frames],
            "start_time": chunk_start_time,
            "end_time": time.time(),
            "createdAt": datetime.datetime.utcnow()
        })
        chunk_frames.clear()
        chunk_start_time = time.time()

cap.release()
