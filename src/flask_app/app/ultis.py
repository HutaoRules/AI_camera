from app.model import User, Camera, Video, DangerEvent, VideoChunkMetadata
from app import db
import cv2 
import os
from datetime import datetime
import time
from flask import current_app

def gen_frames(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def write_action(data):
    # Giả sử bạn đã có một hàm để ghi dữ liệu vào cơ sở dữ liệu
    # Ví dụ: Ghi vào bảng DangerEvent
    new_event = DangerEvent(
        camera_id=data['camera_id'],
        event_type=data['event_type'],
        timestamp=datetime.datetime.now(),
        video_id=data['video_id']
    )
    db.session.add(new_event)
    db.session.commit()

def record_video(camera_id, rtsp_url, duration=1800):  # 30 phút = 1800s
    now = datetime.utcnow()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M")

    folder = f"./videos/{camera_id}/{date_str}"
    os.makedirs(folder, exist_ok=True)

    filepath = f"{folder}/{time_str}.mp4"
    full_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    cap = cv2.VideoCapture(rtsp_url)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, 20.0, (640, 480))

    start = time.time()
    while time.time() - start < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            break

    cap.release()
    out.release()
    # Ghi metadata vào cơ sở dữ liệu
    new_video_chunk = VideoChunkMetadata(
        camera_id=camera_id,
        file_path=filepath,
        timestamp=full_timestamp
    )
    db.session.add(new_video_chunk)
    db.session.commit()