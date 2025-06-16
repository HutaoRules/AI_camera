from app.model import User, Camera, Video, DangerEvent, VideoChunkMetadata
from app import db
import cv2 
import os
from datetime import datetime
import time
from flask import current_app
from collections import deque

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

def write_action(camera_name, timestamp, alert):
    camera = Camera.query.filter_by(name=camera_name).first()
    if not camera:
        current_app.logger.error(f"Camera {camera_name} not found.")
        return

    # Lấy buffer đang chạy tương ứng camera
    buffer = buffer_dict.get(camera.id)
    if not buffer:
        current_app.logger.warning(f"No buffer available for {camera_name}")
        return

    video = create_alert_video(buffer, camera_name, timestamp)
    if not video:
        return

    new_event = DangerEvent(
        video_id=video.id,
        even_type=alert,
        description=f"Alert: {alert} at {timestamp}",
        timestamp=datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S"),
    )

    db.session.add(new_event)
    db.session.commit()


buffer_dict = {}  # Global dict nếu bạn chạy nhiều camera

def record_video(app, camera_id, rtsp_url, duration=1800):
    with app.app_context():
        now = datetime.utcnow()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M")
        static_dir = os.path.join(app.root_path, 'static')
        folder = os.path.join(static_dir, 'videos', str(camera_id), date_str)

        os.makedirs(folder, exist_ok=True)
        filepath = f"videos/{camera_id}/{date_str}/{time_str}.mp4"

        cap = cv2.VideoCapture(rtsp_url)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(static_dir, filepath), fourcc, 20.0, (640, 480))

        alert_frame_buffer = deque(maxlen=300)
        buffer_dict[camera_id] = alert_frame_buffer

        start = time.time()
        while time.time() - start < duration:
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                alert_frame_buffer.append(frame)
            else:
                break

        cap.release()
        out.release()

        new_video_chunk = VideoChunkMetadata(
            camera_id=camera_id,
            chunk_path=filepath,
            timestamp=now  # ✅ đây là kiểu datetime.datetime
        )
        db.session.add(new_video_chunk)
        db.session.commit()


def create_alert_video(buffer, camera_name, timestamp):
    if not buffer:
        return None

    alert_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    static_dir = os.path.join(current_app.root_path, 'static')
    folder = os.path.join(static_dir, 'alert_videos', camera_name, alert_time.strftime("%Y-%m-%d"))

    folder = f"./static/alert_videos/{camera_name}"
    os.makedirs(folder, exist_ok=True)
    filename = alert_time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4"
    path = os.path.join(folder, filename)

    video_path = f"alert_videos/{camera_name}/{alert_time.strftime('%Y-%m-%d')}/{alert_time.strftime('%H-%M-%S')}.mp4"

    height, width, _ = buffer[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 20.0, (width, height))

    for frame in buffer:
        out.write(frame)
    out.release()

    # Ghi vào DB
    camera = Camera.query.filter_by(name=camera_name).first()
    if not camera:
        current_app.logger.error(f"Camera {camera_name} not found.")
        return None

    video = Video(
        camera_id=camera.id,
        file_path=video_path,
        timestamp=timestamp
    )
    db.session.add(video)
    db.session.commit()

    return video

    
                                      