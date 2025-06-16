import cv2
import os
import time
from datetime import datetime
from cassandra_client import get_cassandra_session, insert_video_metadata

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

    # Ghi metadata vào Cassandra
    session = get_cassandra_session()
    insert_video_metadata(session, camera_id, filepath, full_timestamp)
    print(f"[✔] Saved video at {filepath}")
