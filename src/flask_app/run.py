from app import create_app
from app.alert_socket import socketio
from app.model import Camera, Video, DangerEvent, VideoChunkMetadata
from app.ultis import record_video
from threading import Thread
from app import db

#record all cameras when the app starts

app = create_app()


def start_recording_threads():
    with app.app_context():
        camera_list = Camera.query.all()
        for camera in camera_list:
            thread = Thread(target=record_video, args=(app,camera.id, camera.rtsp_url))
            thread.daemon = True
            thread.start()


start_recording_threads()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    