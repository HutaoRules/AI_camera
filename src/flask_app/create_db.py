from flask import Flask
from app.model import  User, Camera, Video, DangerEvent
from app import db, create_app
from datetime import datetime

def create_database():
    app = create_app()
    # Tạo database và các bảng

    with app.app_context():
        db.drop_all()     # Xóa nếu muốn reset database
        db.create_all()

        # Thêm user mẫu
        user = User(username='admin', password='admin')  # Nên mã hóa password khi triển khai
        db.session.add(user)

        # Thêm camera mẫu
        cam1 = Camera(name='Camera 1', location='Cổng chính', rtsp_url='rtsp://example.com/cam1')
        cam2 = Camera(name='Camera 2', location='Tầng hầm', rtsp_url='rtsp://example.com/cam2')
        db.session.add_all([cam1, cam2])
        db.session.commit()

        # Thêm video mẫu
        vid1 = Video(camera_id=1, file_path='videos/camera_1/video1.mp4')
        db.session.add(vid1)
        db.session.commit()

        # Thêm event nguy hiểm mẫu
        event1 = DangerEvent(video_id=1, event_type='Ngủ gật', description='Tài xế nhắm mắt hơn 5s')
        db.session.add(event1)
        db.session.commit()


if __name__ == '__main__':
    create_database()
    print("✅ Database đã được tạo thành công.")