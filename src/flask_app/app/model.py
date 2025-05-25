from datetime import datetime
from app import db


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Camera(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80))
    location = db.Column(db.String(120))
    rtsp_url = db.Column(db.String(200))


    videos = db.relationship('Video', backref='camera', lazy=True)

class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.Integer, db.ForeignKey('camera.id'))
    file_path = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class DangerEvent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.Integer, db.ForeignKey('video.id'))
    event_type = db.Column(db.String(100))
    description = db.Column(db.String(300))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class VideoChunkMetadata(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.Integer, db.ForeignKey('camera.id'))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    chunk_path = db.Column(db.String(200), nullable=False)