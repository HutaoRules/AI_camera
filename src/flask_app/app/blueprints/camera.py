from flask import Blueprint, render_template, redirect, url_for, request, session, flash, jsonify , Response
from app.model import User, Camera, Video, DangerEvent, VideoChunkMetadata
from werkzeug.security import check_password_hash
from app import db
from app.ultis import gen_frames,record_video

camera_bp = Blueprint('camera', __name__)

@camera_bp.route('/cameras')
def cameras():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    cameras = Camera.query.all()
    return render_template('camera_list.html', cameras=cameras)

@camera_bp.route('/cameras/<int:camera_id>')
def camera_videos(camera_id):
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    camera = Camera.query.get_or_404(camera_id)
    videos = Video.query.filter_by(camera_id=camera.id).all()

    # Lấy các sự kiện nguy hiểm liên quan
    video_event_map = {}
    for video in videos:
        events = DangerEvent.query.filter_by(video_id=video.id).all()
        video_event_map[video] = events

    return render_template('camera_videos.html', camera=camera, video_event_map=video_event_map)

@camera_bp.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    camera = Camera.query.get(camera_id)
    if not camera:
        return "Camera not found", 404
    # Kiểm tra quyền truy cập camera
    record_video(camera.id, camera.rtsp_url)

    return Response(gen_frames(camera.rtsp_url),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@camera_bp.route('/camera/<int:camera_id>/recordings')
def view_recordings(camera_id):
    camera = Camera.query.get_or_404(camera_id)
    chunks = VideoChunkMetadata.query.filter_by(camera_id=camera_id).order_by(VideoChunkMetadata.timestamp.desc()).all()
    return render_template('recordings.html', camera=camera, chunks=chunks)
