from flask import Blueprint, render_template, redirect, url_for, request, session, flash, jsonify , Response
from app.model import db, User, Camera, Video, DangerEvent
from werkzeug.security import check_password_hash
from app import db
from app.ultis import gen_frames, write_action
from datetime import datetime

event_bp = Blueprint('event', __name__)



@event_bp.route('/danger-events', methods=['GET'])
def danger_events():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    # Nhận tham số tìm kiếm từ form
    selected_camera = request.args.get('camera', '')
    selected_start_timestamp = request.args.get('start_timestamp', '')
    selected_end_timestamp = request.args.get('end_timestamp', '')
    selected_behavior = request.args.get('behavior', '')

    # Lọc các sự kiện theo các tiêu chí
    query = DangerEvent.query

    # Lọc theo camera
    if selected_camera:
        query = query.join(Video).filter(Video.camera_id == selected_camera)

    # Lọc theo khoảng thời gian
    if selected_start_timestamp:
        start_timestamp = datetime.strptime(selected_start_timestamp, '%Y-%m-%dT%H:%M')
        query = query.filter(DangerEvent.timestamp >= start_timestamp)

    if selected_end_timestamp:
        end_timestamp = datetime.strptime(selected_end_timestamp, '%Y-%m-%dT%H:%M')
        query = query.filter(DangerEvent.timestamp <= end_timestamp)

    # Lọc theo hành vi
    if selected_behavior:
        query = query.filter(DangerEvent.event_type == selected_behavior)

    events = query.order_by(DangerEvent.timestamp.desc()).all()

    # Lấy tất cả camera để hiển thị trong form
    cameras = Camera.query.all()

    return render_template('danger_events.html', events=events, cameras=cameras, 
                           selected_camera=selected_camera, 
                           selected_start_timestamp=selected_start_timestamp,
                           selected_end_timestamp=selected_end_timestamp,
                           selected_behavior=selected_behavior)

# @event_bp.route("/api/receive_action", methods=["POST"])
# def receive_action():
#     data = request.json
#     write_action(data)
#     return jsonify({"status": "received"})
