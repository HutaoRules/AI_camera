from flask import Blueprint, request, jsonify
from app.alert_socket import socketio
from app.ultis import gen_frames, write_action
from app.model import db, User, Camera, Video, DangerEvent


api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/receive_alert', methods=['POST'])
def receive_alert():
    data = request.get_json()
    camera_name = data.get('camera_id')
    timestamp = data.get('timestamp')
    alert = data.get('alert')
    camera = Camera.query.filter_by(name=camera_name).first()

    write_action(camera_name, timestamp, alert)

    print(f"[ALERT RECEIVED] {data}")

    # Phát cảnh báo qua socket
    socketio.emit('new_alert', {
        'camera_id': camera.id,
        'timestamp': timestamp,
        'alert': alert
    })

    return jsonify({'status': 'success'}), 200
