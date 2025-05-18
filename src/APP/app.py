from flask import Flask, render_template, redirect, url_for, request, session, flash, jsonify , Response
from model import db, User, Camera, Video, DangerEvent
from werkzeug.security import check_password_hash
import datetime
import os
import cv2

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)


@app.route('/')
def home():
    if 'logged_in' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username_input = request.form['username']
        password_input = request.form['password']

        user = User.query.filter_by(username=username_input).first()
        if user and user.password == password_input:  # Nên hash password khi triển khai thật
            session['logged_in'] = True
            session['username'] = user.username
            return redirect(url_for('dashboard'))
        else:
            flash('Sai tài khoản hoặc mật khẩu!', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'add':
            name = request.form.get('name')
            location = request.form.get('location')
            if name and location:
                new_camera = Camera(name=name, location=location)
                db.session.add(new_camera)
                db.session.commit()
                flash('Thêm camera thành công!')

        elif action == 'delete':
            camera_id = request.form.get('camera_id')
            camera = Camera.query.get(camera_id)
            if camera:
                for video in camera.videos:
                    db.session.delete(video)
                db.session.delete(camera)
                db.session.commit()
                flash('Xoá camera và video liên quan thành công!')

        elif action == 'delete_video':
            video_id = request.form.get('video_id')
            video = Video.query.get(video_id)
            if video:
                db.session.delete(video)
                db.session.commit()
                flash(f'Đã xoá video: {video.file_path}')

    cameras = Camera.query.all()
    return render_template('admin.html', cameras=cameras)


@app.route('/cameras')
def cameras():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    cameras = Camera.query.all()
    return render_template('camera_list.html', cameras=cameras)

@app.route('/cameras/<int:camera_id>')
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



@app.route('/danger-events', methods=['GET'])
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

@app.route("/api/receive_action", methods=["POST"])
def receive_action():
    data = request.json
    write_action(data)
    return jsonify({"status": "received"})

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

def gen_frames(camera_id):
    # Giả sử bạn đã có một hàm để lấy video từ camera
    # Ví dụ: Lấy video từ RTSP stream
    camera_rtsp = Camera.query.get(camera_id).rtsp_url
    cap = cv2.VideoCapture(camera_rtsp)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return Response(gen_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
