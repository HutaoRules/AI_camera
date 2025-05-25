from flask import Blueprint, render_template, redirect, url_for, request, session, flash, jsonify , Response
from app.model import User, Camera, Video, DangerEvent
from werkzeug.security import check_password_hash
from app import db


admin_bp = Blueprint('admin', __name__)

@admin_bp.route('/dashboard')
def dashboard():
    if 'logged_in' not in session:
        return redirect(url_for('auth.login'))
    return render_template('dashboard.html')

@admin_bp.route('/admin', methods=['GET', 'POST'])
def admin():
    if 'logged_in' not in session:
        return redirect(url_for('auth.login'))

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'add':
            name = request.form.get('name')
            location = request.form.get('location')
            rtsp_url = request.form.get('ip_address')
            if name and location:
                new_camera = Camera(name=name, location=location, rtsp_url=rtsp_url)
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

@admin_bp.route('/admin/edit/<int:camera_id>', methods=['GET', 'POST'])
def edit_camera(camera_id):
    if 'logged_in' not in session:
        return redirect(url_for('auth.login'))

    camera = Camera.query.get_or_404(camera_id)

    if request.method == 'POST':
        camera.name = request.form.get('name')
        camera.location = request.form.get('location')
        camera.rtsp_url = request.form.get('ip_address')
        db.session.commit()
        flash('Cập nhật thông tin camera thành công!')
        return redirect(url_for('admin.admin'))

    return render_template('edit_camera.html', camera=camera)