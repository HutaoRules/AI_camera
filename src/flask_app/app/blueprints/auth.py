from flask import Blueprint, render_template, redirect, url_for, request, session, flash, jsonify , Response
from app.model import  User, Camera, Video, DangerEvent
from werkzeug.security import check_password_hash
from app import db

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/')
def home():
    if 'logged_in' in session:
        return redirect(url_for('admin.dashboard'))
    return redirect(url_for('auth.login'))

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username_input = request.form['username']
        password_input = request.form['password']

        user = User.query.filter_by(username=username_input).first()
        if user and user.password == password_input:  # Nên hash password khi triển khai thật
            session['logged_in'] = True
            session['username'] = user.username
            return redirect(url_for('admin.dashboard'))   
        else:
            flash('Sai tài khoản hoặc mật khẩu!', 'danger')
    return render_template('login.html')

@auth_bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('auth.login'))