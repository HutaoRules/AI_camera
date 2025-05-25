from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from app.config import Config
from datetime import datetime
from app.alert_socket import socketio


db = SQLAlchemy()


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)
    socketio.init_app(app)
    with app.app_context():
        from app.blueprints import auth, admin, camera , event, api
        app.register_blueprint(auth.auth_bp)
        app.register_blueprint(admin.admin_bp)
        app.register_blueprint(camera.camera_bp)
        app.register_blueprint(event.event_bp)
        app.register_blueprint(api.api_bp)

    return app