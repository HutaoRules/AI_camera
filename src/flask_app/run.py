from app import create_app
from app.alert_socket import socketio

app = create_app()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    