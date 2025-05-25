import onnxruntime
import numpy as np
import requests
from collections import defaultdict, deque
import time

# Load ONNX model
sess = onnxruntime.InferenceSession("stgcnppbone_48_5_17_model.onnx")

# Buffer keypoints theo camera with sliding window approach
FRAME_SEQUENCE_LENGTH = 48
SLIDING_WINDOW_STEP = 8  # Process every 8 frames instead of waiting for 48 new frames

# Buffer keypoints theo camera
buffer_dict = defaultdict(lambda: deque(maxlen=FRAME_SEQUENCE_LENGTH))
frame_counters = defaultdict(int)  # Track frames for sliding window

# Counter theo pid (person_id) để phát hiện lảng vảng
loitering_counter = defaultdict(lambda: defaultdict(int))  # {camera_id: {pid: count}}

crowd_counter = defaultdict(int)

# Ngưỡng phát hiện lảng vảng
LOITERING_THRESHOLD = 10000
CROWD_PERSON_THRESHOLD = 10
CROWD_FRAME_THRESHOLD = 1000

def preprocess_keypoints(keypoints_dict):
    if isinstance(keypoints_dict["keypoints"], list):
        keypoints_dict["keypoints"] = np.array(keypoints_dict["keypoints"])
    keypoints_dict["keypoints"] = keypoints_dict["keypoints"].astype(np.float32)
    keypoints_dict["keypoints"] = keypoints_dict["keypoints"].reshape(-1, 2)
    return keypoints_dict

def normalize_pose(pose):
    """Normalize pose coordinates to improve action recognition"""
    # Find valid keypoints (with confidence > threshold)
    valid_mask = pose[:, 2] > 0
    if np.sum(valid_mask) < 2:  # Need at least 2 valid keypoints
        return pose
    
    valid_kpts = pose[valid_mask, :2]
    
    # Calculate scale based on valid keypoints
    min_x, min_y = np.min(valid_kpts, axis=0)
    max_x, max_y = np.max(valid_kpts, axis=0)
    
    # Avoid division by zero
    width = max(max_x - min_x, 1e-5)
    height = max(max_y - min_y, 1e-5)
    scale = max(width, height)
    
    # Get center
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Normalize pose
    norm_pose = pose.copy()
    norm_pose[valid_mask, 0] = (pose[valid_mask, 0] - center_x) / scale + 0.5
    norm_pose[valid_mask, 1] = (pose[valid_mask, 1] - center_y) / scale + 0.5
    
    return norm_pose

def normalize_frame(frame_kpts):
    """Normalize all keypoints in a frame collectively for better group-level consistency"""
    valid_mask = frame_kpts[:, :, 2] > 0
    valid_kpts = frame_kpts[:, :, :2][valid_mask]

    if valid_kpts.shape[0] < 2:
        return frame_kpts  # Not enough valid points

    min_xy = np.min(valid_kpts, axis=0)
    max_xy = np.max(valid_kpts, axis=0)
    
    width = max(max_xy[0] - min_xy[0], 1e-5)
    height = max(max_xy[1] - min_xy[1], 1e-5)
    scale = max(width, height)

    center_x = (min_xy[0] + max_xy[0]) / 2
    center_y = (min_xy[1] + max_xy[1]) / 2

    norm_frame_kpts = frame_kpts.copy()
    for i in range(frame_kpts.shape[0]):
        for j in range(frame_kpts.shape[1]):
            if frame_kpts[i, j, 2] > 0:
                norm_frame_kpts[i, j, 0] = (frame_kpts[i, j, 0] - center_x) / scale + 0.5
                norm_frame_kpts[i, j, 1] = (frame_kpts[i, j, 1] - center_y) / scale + 0.5

    return norm_frame_kpts

def perform_action_recognition(camera_id, pid, timestamp):
    """Run the action recognition inference on the current buffer"""
    if len(buffer_dict[camera_id]) < FRAME_SEQUENCE_LENGTH:
        return  # Not enough frames yet
        
    pose_sequence = np.stack(list(buffer_dict[camera_id]), axis=0)
    pose_sequence = np.transpose(pose_sequence, (1, 0, 2, 3))
    input_shape = (1, 2, 48, 17, 3)
    pose_input = np.zeros(input_shape, dtype=np.float32)
    
    for m in range(min(pose_sequence.shape[0], 2)):
        for t in range(min(pose_sequence.shape[1], 48)):
            for v in range(17):
                for c in range(3):  # x, y, score
                    pose_input[0, m, t, v, c] = pose_sequence[m, t, v, c]

    input_feed = {sess.get_inputs()[0].name: pose_input}
    pred = sess.run(None, input_feed)[0]
    label = int(np.argmax(pred))
    action = ""

    # top 1 hành động 
    if label == 0:
        action = "fight"
    elif label == 1:
        action = "falling"
    elif label == 2:
        action = "walking"

    if label == 0 or label == 1:
        payload = {
            "camera_id": camera_id,
            "pid": pid,
            "timestamp": timestamp,
            "action": action
        }

        try:
            requests.post("http://flask-server-ip:5000/api/receive_action", json=payload)
            print(f"[ACTION] Detected {action} for PID={pid} on Camera={camera_id}")
        except Exception as e:
            print(f"Failed to send action result: {e}")

def run_inference(keypoints_dict):
    camera_id = keypoints_dict["camera_id"]
    pid = keypoints_dict["pid"]
    keypoints = keypoints_dict["keypoints"]
    timestamp = keypoints_dict["timestamp"]

    # tinh so luong nguoi dua tren so luong person_id
    num_people = len(set(keypoints_dict["pid"]))

    if num_people > CROWD_PERSON_THRESHOLD:
        crowd_counter[camera_id] += 1
        if crowd_counter[camera_id] >= CROWD_FRAME_THRESHOLD:
            payload = {
                "camera_id": camera_id,
                "timestamp": timestamp,
                "alert": "crowd"
            }
            try:
                requests.post("http://flask-server-ip:5000/api/receive_alert", json=payload)
                print(f"[ALERT] Crowd detected on Camera={camera_id}")
            except Exception as e:
                print(f"Failed to send crowd alert: {e}")
            # Reset lại count để không gửi lại liên tục
            crowd_counter[camera_id] = 0

    # Cập nhật đếm số frame người này xuất hiện
    loitering_counter[camera_id][pid] += 1

    # Nếu quá ngưỡng lảng vảng, gửi cảnh báo
    if loitering_counter[camera_id][pid] >= LOITERING_THRESHOLD:
        payload = {
            "camera_id": camera_id,
            "pid": pid,
            "timestamp": timestamp,
            "alert": "loitering"
        }
        try:
            requests.post("http://flask-server-ip:5000/api/receive_alert", json=payload)
            print(f"[ALERT] Loitering detected for PID={pid} on Camera={camera_id}")
        except Exception as e:
            print(f"Failed to send loitering alert: {e}")
        # Reset lại count để không gửi lại liên tục
        loitering_counter[camera_id][pid] = 0

    # Lưu keypoints vào buffer để nhận diện hành động
    buffer_dict[camera_id].append(keypoints)
    frame_counters[camera_id] += 1

    # Implement sliding window approach:
    # Run inference when buffer is full and every SLIDING_WINDOW_STEP frames after that
    if len(buffer_dict[camera_id]) == FRAME_SEQUENCE_LENGTH or (
        len(buffer_dict[camera_id]) >= FRAME_SEQUENCE_LENGTH and 
        frame_counters[camera_id] % SLIDING_WINDOW_STEP == 0
    ):
        perform_action_recognition(camera_id, pid, timestamp)
        
        # Instead of clearing the buffer, we keep it for sliding window
        # Remove oldest SLIDING_WINDOW_STEP frames when we reach capacity
        if len(buffer_dict[camera_id]) == FRAME_SEQUENCE_LENGTH and frame_counters[camera_id] % SLIDING_WINDOW_STEP == 0:
            # No need to manually remove elements as deque will handle this automatically
            # We just reset the counter for the next window
            pass