import cv2
import numpy as np
import onnxruntime as ort
import os
import time

# === Configuration ===
rtmo_model_path = "/home/nvt/Workspaces/AI_camera/src/detection/end2end.onnx"
stgcn_model_path = "/home/nvt/Workspaces/AI_camera/src/recognition/stgcnppbone_48_5_17_model.onnx"
video_path = "/home/nvt/Downloads/in-deepIoU.mp4"  # Path to your video file
image_size = 640
num_frames = 48
num_person = 5
num_keypoints = 17
min_confidence = 0  # Increased minimum confidence for keypoints
detection_threshold = 0.1  # Increased detection threshold

print(f"Processing video: {video_path}")
print(f"Using RTMO model: {os.path.basename(rtmo_model_path)}")
print(f"Using STGCN++ model: {os.path.basename(stgcn_model_path)}")


# COCO skeleton connection
SKELETON = [
    (5, 7), (7, 9),     # left arm
    (6, 8), (8, 10),    # right arm
    (5, 6),             # shoulders
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16), # right leg
    (11, 12),           # hips
    (5, 11), (6, 12),   # body
    (0, 1), (0, 2),     # eyes
    (1, 3), (2, 4)      # ears
]

def draw_keypoints(frame, keypoints, conf_thresh=0.3):
    for person in keypoints:
        for i in range(person.shape[0]):
            x, y, c = person[i]
            if c > conf_thresh:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        for pair in SKELETON:
            i, j = pair
            if person[i][2] > conf_thresh and person[j][2] > conf_thresh:
                x1, y1 = int(person[i][0]), int(person[i][1])
                x2, y2 = int(person[j][0]), int(person[j][1])
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

# === Load RTMO ===
rtmo_session = ort.InferenceSession(rtmo_model_path)

# === Load video ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError(f"Could not open video: {video_path}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video has {total_frames} frames at {fps} FPS")

# Function to apply non-maximum suppression
def apply_nms(boxes, scores, iou_threshold=0.65):  # Increased IoU threshold
    if not boxes:
        return []
    boxes_cv = [[int(x), int(y), int(x2 - x), int(y2 - y)] for x, y, x2, y2 in boxes]
    indices = cv2.dnn.NMSBoxes(boxes_cv, scores, detection_threshold, iou_threshold)
    return indices.flatten() if len(indices) > 0 else []

# === Process video frames ===
pose_sequence = []
frame_count = 0
start_time = time.time()

# Function to normalize pose for better action recognition
def normalize_pose(pose):
    """Normalize pose coordinates to improve action recognition"""
    # Find valid keypoints (with confidence > threshold)
    valid_mask = pose[:, 2] > min_confidence
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
    valid_mask = frame_kpts[:, :, 2] > min_confidence
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
            if frame_kpts[i, j, 2] > min_confidence:
                norm_frame_kpts[i, j, 0] = (frame_kpts[i, j, 0] - center_x) / scale + 0.5
                norm_frame_kpts[i, j, 1] = (frame_kpts[i, j, 1] - center_y) / scale + 0.5

    return norm_frame_kpts

# Initialize tracking for better temporal consistency
prev_boxes = []
prev_ids = []

# Process frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    #skip frames to speed up processing
    if frame_count % 5 != 0:
        frame_count += 1
        continue
    # Resize frame for RTMO
    
    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
    
    # Prepare input for RTMO
    resized = cv2.resize(frame, (image_size, image_size))
    inp = resized.astype(np.float32)
    inp = np.transpose(inp, (2, 0, 1))[None]
    
    # Run RTMO detection
    dets, keypoints = rtmo_session.run(['dets', 'keypoints'], {rtmo_session.get_inputs()[0].name: inp})
    
    boxes, scores = [], []
    for i in range(dets.shape[1]):
        x1, y1, x2, y2, score = dets[0, i]
        if score > detection_threshold:
            boxes.append([x1, y1, x2, y2])
            scores.append(float(score))
    # Apply NMS
    keep = apply_nms(boxes, scores)
    frame_kpts = np.zeros((num_person, num_keypoints, 3), dtype=np.float32)
    
    # Extract keypoints for kept detections
    for i, idx in enumerate(keep[:num_person]):
        if idx < keypoints.shape[1]:
            person_kpts = np.zeros((num_keypoints, 3), dtype=np.float32)
            for j in range(num_keypoints):
                x, y, c = keypoints[0, idx, j]
                # Only take keypoints with reasonable confidence
                if c > min_confidence:
                    person_kpts[j] = [x / image_size, y / image_size, c]  # normalize coordinates
                else:
                    person_kpts[j] = [0, 0, 0]  # Zero out low confidence points
            # Normalize all persons in frame together

            # Apply normalization to improve recognition
            frame_kpts[i] = person_kpts  # gán thô keypoints, chưa normalize
    frame_kpts = normalize_frame(frame_kpts)

    pose_sequence.append(frame_kpts)
    
    # Limit to required frames
    if len(pose_sequence) >= num_frames:
        break
    # Check for exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
processing_time = time.time() - start_time
print(f"Processed {frame_count} frames in {processing_time:.2f} seconds ({frame_count/processing_time:.1f} FPS)")
print(f"Collected {len(pose_sequence)} pose frames")

# === Process pose sequence ===
pose_sequence = np.stack(pose_sequence, axis=1)  # (num_person, T, 17, 3)

# Print shape and check data integrity
print(f"Initial pose sequence shape: {pose_sequence.shape}")


# Pad if not enough frames
if pose_sequence.shape[1] < num_frames:
    pad_len = num_frames - pose_sequence.shape[1]
    print(f"Padding with {pad_len} frames to reach {num_frames} frames")
    
    # Instead of zero padding, repeat the last frames
    if pose_sequence.shape[1] > 0:
        last_frame = pose_sequence[:, -1:, :, :]
        pad = np.repeat(last_frame, pad_len, axis=1)
    else:
        pad = np.zeros((num_person, pad_len, num_keypoints, 3), dtype=np.float32)
    
    pose_sequence = np.concatenate([pose_sequence, pad], axis=1)

# Prepare input for STGCN++ - Careful with dimension order
input_shape = (1, num_person, num_frames, num_keypoints, 3)
pose_input = np.zeros(input_shape, dtype=np.float32)

# Fill the tensor with our data (properly aligned)
for m in range(min(pose_sequence.shape[0], num_person)):
    for t in range(min(pose_sequence.shape[1], num_frames)):
        for v in range(num_keypoints):
            for c in range(3):  # x, y, confidence
                pose_input[0, m, t, v, c] = pose_sequence[m, t, v, c]

print(f"Final input shape: {pose_input.shape}")

# === Enhanced data validation ===
print("\nData validation:")
print(f"Input data type: {pose_input.dtype}")
print(f"Input min value: {np.min(pose_input)}")
print(f"Input max value: {np.max(pose_input)}")
print(f"Input mean value: {np.mean(pose_input)}")
print(f"Contains NaN values: {np.isnan(pose_input).any()}")
print(f"Contains Inf values: {np.isinf(pose_input).any()}")

# Check if we have consistent data across time
temporal_consistency = np.mean(np.abs(np.diff(pose_input[0, :, 1:, :, :2], axis=1)))
print(f"Temporal consistency (lower is better): {temporal_consistency:.6f}")

# === Load STGCN++ ONNX ===
stgcn_session = ort.InferenceSession(stgcn_model_path)
input_name = stgcn_session.get_inputs()[0].name

# Print model details
print("\nModel Input Details:")
for i, input_info in enumerate(stgcn_session.get_inputs()):
    print(f"Input {i}: {input_info.name} - Shape: {input_info.shape}")

# Run inference
print("\nRunning inference...")
start_infer = time.time()
logits = stgcn_session.run(None, {input_name: pose_input})[0]
inference_time = time.time() - start_infer
print(f"Inference completed in {inference_time:.4f} seconds")

# === Process results ===
# Apply softmax
logits = np.array(logits)  # đảm bảo là numpy array
logits = logits - np.max(logits, axis=1, keepdims=True)  # tránh overflow
probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

# Get number of classes in the model
num_classes = probs.shape[1]
print(f"\nTotal number of classes in model: {num_classes}")
print(f"Model output index range: 0-{num_classes-1}")


# Get top predictions
top_k = 5  # Show more predictions to see if 55 is close
top_indices = np.argsort(probs[0])[::-1][:top_k]

print(f"\n🎯 Top-{top_k} predictions:")
for i, idx in enumerate(top_indices):
    print(f"  {i+1}. Class {idx}: {probs[0][idx]*100:.2f}%")

# Check specifically for class 55
class_54_prob = probs[0][54] * 100 if 54 < probs.shape[1] else 0
print(f"\nClass 54 probability: {class_54_prob:.4f}%")
if 54 in top_indices:
    rank = list(top_indices).index(54) + 1
    print(f"Class 54 is ranked #{rank}")
else:
    print("Class 54 is not in the top results")

print("\n🔥 Top prediction: Class", top_indices[0], f"({probs[0][top_indices[0]]*100:.2f}%)")