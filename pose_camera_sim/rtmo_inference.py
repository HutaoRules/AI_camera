import cv2
import numpy as np
import onnxruntime as ort
import json
import time
from kafka import KafkaProducer
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.track import STrack

# Load ONNX model
session = ort.InferenceSession("/home/nvt/Workspaces/AI_Camera/camera/models/onnx/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211/end2end.onnx")

video_path = '/home/nvt/Workspaces/AI_Camera/camera/camera_simulator/videos/demo.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Không thể mở video.")
    exit()

# Kafka producer setup
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# BYTETracker config
tracker = BYTETracker(
    args=dict(
        track_thresh=0.3,
        match_thresh=0.8,
        track_buffer=30,
        frame_rate=30
    ),
    frame_rate=30
)


def apply_nms(boxes, scores, iou_threshold=0.1):
    boxes_cv = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        boxes_cv.append([int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)])
    indices = cv2.dnn.NMSBoxes(boxes_cv, scores, score_threshold=0.1, nms_threshold=iou_threshold)
    return indices.flatten() if len(indices) > 0 else []


frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % 2 != 0:
        continue

    input_image = cv2.resize(frame, (640, 640))
    input_image = input_image.astype(np.float32)
    input_image = np.transpose(input_image, (2, 0, 1))
    input_image = np.expand_dims(input_image, axis=0)

    outputs = session.run(['dets', 'keypoints'], {session.get_inputs()[0].name: input_image})
    dets = outputs[0]
    keypoints = outputs[1]

    boxes = []
    scores = []
    ori_w, ori_h = frame.shape[1], frame.shape[0]

    for i in range(dets.shape[1]):
        box = dets[0, i]
        x_min, y_min, x_max, y_max, score = box[0], box[1], box[2], box[3], box[4]
        if score < 0.1:
            continue
        x_min = int(x_min * ori_w / 640)
        y_min = int(y_min * ori_h / 640)
        x_max = int(x_max * ori_w / 640)
        y_max = int(y_max * ori_h / 640)
        boxes.append([x_min, y_min, x_max, y_max])
        scores.append(float(score))

    keep_indices = apply_nms(boxes, scores)

    dets_for_tracking = []
    for idx in keep_indices:
        x_min, y_min, x_max, y_max = boxes[idx]
        score = scores[idx]
        w = x_max - x_min
        h = y_max - y_min
        dets_for_tracking.append([x_min, y_min, w, h, score])

    # Track with ByteTrack
    tracks = tracker.update(np.array(dets_for_tracking), [ori_h, ori_w], [ori_h, ori_w])

    results = []
    for track in tracks:
        tlwh = track.tlwh
        track_id = track.track_id
        x_min, y_min, w, h = map(int, tlwh)
        x_max = x_min + w
        y_max = y_min + h

        person = {
            "id": track_id,
            "bbox": [x_min, y_min, x_max, y_max],
            "keypoints": []
        }

        # Find the closest bbox from original detection to get keypoints
        found_idx = -1
        for i, box in enumerate(boxes):
            if abs(box[0] - x_min) < 10 and abs(box[1] - y_min) < 10:
                found_idx = i
                break

        if found_idx >= 0 and found_idx < keypoints.shape[1]:
            for j in range(keypoints.shape[2]):
                x, y, conf = keypoints[0, found_idx, j]
                if conf > 0.5:
                    x = int(x * ori_w / 640)
                    y = int(y * ori_h / 640)
                    person["keypoints"].append([x, y, conf])

        results.append(person)

        # Draw for debug
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, f'ID:{track_id}', (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for kp in person["keypoints"]:
            cv2.circle(frame, (kp[0], kp[1]), 3, (0, 255, 0), -1)

    # Send result to Kafka
    message = {
        "timestamp": time.time(),
        "frame_id": frame_id,
        "results": results
    }
    producer.send("keypoints-topic", message)

    # Optional: display video
    cv2.imshow('RTMO + ByteTrack Inference', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
