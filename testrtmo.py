import cv2
import numpy as np
import onnxruntime as ort
from src.detection.bytetrack_utils.byte_tracker import BYTETracker

# Load ONNX model
session = ort.InferenceSession("/home/nvt/Workspaces/AI_camera/src/detection/end2end.onnx")
video_path = "/home/nvt/Workspaces/AI_camera/src/camera/videos/in-deepIoU.mp4"  # Thay 'video_path.mp4' bằng đường dẫn tới video của bạn
cap = cv2.VideoCapture(video_path)


tracker = BYTETracker(
    track_thresh=0.5,
    track_buffer=30,
    match_thresh=0.8,
    frame_rate=30,
)
img_size = (640, 640)  # Kích thước đầu vào của mô hình

def apply_nms(boxes, scores, iou_threshold=0.65):
    boxes_cv = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        boxes_cv.append([int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)])
    indices = cv2.dnn.NMSBoxes(boxes_cv, scores, score_threshold=0.1, nms_threshold=iou_threshold)
    return indices.flatten() if len(indices) > 0 else []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_image = cv2.resize(frame, (640, 640))
    input_image = input_image.astype(np.float32)
    input_image = np.transpose(input_image, (2, 0, 1))
    input_image = np.expand_dims(input_image, axis=0)

    outputs = session.run(['dets', 'keypoints'], {session.get_inputs()[0].name: input_image})
    dets, keypoints = outputs[0], outputs[1]

    boxes, scores = [], []
    for i in range(dets.shape[1]):
        box = dets[0, i]
        x_min, y_min, x_max, y_max, score = box[0], box[1], box[2], box[3], box[4]
        if score < 0.1:
            continue
        x_min = int(x_min * frame.shape[1] / 640)
        y_min = int(y_min * frame.shape[0] / 640)
        x_max = int(x_max * frame.shape[1] / 640)
        y_max = int(y_max * frame.shape[0] / 640)
        boxes.append([x_min, y_min, x_max, y_max])
        scores.append(float(score))

    keep_indices = apply_nms(boxes, scores)
    print(keep_indices)
    print(f"Số lượng bbox trong video (sau NMS): {len(keep_indices)}")

    dets_for_tracker = []
    index_mapping = {}  # ánh xạ local index → RTMO index

    for local_idx, idx in enumerate(keep_indices):
        x_min, y_min, x_max, y_max = boxes[idx]
        score = scores[idx]
        cls_id = 0
        dets_for_tracker.append([x_min, y_min, x_max, y_max, score, cls_id])
        index_mapping[local_idx] = idx  # ánh xạ đến index keypoints

    dets_for_tracker = np.array(dets_for_tracker)
    online_targets = tracker.update(dets_for_tracker, frame.shape[:2], frame.shape[:2], (640, 640))

    for track in online_targets:
        tlwh = track.tlwh
        track_id = track.track_id
        x1, y1, w, h = map(int, tlwh)
        x2, y2 = x1 + w, y1 + h

        # Vẽ bbox và ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Tìm bbox gần nhất trong dets_for_tracker (theo IOU)
        best_idx = -1
        max_iou = 0
        for local_idx, det in enumerate(dets_for_tracker):
            xA = max(det[0], x1)
            yA = max(det[1], y1)
            xB = min(det[2], x2)
            yB = min(det[3], y2)
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (det[2] - det[0]) * (det[3] - det[1])
            boxBArea = (x2 - x1) * (y2 - y1)
            iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
            if iou > max_iou and iou > 0.1:
                max_iou = iou
                best_idx = local_idx

        # Vẽ keypoints nếu tìm được index tương ứng
        if best_idx != -1 and best_idx in index_mapping:
            kp_idx = index_mapping[best_idx]
            if kp_idx < keypoints.shape[1]:
                for j in range(keypoints.shape[2]):
                    x, y, conf = keypoints[0, kp_idx, j]
                    if conf > 0.5:
                        x = int(x * frame.shape[1] / 640)
                        y = int(y * frame.shape[0] / 640)
                        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    cv2.imshow('RTMO Pose Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()