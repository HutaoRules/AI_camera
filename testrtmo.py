import cv2
import numpy as np
import onnxruntime as ort
from src.detection.bytetrack_utils.byte_tracker import BYTETracker, STrack
from argparse import Namespace


session = ort.InferenceSession("/home/nvt/Workspaces/AI_camera/src/detection/end2end.onnx")
video_path = "/home/nvt/Workspaces/AI_camera/src/camera/videos/in-deepIoU.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Không thể mở video.")
    exit()

def apply_nms(boxes, scores, iou_threshold=0.65):
    boxes_cv = [[int(x), int(y), int(x2 - x), int(y2 - y)] for x, y, x2, y2 in boxes]
    indices = cv2.dnn.NMSBoxes(boxes_cv, scores, score_threshold=0.1, nms_threshold=iou_threshold)
    return indices.flatten() if len(indices) > 0 else []


tracker_args = Namespace(
    track_thresh=0.5,
    track_buffer=30,
    match_thresh=0.8,
    frame_rate=30,
    mot20=False
)
tracker = BYTETracker(tracker_args)


frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    input_image = cv2.resize(frame, (640, 640)).astype(np.float32)
    input_image = np.transpose(input_image, (2, 0, 1))[None, :, :, :]

    outputs = session.run(['dets', 'keypoints'], {session.get_inputs()[0].name: input_image})
    dets, keypoints = outputs

    boxes, scores = [], []
    for i in range(dets.shape[1]):
        x1, y1, x2, y2, score = dets[0, i, :5]
        if score < 0.1:
            continue
        x1 = int(x1 * frame.shape[1] / 640)
        y1 = int(y1 * frame.shape[0] / 640)
        x2 = int(x2 * frame.shape[1] / 640)
        y2 = int(y2 * frame.shape[0] / 640)
        boxes.append([x1, y1, x2, y2])
        scores.append(float(score))

    keep_indices = apply_nms(boxes, scores)
    if len(keep_indices) == 0:
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Tạo STrack objects
    detections = []
    for idx in keep_indices:
        x1, y1, x2, y2 = boxes[idx]
        w, h = x2 - x1, y2 - y1
        det_score = scores[idx]
        print(f"Detection {idx}: {x1}, {y1}, {x2}, {y2}, score: {det_score}")
        detections.append(STrack(np.array([x1, y1, x2, y2]), det_score))

    STrack.multi_predict(tracker.tracked_stracks)
    detections_array = np.array([det.tlbr.tolist() + [det.score] for det in detections])
    online_targets = tracker.update(detections_array, [frame.shape[0], frame.shape[1]], (frame.shape[1], frame.shape[0]))

    for t in online_targets:
        x1, y1, x2, y2 = map(int, t.tlbr)
        track_id = t.track_id

        # Vẽ bbox và ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Vẽ keypoints
        if track_id < keypoints.shape[1]:
            for j in range(keypoints.shape[2]):
                x, y, conf = keypoints[0, track_id, j]
                if conf > 0.5:
                    x = int(x * frame.shape[1] / 640)
                    y = int(y * frame.shape[0] / 640)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import onnxruntime as ort

# # Load ONNX model
# # Load ONNX model
# session = ort.InferenceSession("/home/nvt/Workspaces/AI_camera/src/detection/end2end.onnx")
# video_path = "/home/nvt/Workspaces/AI_camera/src/camera/videos/in-deepIoU.mp4"  # Thay 'video_path.mp4' bằng đường dẫn tới video của bạn
# cap = cv2.VideoCapture(video_path)


# if not cap.isOpened():
#     print("Không thể mở video.")
#     exit()

# def apply_nms(boxes, scores, iou_threshold=0.65):
#     boxes_cv = []
#     for box in boxes:
#         x_min, y_min, x_max, y_max = box
#         boxes_cv.append([int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)])
#     indices = cv2.dnn.NMSBoxes(boxes_cv, scores, score_threshold=0.1, nms_threshold=iou_threshold)
#     return indices.flatten() if len(indices) > 0 else []

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     input_image = cv2.resize(frame, (640, 640))
#     input_image = input_image.astype(np.float32)
#     input_image = np.transpose(input_image, (2, 0, 1))
#     input_image = np.expand_dims(input_image, axis=0)

#     outputs = session.run(['dets', 'keypoints'], {session.get_inputs()[0].name: input_image})

#     dets = outputs[0]
#     keypoints = outputs[1]

#     boxes = []
#     scores = []

#     for i in range(dets.shape[1]):
#         box = dets[0, i]
#         x_min, y_min, x_max, y_max, score = box[0], box[1], box[2], box[3], box[4]
#         if score < 0.1:
#             continue
#         x_min = int(x_min * frame.shape[1] / 640)
#         y_min = int(y_min * frame.shape[0] / 640)
#         x_max = int(x_max * frame.shape[1] / 640)
#         y_max = int(y_max * frame.shape[0] / 640)
#         boxes.append([x_min, y_min, x_max, y_max])
#         scores.append(float(score))

#     keep_indices = apply_nms(boxes, scores)

#     print(keep_indices)


#     print(f"Số lượng bbox trong video (sau NMS): {len(keep_indices)}")

#     for idx in keep_indices:
#         x_min, y_min, x_max, y_max = boxes[idx]
#         score = scores[idx]
#         print(f"Bbox {idx}: ({x_min}, {y_min}), ({x_max}, {y_max}), score: {score:.2f}")
#         cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#         cv2.putText(frame, f'{score:.2f}', (x_min, y_min - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         if idx < keypoints.shape[1]:
#             for j in range(keypoints.shape[2]):
#                 x, y, conf = keypoints[0, idx, j]
#                 if conf > 0.5:
#                     x = int(x * frame.shape[1] / 640)
#                     y = int(y * frame.shape[0] / 640)
#                     cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

#     cv2.imshow('Pose Estimation - Video (with NMS)', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# cap.release()
# cv2.destroyAllWindows()