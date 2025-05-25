import cv2
import numpy as np
import onnxruntime as ort
from src.detection.bytetrack_utils.byte_tracker import BYTETracker, STrack

def apply_nms(boxes, scores, iou_threshold=0.65):
    boxes_cv = [[int(x), int(y), int(x2 - x), int(y2 - y)] for x, y, x2, y2 in boxes]
    indices = cv2.dnn.NMSBoxes(boxes_cv, scores, score_threshold=0.1, nms_threshold=iou_threshold)
    return indices.flatten() if len(indices) > 0 else []

def process_frame(frame, session, tracker):
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
        return []

    detections = []
    for idx in keep_indices:
        x1, y1, x2, y2 = boxes[idx]
        det_score = scores[idx]
        detections.append(STrack(np.array([x1, y1, x2, y2]), det_score))

    STrack.multi_predict(tracker.tracked_stracks)
    detections_array = np.array([det.tlbr.tolist() + [det.score] for det in detections])
    online_targets = tracker.update(detections_array, [frame.shape[0], frame.shape[1]], (frame.shape[1], frame.shape[0]))

    results = []
    for t in online_targets:
        pid = t.track_id
        x1, y1, x2, y2 = map(int, t.tlbr)

        # Lấy keypoints tương ứng với pid nếu có
        if pid < keypoints.shape[1]:
            kps = []
            for j in range(keypoints.shape[2]):
                x, y, conf = keypoints[0, pid, j]
                kps.append((float(x), float(y), float(conf)))
            results.append({
                'pid': pid,
                'keypoints': kps
            })

    return results