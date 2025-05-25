from tracker import STrack, ByteTracker
import onnxruntime as ort
import numpy as np
from typing import List, Tuple
import cv2


tracker = ByteTracker()

def apply_nms(boxes, scores, iou_threshold=0.65):
    boxes_cv = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        boxes_cv.append([int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)])
    indices = cv2.dnn.NMSBoxes(boxes_cv, scores, score_threshold=0.1, nms_threshold=iou_threshold)
    return indices.flatten() if len(indices) > 0 else []

def rtmo_inference(
    model_path: str,
    input_data: np.ndarray,
    input_shape: Tuple[int, int],
    conf_thres: float = 0.5,
    nms_thres: float = 0.4,
) -> List[STrack]:
    """
    Run inference on the RTMO model.

    Args:
        model_path (str): Path to the ONNX model.
        input_data (np.ndarray): Input data for the model.
        input_shape (Tuple[int, int]): Shape of the input data.
        conf_thres (float): Confidence threshold for detections.
        nms_thres (float): Non-maximum suppression threshold.

    Returns:
        List[STrack]: List of detected tracks.
    """
    # Load the ONNX model
    session = ort.InferenceSession(model_path)

    # Prepare input data
    input_name = session.get_inputs()[0].name
    input_data = np.expand_dims(input_data, axis=0)
    #resize input data to the model's expected input shape
    input_data = cv2.resize(input_data, input_shape)
    input_data = input_data.astype(np.float32)
    input_data = np.transpose(input_data, (2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0)
    
    # Run inference
    output = session.run(['dets', 'keypoints'], {input_name: input_data})

    # Extract detections and keypoints
    detections = output[0]
    keypoints = output[1]

    boxes = []
    scores = []
    for i in range(detections.shape[1]):
        box = detections[0, i]
        x_min, y_min, x_max, y_max, score = box[0], box[1], box[2], box[3], box[4]
        if score < conf_thres:
            continue
        x_min = int(x_min * input_shape[1] / 640)
        y_min = int(y_min * input_shape[0] / 640)
        x_max = int(x_max * input_shape[1] / 640)
        y_max = int(y_max * input_shape[0] / 640)
        boxes.append([x_min, y_min, x_max, y_max])
        scores.append(float(score))
    keep_indices = apply_nms(boxes, scores, iou_threshold=nms_thres)
    detections = []
    for idx in keep_indices:
        x_min, y_min, x_max, y_max = boxes[idx]
        score = scores[idx]
        detection = [x_min, y_min, x_max, y_max, score]
        detections.append(detection)
    
    #apply tracker
    for idx in keep_indices:
        x_min, y_min, x_max, y_max = boxes[idx]
        score = scores[idx]
        tracker.update([x_min, y_min, x_max, y_max], score)
        print(f"Tracker update: {x_min}, {y_min}, {x_max}, {y_max}, {score}")

    # Process detections
    tracks = []
    for detection in detections:
        if detection[4] > conf_thres:
            track = STrack(detection[:4], detection[4])
            tracks.append(track)

    # Apply NMS
    tracker.update_tracks(tracks, frame_id=0, match_thresh=nms_thres)

    return tracker.tracked_tracks
