import cv2
import numpy as np
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession("/home/nvt/Workspaces/AI_Camera/camera/models/onnx/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211/end2end.onnx")
image = cv2.imread("/home/nvt/Workspaces/AI_Camera/demo.jpg")  # Đọc ảnh từ file
video_path = "/home/nvt/Workspaces/mmaction2/demo/demo_skeleton.mp4"  # Thay 'video_path.mp4' bằng đường dẫn tới video của bạn
cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    print("Không thể mở video.")
    exit()

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

    dets = outputs[0]
    keypoints = outputs[1]

    boxes = []
    scores = []

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

    print(f"Số lượng bbox trong video (sau NMS): {len(keep_indices)}")
    print(f"Số lượng người trong video: {keypoints.shape[1]}")

    for idx in keep_indices:
        x_min, y_min, x_max, y_max = boxes[idx]
        score = scores[idx]

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, f'{score:.2f}', (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if idx < keypoints.shape[1]:
            for j in range(keypoints.shape[2]):
                x, y, conf = keypoints[0, idx, j]
                if conf > 0.5:
                    x = int(x * frame.shape[1] / 640)
                    y = int(y * frame.shape[0] / 640)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    cv2.imshow('Pose Estimation - Video (with NMS)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()