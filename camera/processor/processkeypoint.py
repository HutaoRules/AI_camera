import cv2
import numpy as np
import onnxruntime as ort

# Cấu hình
onnx_path = "/home/nvt/Workspaces/AI_Camera/camera/models/onnx/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211/end2end.onnx"
video_path = "/home/nvt/Workspaces/mmaction2/demo/demo_skeleton.mp4"
num_frames = 100
num_person = 3
num_joints = 17
img_size = 640

# Load ONNX
session = ort.InferenceSession(onnx_path)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Không thể mở video.")
    exit()

rtmo_results = []

def apply_nms(boxes, scores, iou_threshold=0.1):
    boxes_cv = [[int(x), int(y), int(x2 - x), int(y2 - y)] for (x, y, x2, y2) in boxes]
    indices = cv2.dnn.NMSBoxes(boxes_cv, scores, score_threshold=0.1, nms_threshold=iou_threshold)
    return indices.flatten() if len(indices) > 0 else []

while len(rtmo_results) < num_frames:
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]
    resized_frame = cv2.resize(frame, (img_size, img_size))
    input_image = resized_frame.astype(np.float32)
    input_image = np.transpose(input_image, (2, 0, 1))
    input_image = np.expand_dims(input_image, axis=0)

    dets, keypoints = session.run(['dets', 'keypoints'], {session.get_inputs()[0].name: input_image})

    boxes = []
    scores = []
    for i in range(dets.shape[1]):
        x_min, y_min, x_max, y_max, score = dets[0, i]
        if score < 0.1:
            continue
        boxes.append([x_min, y_min, x_max, y_max])
        scores.append(float(score))

    keep_indices = apply_nms(boxes, scores)
    poses_in_frame = []

    for idx in keep_indices[:num_person]:
        if idx < keypoints.shape[1]:
            kpts = keypoints[0, idx]  # (17, 3)
            # Scale tọa độ ngược về kích thước gốc
            kpts[:, 0] *= orig_w / img_size
            kpts[:, 1] *= orig_h / img_size
            poses_in_frame.append(kpts)

    rtmo_results.append(poses_in_frame)

cap.release()
cv2.destroyAllWindows()

# ===== Chuyển đổi sang định dạng STGCN++ =====
def build_stgcn_input(rtmo_results, img_width, img_height, num_frames=100, num_person=3, num_joints=17):
    input_tensor = np.zeros((1, 3, num_frames, num_joints, num_person), dtype=np.float32)

    for t in range(min(num_frames, len(rtmo_results))):
        poses = rtmo_results[t]
        poses = sorted(poses, key=lambda p: np.mean(p[:, 2]), reverse=True)

        for m, pose in enumerate(poses[:num_person]):
            x = pose[:, 0] / img_width * 2 - 1
            y = pose[:, 1] / img_height * 2 - 1
            s = pose[:, 2]
            input_tensor[0, 0, t, :, m] = x
            input_tensor[0, 1, t, :, m] = y
            input_tensor[0, 2, t, :, m] = s

    return input_tensor

# Tạo input cho STGCN++
input_tensor = build_stgcn_input(rtmo_results, img_width=orig_w, img_height=orig_h)

print("✅ Định dạng input STGCN++:", input_tensor.shape)
np.save("stgcnpp_input.npy", input_tensor)
print("✅ Đã lưu input thành công vào stgcnpp_input.npy")
