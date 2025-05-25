from rtmo import tracker, process_frame
import cv2
from pub_sub_producer import publisher, topic_path, send_keypoints
import onnxruntime as ort
from argparse import Namespace
from src.detection.bytetrack_utils.byte_tracker import BYTETracker
import datetime

onnx_path ="/home/nvt/Workspaces/AI_camera/src/detection/end2end.onnx"
session = ort.InferenceSession(onnx_path)

tracker_args = Namespace(
    track_thresh=0.5,
    track_buffer=30,
    match_thresh=0.8,
    frame_rate=30,
    mot20=False
)

tracker = BYTETracker(tracker_args)


def main(video_path, cam_id):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Không thể mở video.")
        return

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        #skip frame de tang toc do xu ly 30->15FPS
        if frame_id % 2 != 0:
            continue

        results = process_frame(frame, session, tracker)
        #tack pid va keypoints
        if not results:
            continue
        for result in results:
            pid = result['pid']
            keypoints = result['keypoints']

        # Gửi kết quả lên Pub/Sub
        send_keypoints({
            'cam_id': cam_id,
            'pid': pid,
            'keypoints': keypoints,
            'timestamp': datetime.datetime.now().isoformat()
        })

        # Hiển thị khung hình (tuỳ chọn)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()