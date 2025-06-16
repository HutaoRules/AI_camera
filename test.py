import cv2

cap = cv2.VideoCapture("rtsp://localhost:8554/webcam")
while True:
    success, frame = cap.read()
    if not success:
        print("Failed to read from stream")
        break
    cv2.imshow("test", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
