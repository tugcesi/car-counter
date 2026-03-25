from ultralytics import YOLO
from sort import Sort
import cv2
import numpy as np

from config import (VIDEO_PATH, MODEL_PATH, VEHICLE_CLASSES, CONF_THRESHOLD,
                    SORT_MAX_AGE, SORT_MIN_HITS, SORT_IOU_THRESHOLD,
                    DETECTION_LINE, LINE_TOLERANCE)

model = YOLO(MODEL_PATH)
tracker = Sort(max_age=SORT_MAX_AGE, min_hits=SORT_MIN_HITS, iou_threshold=SORT_IOU_THRESHOLD)
cap = cv2.VideoCapture(VIDEO_PATH)
counted = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = np.empty((0, 5))
    for r in model(frame, stream=True):
        for box in r.boxes:
            cls_name = model.names[int(box.cls[0])]
            if cls_name in VEHICLE_CLASSES and float(box.conf[0]) > CONF_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections = np.vstack((detections, [x1, y1, x2, y2, float(box.conf[0])]))

    x1_line, y1_line, x2_line, y2_line = DETECTION_LINE
    cv2.line(frame, (x1_line, y1_line), (x2_line, y2_line), (0, 0, 255), 3)

    for x1, y1, x2, y2, id in tracker.update(detections):
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(frame, str(id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)

        # Line crossing check (assumes horizontal detection line)
        if x1_line < cx < x2_line and y1_line - LINE_TOLERANCE < cy < y1_line + LINE_TOLERANCE:
            if id not in counted:
                counted.add(id)
                cv2.line(frame, (x1_line, y1_line), (x2_line, y2_line), (0, 255, 0), 3)

    cv2.putText(frame, f"Count: {len(counted)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imshow("Vehicle Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
