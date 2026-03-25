# Configuration parameters for vehicle detection and counting

VIDEO_PATH = "cars.mp4"
MODEL_PATH = "yolov8n.pt"

# Vehicle classes to detect
VEHICLE_CLASSES = ["car", "truck", "bus", "motorbike"]

# Minimum detection confidence
CONF_THRESHOLD = 0.3

# SORT tracker settings
SORT_MAX_AGE = 20
SORT_MIN_HITS = 3
SORT_IOU_THRESHOLD = 0.3

# Detection line [x1, y1, x2, y2]
DETECTION_LINE = [400, 297, 673, 297]
LINE_TOLERANCE = 15
