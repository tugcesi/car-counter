# Car Counter

A clean, minimal vehicle detection and counting system for highway traffic monitoring using YOLOv8 and SORT tracking.

## Features

- Detects vehicles (cars, trucks, buses, motorcycles) using YOLOv8n
- Tracks vehicles across frames with SORT for consistent IDs
- Counts vehicles crossing a defined detection line
- Displays real-time count and bounding boxes

## Setup

```bash
pip install -r requirements.txt
```

> **Note:** The SORT tracker requires the `sort` package. Install it from the official repo if needed:
> ```bash
> pip install git+https://github.com/abewley/sort.git
> ```

## Usage

1. Place your video file as `cars.mp4` in the project directory (or update `VIDEO_PATH` in `config.py`).
2. Run the script:

```bash
python main.py
```

Press **Q** to quit.

## Configuration

Edit `config.py` to customize:

| Parameter | Description |
|-----------|-------------|
| `VIDEO_PATH` | Path to input video |
| `MODEL_PATH` | YOLOv8 model file |
| `VEHICLE_CLASSES` | Classes to detect |
| `CONF_THRESHOLD` | Minimum confidence |
| `DETECTION_LINE` | `[x1, y1, x2, y2]` counting line |
| `LINE_TOLERANCE` | Crossing detection margin (px) |
