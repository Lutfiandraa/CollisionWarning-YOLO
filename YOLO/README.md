# Dashcam Forward Collision Warning

Real-time forward collision warning system for dashcam footage using YOLOv8n object detection and DeepSort multi-object tracking, built with OpenCV and Python.

---

## Features

- Real-time vehicle detection (car, bus, truck, motorcycle) via YOLOv8n pretrained on COCO
- Multi-object tracking with persistent track IDs across frames (DeepSort, IOU-based, CPU-optimized)
- Monocular distance estimation per tracked vehicle using pinhole camera model
- Three-level collision status: SAFE, CAUTION, WARNING — per vehicle and globally
- Visual overlay: colored bounding boxes, track ID, distance label, status text, warning banner
- Optional audio alert with cooldown and anti-flicker persistence

---

## Project Structure

```
DashcamYOLO/
├── YOLO/
│   ├── main.py            Entry point: webcam loop, pipeline orchestration
│   ├── detector.py        YOLOv8n inference, returns List[Detection]
│   ├── tracker.py         DeepSort wrapper, returns List[TrackedVehicle] with stable IDs
│   ├── collision_logic.py Area/delta analysis, distance estimation, SAFE/CAUTION/WARNING logic
│   ├── alert.py           OpenCV visual rendering and audio alert
│   └── config.py          All thresholds, parameters, and constants
├── assets/sounds/         Optional alert.wav audio file
├── requirements.txt
└── README.md
```

---

## Installation

Requires **Python 3.8 or higher**.

```bash
pip install -r requirements.txt
```

The YOLOv8n model (`yolov8n.pt`) is downloaded automatically on first run.

For audio alerts, place an `alert.wav` file in `assets/sounds/`. If the file is absent, the system runs silently without errors.

---

## Running

Run from the project root (`DashcamYOLO/`):

```bash
python YOLO/main.py
```

Press **`q`** in the camera window to stop the program.

If the camera fails to open, check `CAMERA_INDEX` in `YOLO/config.py` (default: `0`).

---

## Technical Notes

### Distance Estimation

The displayed distance (`X.Xm`) is an **approximation**, not a precision measurement. It is computed using the monocular pinhole camera model:

```
distance = (real_vehicle_width * focal_length_px) / bbox_width_px
```

Default assumptions: vehicle width = 1.8 m, focal length = 700 px. These values are **not calibrated** and vary by vehicle type and camera lens. The estimate is suitable as a qualitative proximity indicator, not as a replacement for LIDAR or radar.

To improve accuracy, perform a manual calibration: place an object of known width at a known distance, measure its pixel width in the frame, then compute:

```
focal_length_px = (pixel_width * real_distance_m) / real_width_m
```

Update `FOCAL_LENGTH_PX` and `DEFAULT_VEHICLE_WIDTH_M` in `YOLO/config.py` accordingly.

---

## Configuration

All system parameters are defined in `YOLO/config.py`:

| Parameter | Description |
|-----------|-------------|
| `CAMERA_INDEX` | Webcam index (default: 0) |
| `CONFIDENCE_THRESHOLD` | Minimum YOLO detection confidence |
| `AREA_CAUTION_MIN`, `AREA_CAUTION_MAX` | Bounding box area thresholds for CAUTION |
| `AREA_WARNING_DIRECT` | Area threshold for immediate WARNING |
| `DELTA_CAUTION_MIN`, `DELTA_WARNING_MIN` | Frame-to-frame area growth thresholds |
| `ALERT_COOLDOWN_SECONDS` | Minimum interval between audio alerts |
| `ALERT_PERSISTENCE_SECONDS` | Duration to hold WARNING/CAUTION after detection loss |
| `FOCAL_LENGTH_PX` | Assumed focal length in pixels for distance estimation |
| `DEFAULT_VEHICLE_WIDTH_M` | Assumed vehicle width in meters for distance estimation |
| `SOUND_ENABLED`, `SOUND_PATH` | Audio alert toggle and file path |
