"""
Konfigurasi sistem Forward Collision Warning (FCW).
Digunakan untuk penelitian/skripsi: Dashcam FCW berbasis YOLOv8n dan OpenCV.
"""

from typing import List, Tuple

# =============================================================================
# RESOLUSI & KAMERA
# =============================================================================
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480
CAMERA_INDEX: int = 0

# =============================================================================
# MODEL YOLO
# =============================================================================
MODEL_PATH: str = "yolov8n.pt"
# Confidence rendah agar tampak belakang/partial/mainan mobil tetap terdeteksi
CONFIDENCE_THRESHOLD: float = 0.25
IOU_THRESHOLD: float = 0.45
DEVICE: str = "cpu"  # Optimasi untuk laptop tanpa GPU

# =============================================================================
# CLASS KENDARAAN (COCO dataset)
# ID: 2=car, 3=motorcycle, 5=bus, 7=truck
# =============================================================================
VEHICLE_CLASS_IDS: List[int] = [2, 3, 5, 7]
VEHICLE_CLASS_NAMES: List[str] = ["car", "motorcycle", "bus", "truck"]

# =============================================================================
# THRESHOLD AREA (pixel²) - untuk ukuran bounding box di frame
# Area kecil = objek jauh, area besar = objek dekat
# Frame 640x480 = 307200 px²; mobil mengisi ~30%+ ≈ 90000+ px²
# =============================================================================
AREA_SAFE_MAX: float = 25000.0      # Di bawah ini = SAFE (objek masih jauh)
AREA_CAUTION_MIN: float = 25000.0   # Mulai waspada
AREA_CAUTION_MAX: float = 80000.0   # Batas CAUTION; di atas = sangat dekat
# Di atas AREA_CAUTION_MAX = area besar → berpotensi WARNING jika delta tinggi
AREA_WARNING_DIRECT: float = 100000.0  # Area di atas ini = WARNING (objek sangat dekat)

# =============================================================================
# THRESHOLD DELTA AREA (perubahan area antar frame)
# Delta positif besar = objek mendekat cepat
# =============================================================================
DELTA_SAFE_MAX: float = 3000.0      # Perubahan kecil = aman
DELTA_CAUTION_MIN: float = 3000.0   # Mulai waspada
DELTA_WARNING_MIN: float = 8000.0   # Peningkatan cepat = WARNING

# =============================================================================
# ALERT & COOLDOWN
# =============================================================================
ALERT_COOLDOWN_SECONDS: float = 2.0  # Jangan spam suara alert
# Tetap tampilkan WARNING/CAUTION selama N detik setelah terakhir terdeteksi
# (mencegah alert hilang saat deteksi flicker/tidak konsisten)
ALERT_PERSISTENCE_SECONDS: float = 2.5
SOUND_ENABLED: bool = True
SOUND_PATH: str = "assets/sounds/alert.wav"

# =============================================================================
# WARNA (BGR - OpenCV)
# =============================================================================
COLOR_SAFE: Tuple[int, int, int] = (0, 255, 0)    # Hijau
COLOR_CAUTION: Tuple[int, int, int] = (0, 255, 255)  # Kuning
COLOR_WARNING: Tuple[int, int, int] = (0, 0, 255)  # Merah

# =============================================================================
# FPS & STABILITY
# =============================================================================
TARGET_FPS: int = 30
MIN_DETECTIONS_FOR_DELTA: int = 1  # Minimal deteksi untuk hitung delta
