"""
Main entry point: Forward Collision Warning (FCW) real-time dengan webcam.
Dashcam FCW berbasis YOLOv8n dan OpenCV - untuk penelitian/skripsi.
"""

import sys
import time
from pathlib import Path
from typing import Dict

import cv2

# Pastikan parent folder ada di path untuk import
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from detector import VehicleDetector
from tracker import VehicleTracker
from collision_logic import analyze_detections, get_previous_areas_from_states, CollisionStatus
from alert import apply_alert_visuals, play_alert_sound


def open_camera() -> cv2.VideoCapture:
    """Buka webcam dengan error handling."""
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Tidak dapat membuka kamera (index={config.CAMERA_INDEX}). "
            "Pastikan webcam terhubung dan tidak dipakai aplikasi lain."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap


def run() -> None:
    """Loop utama: capture -> detect -> track -> collision logic -> alert -> display."""
    cap = open_camera()
    detector = VehicleDetector()
    detector.load_model()
    tracker = VehicleTracker()  # Dibuat sekali di luar loop — penting untuk state tracking

    previous_areas: Dict[int, float] = {}  # {track_id: area} dari frame sebelumnya
    last_warning_time: float = 0.0
    last_non_safe_time: float = 0.0  # Terakhir kali status WARNING/CAUTION
    display_status = CollisionStatus.SAFE  # Status yang ditampilkan (dengan persistensi)
    fps_start = time.perf_counter()
    fps_frames = 0
    current_fps = 0.0

    print("Forward Collision Warning - Tekan 'q' untuk keluar.")
    print("Resolusi:", config.FRAME_WIDTH, "x", config.FRAME_HEIGHT)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Gagal membaca frame.")
                break

            # 1. Deteksi kendaraan → List[Detection]
            detections = detector.detect(frame)

            # 2. Tracking → List[TrackedVehicle] dengan track_id stabil antar frame
            tracked_vehicles = tracker.update(detections, frame=frame)

            # 3. Analisis collision berbasis track_id (delta area per objek, bukan per index)
            states, status = analyze_detections(
                tracked_vehicles,
                previous_areas,
                focal_length_px=config.FOCAL_LENGTH_PX,
            )
            previous_areas = get_previous_areas_from_states(states)

            now = time.perf_counter()

            # Persistensi: tetap tampilkan WARNING/CAUTION beberapa detik setelah
            # deteksi hilang (agar alert tidak langsung hilang saat deteksi flicker)
            if status in (CollisionStatus.WARNING, CollisionStatus.CAUTION):
                display_status = status
                last_non_safe_time = now
            elif (now - last_non_safe_time) < config.ALERT_PERSISTENCE_SECONDS:
                # Masih dalam jangka persistensi, pertahankan status terakhir
                pass
            else:
                display_status = CollisionStatus.SAFE

            # Cooldown suara 2 detik
            show_warning_banner = display_status == CollisionStatus.WARNING
            if status == CollisionStatus.WARNING and (
                now - last_warning_time >= config.ALERT_COOLDOWN_SECONDS
            ):
                last_warning_time = now
                play_alert_sound()

            # FPS
            fps_frames += 1
            if fps_frames >= 10:
                elapsed = now - fps_start
                current_fps = fps_frames / elapsed if elapsed > 0 else 0
                fps_frames = 0
                fps_start = now

            # Gambar overlay (pakai display_status agar visual tidak kedip)
            display = apply_alert_visuals(
                frame.copy(),
                states,
                display_status,
                current_fps,
                show_warning_banner=show_warning_banner,
            )

            cv2.imshow("Dashcam FCW - YOLOv8n", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
    print("Selesai.")


if __name__ == "__main__":
    try:
        run()
    except RuntimeError as e:
        print("Error:", e)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDihentikan oleh user.")
        sys.exit(0)
