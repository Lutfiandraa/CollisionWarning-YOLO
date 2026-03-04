"""
Modul alert: gambar bounding box, teks status, warning visual, dan optional sound.
"""

import os
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
from collision_logic import CollisionStatus, VehicleState


def get_color_for_status(status: CollisionStatus) -> Tuple[int, int, int]:
    """Warna BGR berdasarkan status."""
    if status == CollisionStatus.WARNING:
        return config.COLOR_WARNING
    if status == CollisionStatus.CAUTION:
        return config.COLOR_CAUTION
    return config.COLOR_SAFE


def draw_bounding_boxes(
    frame: np.ndarray,
    states: List[VehicleState],
    thickness: int = 2,
) -> np.ndarray:
    """
    Menggambar bounding box di frame. Warna sesuai status (WARNING = merah).
    """
    for state in states:
        x1, y1, x2, y2 = state.detection.bbox
        color = get_color_for_status(state.status)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f"{state.detection.class_name} {state.detection.confidence:.1%}"
        cv2.putText(
            frame,
            label,
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame


def draw_status_text(
    frame: np.ndarray,
    status: CollisionStatus,
    position: Tuple[int, int] = (10, 30),
    no_vehicle_detected: bool = False,
) -> np.ndarray:
    """Menampilkan teks status (SAFE/CAUTION/WARNING) di frame."""
    color = get_color_for_status(status)
    text = f"Status: {status.value}"
    if no_vehicle_detected and status == CollisionStatus.SAFE:
        text += " (no vehicle)"
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame


def draw_warning_banner(
    frame: np.ndarray,
    text: str = "WARNING: Possible Collision!",
) -> np.ndarray:
    """Menampilkan banner peringatan merah di atas frame."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (w - tw) // 2
    y = 30
    # Background strip
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (x, y + th),
        font,
        font_scale,
        config.COLOR_WARNING,
        thickness,
        cv2.LINE_AA,
    )
    return frame


def draw_fps(frame: np.ndarray, fps: float, position: Tuple[int, int] = (10, 60)) -> np.ndarray:
    """Menampilkan FPS counter."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


def _play_sound_non_blocking(path: str) -> None:
    """Memutar sound di thread terpisah agar tidak blocking."""
    try:
        import playsound
        playsound.playsound(path, block=False)
    except Exception:
        try:
            import sys
            import winsound
            if sys.platform == "win32" and Path(path).is_file():
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                winsound.Beep(1000, 200)  # Fallback: beep pendek
        except Exception:
            pass


def play_alert_sound() -> None:
    """
    Memutar file alert.wav secara non-blocking.
    Jika file tidak ada atau playsound error, diabaikan.
    """
    if not config.SOUND_ENABLED:
        return
    path = config.SOUND_PATH
    # Support path relatif dari root proyek
    root = Path(__file__).resolve().parent.parent
    full_path = (root / path).resolve()
    if not full_path.is_file():
        return
    t = threading.Thread(target=_play_sound_non_blocking, args=(str(full_path),))
    t.daemon = True
    t.start()


def apply_alert_visuals(
    frame: np.ndarray,
    states: List[VehicleState],
    status: CollisionStatus,
    fps: float,
    show_warning_banner: bool,
) -> np.ndarray:
    """
    Menggabungkan semua visual: bbox, status, FPS, dan optional warning banner.
    """
    frame = draw_bounding_boxes(frame, states)
    frame = draw_status_text(frame, status, no_vehicle_detected=(len(states) == 0))
    frame = draw_fps(frame, fps)
    if show_warning_banner:
        frame = draw_warning_banner(frame)
    return frame
