"""
Logika analisis collision: area, delta area, estimasi jarak,
dan status SAFE/CAUTION/WARNING berbasis track_id dari tracker.py.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import config
from tracker import TrackedVehicle


class CollisionStatus(str, Enum):
    """Status peringatan tabrakan."""
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    WARNING = "WARNING"


@dataclass
class VehicleState:
    """State satu kendaraan per frame, termasuk estimasi jarak monocular."""

    track_id: int
    tracked: TrackedVehicle
    area: float
    delta_area: float = 0.0
    status: CollisionStatus = CollisionStatus.SAFE
    distance_m: float = 0.0


def compute_area(bbox: Tuple[int, int, int, int]) -> float:
    """
    Menghitung area bounding box (width × height).
    bbox: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return float(width * height)


def compute_delta_area(area_now: float, area_previous: Optional[float]) -> float:
    """
    Delta area = area_now - area_previous.
    Jika tidak ada area_previous, return 0.0.
    """
    if area_previous is None:
        return 0.0
    return area_now - area_previous


def estimate_distance(
    bbox_width_px: float,
    focal_length_px: float,
    real_width_m: float = 1.8,
) -> float:
    """
    Estimasi jarak kendaraan dalam meter menggunakan pinhole camera model.

    Rumus: distance = (real_width_m * focal_length_px) / bbox_width_px

    Ini adalah estimasi monocular berbasis asumsi lebar kendaraan konstan
    (default 1.8 m untuk sedan/kendaraan rata-rata). BUKAN pengukuran presisi —
    akurasi bergantung pada kalibrasi focal_length_px dan variasi ukuran
    kendaraan nyata. Gunakan hanya sebagai referensi kualitatif.

    Args:
        bbox_width_px:   Lebar bounding box kendaraan dalam piksel.
        focal_length_px: Panjang fokal kamera dalam piksel (dari kalibrasi
                         atau estimasi berdasarkan FOV dan resolusi).
        real_width_m:    Asumsi lebar fisik kendaraan dalam meter (default 1.8 m).

    Returns:
        Estimasi jarak dalam meter. Mengembalikan 0.0 jika bbox_width_px <= 0.
    """
    if bbox_width_px <= 0:
        return 0.0
    return (real_width_m * focal_length_px) / bbox_width_px


def get_status(area: float, delta_area: float) -> CollisionStatus:
    """
    Menentukan status berdasarkan area dan delta area.
    - SAFE: area kecil
    - CAUTION: area sedang atau delta sedang
    - WARNING: area sangat besar (objek sangat dekat) ATAU area besar + delta cepat
    """
    # Area sangat besar = objek sangat dekat → selalu WARNING
    if area >= config.AREA_WARNING_DIRECT:
        return CollisionStatus.WARNING
    if area >= config.AREA_CAUTION_MAX and delta_area >= config.DELTA_WARNING_MIN:
        return CollisionStatus.WARNING
    if area >= config.AREA_CAUTION_MIN or delta_area >= config.DELTA_CAUTION_MIN:
        return CollisionStatus.CAUTION
    return CollisionStatus.SAFE


def analyze_detections(
    tracked_vehicles: List[TrackedVehicle],
    previous_areas: Optional[Dict[int, float]] = None,
    focal_length_px: float = 700.0,
) -> Tuple[List[VehicleState], CollisionStatus]:
    """
    Menganalisis tracked vehicles: hitung area, delta (berbasis track_id),
    estimasi jarak, dan status per kendaraan.

    Delta area dihitung per track_id — bukan per indeks urutan —
    sehingga benar meski urutan deteksi berubah antar frame.

    Args:
        tracked_vehicles: Output dari VehicleTracker.update() — List[TrackedVehicle].
        previous_areas:   Dict {track_id: area} dari frame sebelumnya.
                          None atau {} jika ini frame pertama.
        focal_length_px:  Panjang fokal kamera dalam piksel, untuk estimate_distance().
                          Default 700.0 px (estimasi umum webcam 640×480 ~60° FOV).

    Returns:
        Tuple (List[VehicleState], CollisionStatus global).
        Status global adalah WARNING jika ada satu kendaraan ber-status WARNING,
        selain itu ditentukan dari kendaraan dengan area/delta terbesar.
    """
    previous_areas = previous_areas or {}
    states: List[VehicleState] = []
    max_area = 0.0
    max_delta = 0.0
    global_status = CollisionStatus.SAFE

    for tv in tracked_vehicles:
        area = compute_area(tv.bbox)
        prev_area: Optional[float] = previous_areas.get(tv.track_id)
        delta = compute_delta_area(area, prev_area)
        status = get_status(area, delta)

        x1, y1, x2, y2 = tv.bbox
        bbox_width_px = float(max(0, x2 - x1))
        dist = estimate_distance(bbox_width_px, focal_length_px)

        states.append(
            VehicleState(
                track_id=tv.track_id,
                tracked=tv,
                area=area,
                delta_area=delta,
                status=status,
                distance_m=dist,
            )
        )
        if area > max_area:
            max_area = area
        if delta > max_delta:
            max_delta = delta

    # Status global: WARNING jika ada yang WARNING, else dari max area/delta
    if states:
        for s in states:
            if s.status == CollisionStatus.WARNING:
                global_status = CollisionStatus.WARNING
                break
        else:
            global_status = get_status(max_area, max_delta)

    return states, global_status


def get_previous_areas_from_states(states: List[VehicleState]) -> Dict[int, float]:
    """
    Bangun dict {track_id: area} dari states saat ini untuk dipakai frame berikutnya.

    Menggunakan track_id sebagai key agar delta area dihitung per objek,
    bukan per urutan index — menghindari bug saat urutan deteksi berubah.
    """
    return {s.track_id: s.area for s in states}
