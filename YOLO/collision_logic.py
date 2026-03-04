"""
Logika analisis collision: area, delta area, dan status SAFE/CAUTION/WARNING.
"""

from typing import List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

import config
from detector import Detection


class CollisionStatus(str, Enum):
    """Status peringatan tabrakan."""
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    WARNING = "WARNING"


@dataclass
class VehicleState:
    """State satu kendaraan (area saat ini + area sebelumnya untuk delta)."""
    detection: Detection
    area: float
    delta_area: float = 0.0
    status: CollisionStatus = CollisionStatus.SAFE


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


def get_status(area: float, delta_area: float) -> CollisionStatus:
    """
    Menentukan status berdasarkan area dan delta area.
    - SAFE: area kecil
    - CAUTION: area sedang atau delta sedang
    - WARNING: area sangat besar (objek sangat dekat) ATAU area besar + delta cepat
    """
    # Area sangat besar = objek sangat dekat → selalu WARNING
    if hasattr(config, "AREA_WARNING_DIRECT") and area >= config.AREA_WARNING_DIRECT:
        return CollisionStatus.WARNING
    if area >= config.AREA_CAUTION_MAX and delta_area >= config.DELTA_WARNING_MIN:
        return CollisionStatus.WARNING
    if area >= config.AREA_CAUTION_MIN or delta_area >= config.DELTA_CAUTION_MIN:
        return CollisionStatus.CAUTION
    return CollisionStatus.SAFE


def analyze_detections(
    detections: List[Detection],
    previous_areas: Optional[List[float]] = None,
) -> Tuple[List[VehicleState], CollisionStatus]:
    """
    Menganalisis deteksi: hitung area, delta, dan status per kendaraan.
    Mengembalikan (list VehicleState, status global).
    previous_areas: list area dari frame sebelumnya (urut per indeks deteksi).
    """
    previous_areas = previous_areas or []
    states: List[VehicleState] = []
    max_area = 0.0
    max_delta = 0.0
    global_status = CollisionStatus.SAFE

    for i, det in enumerate(detections):
        area = compute_area(det.bbox)
        prev_area = previous_areas[i] if i < len(previous_areas) else None
        delta = compute_delta_area(area, prev_area)
        status = get_status(area, delta)

        states.append(
            VehicleState(
                detection=det,
                area=area,
                delta_area=delta,
                status=status,
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


def get_previous_areas_from_states(states: List[VehicleState]) -> List[float]:
    """Daftar area dari states saat ini (untuk frame berikutnya)."""
    return [s.area for s in states]
