"""
Modul tracking kendaraan menggunakan DeepSort (deep-sort-realtime).
Membungkus (wrap) DeepSort agar kompatibel dengan output detector.py
dan menghasilkan TrackedVehicle dengan ID konsisten antar frame.

Dirancang ringan untuk CPU: tidak memuat model re-identification tambahan,
cukup menggunakan IOU-based tracking bawaan (embedder=None).
"""

from dataclasses import dataclass
from typing import List, Tuple

from deep_sort_realtime.deepsort_tracker import DeepSort

from detector import Detection


@dataclass
class TrackedVehicle:
    """Hasil tracking satu kendaraan dengan ID konsisten antar frame."""

    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_name: str
    confidence: float


class VehicleTracker:
    """
    Wrapper DeepSort untuk tracking kendaraan antar frame.

    Menerima List[Detection] dari VehicleDetector dan mengembalikan
    List[TrackedVehicle] dengan track_id yang stabil selama objek terlihat.

    Dikonfigurasi untuk CPU (embedder=None) agar ringan dan tidak
    memerlukan model re-identification tambahan. Mode IOU-only.
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
    ) -> None:
        """
        Inisialisasi VehicleTracker.

        Args:
            max_age: Jumlah frame maksimum sebuah track dipertahankan
                     tanpa deteksi baru sebelum dihapus.
            n_init:  Jumlah frame berturut-turut sebelum track dinyatakan
                     terkonfirmasi (confirmed).
            max_iou_distance: Batas jarak IoU untuk asosiasi deteksi ke track.
        """
        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            embedder=None,          # Nonaktifkan re-id: pakai IOU saja (CPU-friendly)
            half=False,             # Tidak ada GPU half-precision
        )

    def update(
        self,
        detections: List[Detection],
        frame=None,
    ) -> List[TrackedVehicle]:
        """
        Perbarui state tracker dengan deteksi terbaru.

        Mengkonversi List[Detection] ke format raw DeepSort, menjalankan
        update tracker, lalu mengembalikan hanya track yang sudah terkonfirmasi.

        Args:
            detections: Output dari VehicleDetector.detect() — List[Detection].
            frame:      Frame numpy. Meskipun embedder=None, parameter frame
                        tetap di-pass untuk kompatibilitasi, namun kita juga
                        harus menyediakan dummy embeds karena ini mode IOU-only.

        Returns:
            List[TrackedVehicle] dengan track_id stabil. Hanya track yang
            sudah terkonfirmasi (n_init frame berturut-turut) yang dikembalikan.
        """
        raw: List[Tuple] = self._to_raw_detections(detections)
        # Sediakan dummy embeds (list berisi None) agar tidak error 'Embedder not created'
        dummy_embeds = [None] * len(raw)
        tracks = self._tracker.update_tracks(raw, embeds=dummy_embeds, frame=frame)

        results: List[TrackedVehicle] = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = (
                int(ltrb[0]),
                int(ltrb[1]),
                int(ltrb[2]),
                int(ltrb[3]),
            )
            
            # Validasi bbox: pastikan koordinat membentuk kotak yang valid
            if x2 <= x1 or y2 <= y1:
                continue

            results.append(
                TrackedVehicle(
                    track_id=int(track.track_id),
                    bbox=(x1, y1, x2, y2),
                    class_name=track.det_class or "vehicle",
                    confidence=float(track.det_conf) if track.det_conf is not None else 0.0,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Helper private
    # ------------------------------------------------------------------

    @staticmethod
    def _to_raw_detections(
        detections: List[Detection],
    ) -> List[Tuple]:
        """
        Konversi List[Detection] ke format yang diharapkan DeepSort.

        Format raw DeepSort: ([left, top, width, height], confidence, class_name)
        Sumber bbox Detection: (x1, y1, x2, y2) → konversi ke (x1, y1, w, h).

        Args:
            detections: List[Detection] dari VehicleDetector.

        Returns:
            List of tuples dalam format DeepSort raw detections.
        """
        raw = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            raw.append(([x1, y1, w, h], det.confidence, det.class_name))
        return raw
