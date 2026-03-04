"""
Modul deteksi objek menggunakan YOLOv8n (pretrained COCO).
Hanya mengembalikan bounding box untuk class kendaraan.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass

from ultralytics import YOLO

import config


@dataclass
class Detection:
    """Satu hasil deteksi kendaraan."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float


class VehicleDetector:
    """
    Detektor kendaraan berbasis YOLOv8n.
    Memuat model sekali dan menyaring hanya car, bus, truck, motorcycle.
    """

    def __init__(
        self,
        model_path: str = config.MODEL_PATH,
        confidence: float = config.CONFIDENCE_THRESHOLD,
        iou: float = config.IOU_THRESHOLD,
        device: str = config.DEVICE,
        vehicle_class_ids: Optional[List[int]] = None,
    ) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.vehicle_class_ids = vehicle_class_ids or config.VEHICLE_CLASS_IDS
        self._model: Optional[YOLO] = None

    def load_model(self) -> None:
        """Memuat model YOLOv8n (unduh otomatis jika belum ada)."""
        if self._model is None:
            self._model = YOLO(self.model_path)

    def detect(self, frame) -> List[Detection]:
        """
        Mendeteksi kendaraan pada frame.
        Mengembalikan list Detection (bbox, class_id, class_name, confidence).
        """
        if self._model is None:
            self.load_model()

        results = self._model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
            classes=self.vehicle_class_ids,
        )

        detections: List[Detection] = []
        if not results or len(results) == 0:
            return detections

        result = results[0]
        if result.boxes is None:
            return detections

        names = result.names or {}
        for box in result.boxes:
            xyxy = box.xyxy[0]
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = names.get(cls_id, "vehicle")
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                )
            )

        return detections
