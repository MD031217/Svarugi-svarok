import logging
from threading import Lock
from typing import TYPE_CHECKING

from PIL import Image

from app.config import Settings
from app.schemas import Detection, PredictionResponse, YoloLabel

if TYPE_CHECKING:
    from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ModelUnavailableError(Exception):
    """Detection weights are missing or cannot be initialized."""


class Detector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: YOLO | None = None
        self._lock = Lock()

    def _load_model(self) -> "YOLO":
        path = self.settings.yolo_model
        if path.suffix != ".pt" or not path.is_file():
            raise ModelUnavailableError("Configure YOLO_MODEL with an existing local .pt file")
        from ultralytics import YOLO

        try:
            model = YOLO(str(path))
        except (OSError, RuntimeError, ValueError) as exc:
            logger.exception("Unable to load detection weights")
            raise ModelUnavailableError("Unable to load YOLO weights") from exc
        if model.task != "detect":
            raise ModelUnavailableError("YOLO_MODEL must contain detection weights")
        return model

    def predict(self, image: Image.Image) -> PredictionResponse:
        # Ultralytics predictors share mutable state. Serialize access per process.
        with self._lock:
            if self._model is None:
                self._model = self._load_model()
            result = self._model.predict(
                source=image,
                conf=self.settings.yolo_confidence,
                imgsz=self.settings.yolo_image_size,
                device=self.settings.yolo_device,
                verbose=False,
                save=False,
            )[0]
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    x, y, width, height = box.xywhn[0].tolist()
                    detections.append(
                        Detection(
                            class_name=result.names[class_id],
                            confidence=float(box.conf.item()),
                            label=YoloLabel(
                                class_id=class_id,
                                x_center=x,
                                y_center=y,
                                width=width,
                                height=height,
                            ),
                        )
                    )
        return PredictionResponse(
            model=self.settings.yolo_model.name,
            image_width=image.width,
            image_height=image.height,
            detections=detections,
            labels_txt="".join(detection.label.to_line() + "\n" for detection in detections),
        )
