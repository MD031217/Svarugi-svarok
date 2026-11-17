from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class DatasetClass(BaseModel):
    id: int
    name: str
    description: str


class YoloLabel(BaseModel):
    """One detection label: normalized center and size, not pixel corner coordinates."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    class_id: int = Field(ge=0)
    x_center: float = Field(ge=0, le=1)
    y_center: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def box_inside_image(self) -> Self:
        # Six-decimal YOLO text can introduce sub-pixel rounding at the image edge.
        tolerance = 1e-6
        for center, size in ((self.x_center, self.width), (self.y_center, self.height)):
            if center - size / 2 < -tolerance or center + size / 2 > 1 + tolerance:
                raise ValueError("Bounding box extends beyond the image")
        return self

    @classmethod
    def from_line(cls, line: str) -> Self:
        fields = line.split()
        if len(fields) != 5:
            raise ValueError("Expected exactly 5 fields: class x_center y_center width height")
        return cls(
            class_id=int(fields[0]),
            x_center=float(fields[1]),
            y_center=float(fields[2]),
            width=float(fields[3]),
            height=float(fields[4]),
        )

    def to_line(self) -> str:
        coordinates = (self.x_center, self.y_center, self.width, self.height)
        return f"{self.class_id} " + " ".join(f"{value:.9g}" for value in coordinates)


class Detection(BaseModel):
    class_name: str
    confidence: float = Field(ge=0, le=1)
    label: YoloLabel


class PredictionResponse(BaseModel):
    model: str
    image_width: int
    image_height: int
    detections: list[Detection]
    labels_txt: str = Field(description="YOLO detection labels, 5 columns, no confidence column")
