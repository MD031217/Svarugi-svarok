from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.detector import Detector
from app.main import create_app, decode_image
from app.schemas import Detection, PredictionResponse, YoloLabel


def image_bytes(image_format: str = "PNG") -> bytes:
    with BytesIO() as buffer:
        Image.new("RGB", (20, 10)).save(buffer, format=image_format)
        return buffer.getvalue()


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    settings = Settings(_env_file=None, yolo_model=tmp_path / "missing.pt")
    return TestClient(create_app(settings))


def test_api_works_without_model_downloads(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    classes = client.get("/api/v1/classes").json()
    assert [(item["id"], item["name"]) for item in classes] == [
        (0, "adj"),
        (1, "int"),
        (2, "geo"),
        (3, "pro"),
        (4, "non"),
    ]
    assert client.get("/openapi.json").status_code == 200
    response = client.post(
        "/api/v1/predictions", content=image_bytes(), headers={"content-type": "image/png"}
    )
    assert response.status_code == 503


@pytest.mark.parametrize(
    ("content", "media", "status"),
    [
        (b"garbage", "image/png", 400),
        (b"", "image/jpeg", 400),
        (image_bytes(), "text/plain", 415),
        (image_bytes("GIF"), "image/png", 415),
    ],
)
def test_invalid_uploads(client: TestClient, content: bytes, media: str, status: int) -> None:
    response = client.post("/api/v1/predictions", content=content, headers={"content-type": media})
    assert response.status_code == status


@pytest.mark.parametrize("limits", [{"max_upload_bytes": 5}, {"max_image_pixels": 100}])
def test_upload_limits(limits: dict[str, int]) -> None:
    client = TestClient(create_app(Settings(_env_file=None, **limits)))
    response = client.post(
        "/api/v1/predictions", content=image_bytes(), headers={"content-type": "image/png"}
    )
    assert response.status_code == 413


def test_prediction_contract(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def predict(self: Detector, image: Image.Image) -> PredictionResponse:
        label = YoloLabel.from_line("2 0.5 0.5 0.2 0.4")
        return PredictionResponse(
            model="best.pt",
            image_width=image.width,
            image_height=image.height,
            detections=[Detection(class_name="geo", confidence=0.9, label=label)],
            labels_txt=label.to_line() + "\n",
        )

    # Only the expensive model boundary is replaced; HTTP and image decoding are real.
    monkeypatch.setattr(Detector, "predict", predict)
    response = client.post(
        "/api/v1/predictions", content=image_bytes(), headers={"content-type": "image/png"}
    )
    assert response.status_code == 200
    body = response.json()
    assert (body["image_width"], body["image_height"]) == (20, 10)
    assert body["detections"][0]["class_name"] == "geo"
    assert body["labels_txt"] == "2 0.5 0.5 0.2 0.4\n"


def test_exif_orientation_applied_before_prediction() -> None:
    with BytesIO() as buffer:
        source = Image.new("RGB", (20, 10))
        exif = source.getexif()
        exif[274] = 6
        source.save(buffer, format="JPEG", exif=exif)
        with decode_image(buffer.getvalue(), 1000) as image:
            assert image.size == (10, 20)


def test_environment_overrides_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("YOLO_CONFIDENCE=0.4\nYOLO_DEVICE=cpu\n")
    monkeypatch.setenv("YOLO_CONFIDENCE", "0.75")
    assert Settings(_env_file=env_file).yolo_confidence == 0.75
