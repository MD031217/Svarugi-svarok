from unittest.mock import Mock

import pytest
from PIL import Image

from app.config import Settings
from app.detector import Detector


@pytest.mark.parametrize("empty", [False, True])
def test_model_results_use_weight_class_names_and_normalized_boxes(
    monkeypatch: pytest.MonkeyPatch, empty: bool
) -> None:
    box = Mock()
    box.cls.item.return_value = 0
    box.conf.item.return_value = 0.8
    coordinates = Mock()
    coordinates.tolist.return_value = [0.5, 0.25, 0.2, 0.1]
    box.xywhn = [coordinates]
    result = Mock()
    result.names = {0: "person"}
    result.boxes = [] if empty else [box]
    model = Mock()
    model.predict.return_value = [result]
    monkeypatch.setattr(Detector, "_load_model", lambda self: model)

    detector = Detector(Settings(_env_file=None))
    with Image.new("RGB", (100, 200)) as image:
        response = detector.predict(image)
    assert (response.image_width, response.image_height) == (100, 200)
    if empty:
        assert response.detections == []
        assert response.labels_txt == ""
    else:
        assert response.detections[0].class_name == "person"
        assert response.detections[0].label.class_id == 0
        assert response.detections[0].confidence == pytest.approx(0.8)
        assert response.labels_txt == "0 0.5 0.25 0.2 0.1\n"
