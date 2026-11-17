import pytest

from app.schemas import YoloLabel


def test_parse_source_label_and_round_trip() -> None:
    label = YoloLabel.from_line("0 0.422419 0.311986 0.022602 0.052879")
    assert label.class_id == 0
    assert label.x_center == pytest.approx(0.422419)
    assert label.width == pytest.approx(0.022602)
    assert YoloLabel.from_line(label.to_line()) == label


@pytest.mark.parametrize(
    "line",
    [
        "0 0.5 0.5 0.1",
        "0 0.5 0.5 0.1 0.1 0.9",
        "-1 0.5 0.5 0.1 0.1",
        "0.5 0.5 0.5 0.1 0.1",
        "0 nan 0.5 0.1 0.1",
        "0 0.5 0.5 inf 0.1",
        "0 0.5 0.5 0 0.1",
        "0 0.99 0.5 0.2 0.1",
    ],
)
def test_reject_invalid_detection_labels(line: str) -> None:
    with pytest.raises(ValueError):
        YoloLabel.from_line(line)


def test_allow_rounding_at_boundary() -> None:
    label = YoloLabel.from_line("0 0.166666 0.5 0.333333 1")
    assert label.class_id == 0
