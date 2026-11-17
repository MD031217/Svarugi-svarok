from pathlib import Path

from PIL import Image

from scripts.audit_dataset import audit_dataset


def test_audit_reports_objects_negatives_and_invalid_classes(tmp_path: Path) -> None:
    (tmp_path / "classes.txt").write_text("adj\ngeo\n")
    for name in ("a", "b", "c"):
        Image.new("RGB", (20, 10)).save(tmp_path / f"{name}.jpg")
    (tmp_path / "a.txt").write_text("1 0.5 0.5 0.2 0.4\n")
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "c.txt").write_text("5 0.5 0.5 0.2 0.4\n")
    (tmp_path / "orphan.txt").write_text("")
    report = audit_dataset(tmp_path)
    assert report["image_count"] == 3
    assert report["object_count"] == 1
    assert report["empty_label_count"] == 2
    assert report["orphan_labels"] == ["orphan"]
    assert report["errors"]
    assert report["duplicate_image_groups"] == [["a.jpg", "b.jpg", "c.jpg"]]
    assert report["conflicting_duplicate_labels"] == [["a.jpg", "b.jpg", "c.jpg"]]
