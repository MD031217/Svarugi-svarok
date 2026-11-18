import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.train import PROJECT_ROOT, main


def test_training_and_dataset_paths_from_another_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ultralytics
    from ultralytics.data import utils

    from scripts import train

    project = tmp_path / "project"
    model_path = project / "configs/models/yolo26l.pt"
    model_path.parent.mkdir(parents=True)
    model_path.touch()
    for split in ("train", "val"):
        (project / "datasets/images" / split).mkdir(parents=True)
    (project / "datasets/dataset.yaml").write_text(
        "train: images/train\nval: images/val\nnames: [adj, int, geo, pro, non]\n"
    )
    monkeypatch.setattr(train, "PROJECT_ROOT", project)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    # Font retrieval is unrelated to dataset path validation and may use the network.
    monkeypatch.setattr(utils, "check_font", lambda *_: None)
    model = Mock(task="detect")
    factory = Mock(return_value=model)
    monkeypatch.setattr(ultralytics, "YOLO", factory)
    main(["--epochs", "1"])

    model_path = Path(factory.call_args.args[0])
    dataset_path = Path(model.train.call_args.kwargs["data"])
    assert model_path == project / "configs/models/yolo26l.pt"
    assert dataset_path == project / "datasets/dataset.yaml"
    data = utils.check_det_dataset(str(dataset_path), autodownload=False)
    for split in ("train", "val"):
        assert Path(data[split]) == project / "datasets/images" / split
        assert Path(data[split]).is_dir()
    assert model.val.call_args.kwargs["data"] == str(dataset_path)


def test_importing_entrypoint_does_not_start_training() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import main; assert main.app; assert 'ultralytics' not in sys.modules",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
