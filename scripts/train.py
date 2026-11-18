"""Train a detection model using paths relative to the project, independent of cwd."""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("configs/models/yolo26l.pt"))
    parser.add_argument("--data", type=Path, default=Path("datasets/dataset.yaml"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. cpu, mps, 0")
    parser.add_argument("--source", help="Optional image path or URL for prediction after training")
    args = parser.parse_args(argv)

    model_path = (PROJECT_ROOT / args.model).resolve()
    dataset_path = (PROJECT_ROOT / args.data).resolve()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    for path in (model_path, dataset_path):
        if not path.is_file():
            parser.error(f"File does not exist: {path}")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    if model.task != "detect":
        parser.error("The dataset uses detection labels; choose detection weights, not OBB weights")
    model.train(
        data=str(dataset_path),
        epochs=args.epochs,
        device=args.device,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name="weld26",
    )
    model.val(data=str(dataset_path))
    if args.source:
        source = (
            args.source if "://" in args.source else str((PROJECT_ROOT / args.source).resolve())
        )
        print(model(source))


if __name__ == "__main__":
    main()
