
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Обучение YOLO для детекции дефектов сварки"
    )

    parser.add_argument(
        "--data", type=str, required=True,
        help="Путь к data.yaml"
    )
    parser.add_argument(
        "--model", type=str, default="yolo26n.pt"
    )
    parser.add_argument(
        "--epochs", type=int, default=100
    )
    parser.add_argument(
        "--batch", type=int, default=8
    )
    parser.add_argument(
        "--imgsz", type=int, default=640
    )

    args = parser.parse_args()

    data_path = Path(args.data).resolve()

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Не найден data.yaml: {data_path}"
        )

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")

    model = YOLO(args.model)

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        patience=20,
        seed=42,
        workers=0,
        plots=True,
        project="runs/weld",
        name="train",
        exist_ok=True,
    )

    print("Обучение завершено")
    print("Лучшая модель:", model.trainer.best)


if __name__ == "__main__":
    main()
