
import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Валидация модели дефектов сварки"
    )

    parser.add_argument(
        "--data", type=str, required=True
    )
    parser.add_argument(
        "--weights", type=str,
        default="runs/weld/train/weights/best.pt"
    )
    parser.add_argument(
        "--split", choices=["val", "test"],
        default="val"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640
    )

    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    weights_path = Path(args.weights).resolve()

    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)

    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(str(weights_path))

    metrics = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        device=device,
        workers=0,
        plots=True,
        project="runs/weld",
        name=f"validation_{args.split}",
        exist_ok=True,
    )

    report = {
        "split": args.split,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }

    # Сохраняем метрики отдельно для каждого класса
    per_class = {}

    for class_id in metrics.box.ap_class_index:
        class_id = int(class_id)

        p, r, ap50, ap = metrics.box.class_result(
            class_id
        )

        per_class[model.names[class_id]] = {
            "precision": float(p),
            "recall": float(r),
            "mAP50": float(ap50),
            "mAP50_95": float(ap),
        }

    report["classes"] = per_class

    output_dir = Path(metrics.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "metrics.json"

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Отчёт сохранён: {report_path}")


if __name__ == "__main__":
    main()
