from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    project="runs/weld",
    name="defect_detector"
)

metrics = model.val(
    data="dataset/data.yaml"
)

print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)