from ultralytics import YOLO

# load a pre-trained yolov8 model
model = YOLO("yolov8n.pt")  # or yolov8s.pt, yolov8m.pt etc.

# train on your dataset
model.train(
    data="dataset_yolo/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16  # change based on your GPU/CPU RAM
)
