from ultralytics import YOLO
import os

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, "dataset")

model = YOLO("yolo11n-cls.pt")

if __name__ == "__main__":
    model.train(data= data_path, epochs=20, imgsz=224, batch=32, device = 0, workers = 0, project="runs/classify", name="banana_v12")