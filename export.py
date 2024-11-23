from ultralytics import YOLO
import warnings


warnings.filterwarnings("ignore")

model = YOLO(model="models/yolov11-two_stream.yaml", task="mir").load(
    "runs/weights/yolo11n-multimodal-two-stream.pt"
)  # 初始化模型
model.export(format="engine", simplify=True)
