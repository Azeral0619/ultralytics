from ultralytics import YOLO
import warnings


warnings.filterwarnings("ignore")

model = YOLO(model="runs/weights/yolov11-two_stream.engine", task="mir")
model.predict("datasets/LLVIP/rgb/test")
