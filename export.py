from ultralytics import YOLO
import warnings
import os
from multiprocessing import cpu_count

cpu_num = cpu_count()
os.environ["OMP_NUM_THREADS"] = str(cpu_num)


warnings.filterwarnings("ignore")

model = YOLO(model="runs/weights/yolo11n-multimodal-two-stream.pt", task="mir")
model.export(format="engine", simplify=True)
