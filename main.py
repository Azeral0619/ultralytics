from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionMIRTrainer
import warnings


warnings.filterwarnings("ignore")
model = YOLO(model="models/yolov11-two_stream.yaml", task="mir").load("yolo11n.pt")  # 初始化模型
model.train(
    trainer=DetectionMIRTrainer,
    data=r"/mnt/wsl/data/datasets/LLVIP/data.yaml",
    task="mir",
    batch=-1,
    epochs=100,
    name="MIR-yolo",
    workers=4,
    amp=True,
)  # 训练

# ############## 这是val和predict的代码 ##############
# model = YOLO(r"")
# # model.val(data=r"", batch=1, save_json=True, save_txt=False)  # 验证
# # model.predict(source=r"", save=True)  #   检测
