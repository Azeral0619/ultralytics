import cv2
import numpy as np
from ultralytics import YOLO
import warnings

from ultralytics.utils.patches import imread, imwrite

"""
Result:
    boxes:
    keypoints:
    masks:
    names: dict
        {iota: 'class'}
    obb: 
    orig_img: array 
        shape (W, H, C)
    orig_shape: tuple 
        (W, H)
    path: str
        source img path
    probs:
    save_dir:
    
"""

warnings.filterwarnings("ignore")
# model = YOLO(model="runs/weights/yolo11n-multimodal-two-stream.pt", task="mir")
model = YOLO(model="runs/weights/yolo11n-multimodal-two-stream.engine", task="mir")

# result = model.predict("datasets/LLVIP/rgb/test", save=True)
im1, im2 = imread("datasets/LLVIP/ir/test/010001.jpg"), imread("datasets/LLVIP/rgb/test/010001.jpg")
im = cv2.merge((im1, im2))
result = model(im)
print(vars(result[0][0]))
plotted_img0 = result[0][0].plot(
    conf=0.3, im_gpu=result[0][0].orig_img[0] if len(result[0][0].orig_img.shape) != 3 else result[0][0].orig_img
)
plotted_img1 = result[1][0].plot(
    conf=0.3, im_gpu=result[1][0].orig_img[0] if len(result[1][0].orig_img.shape) != 3 else result[1][0].orig_img
)
concat_img = np.hstack((plotted_img0, plotted_img1))
imwrite("example.jpg", concat_img)
