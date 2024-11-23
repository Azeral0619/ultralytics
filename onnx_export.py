import torch
import onnx
import onnxruntime
import numpy as np
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.datasets import letterbox
from ultralytics.utils.checks import check_imgsz

import warnings
import os
import datetime
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

device = select_device("0")
model_path = "runs/weights/yolo11n-multimodal-two-stream.pt"
onnx_path = "runs/weights/yolo11n-multimodal-two-stream.onnx"
model = attempt_load(model_path, map_location=device)
model.eval()
names = model.names
stride = max(int(model.stride.max()), 32)
imgsz = check_imgsz(640, stride=stride, min_dim=2)
# 创建两个示例输入
rgb_path = "datasets/LLVIP/ir/test/010001.jpg"
ir_path = "datasets/LLVIP/rgb/test/010001.jpg"
img_rgb = cv2.imread(rgb_path)
img_ir = cv2.imread(ir_path)


def process_image(image, imgsz, stride, device):
    img = letterbox(image, imgsz, stride=stride, auto=False)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    im = img.float()  # uint8 to fp16/32
    im /= 255.0
    im = im[None]
    return im


dummy_input1 = process_image(img_rgb, imgsz, stride, device)
dummy_input2 = process_image(img_ir, imgsz, stride, device)


torch_out = model(dummy_input1, dummy_input2)
print(torch_out[0].shape)  # , torch_out)

# export the model
# using torch:
torch.onnx.export(
    model,  # model being run
    (dummy_input1, dummy_input2),  # model input (or a tuple for multiple inputs)
    onnx_path,  # where to save the model (can be a file or file-like object)
    do_constant_folding=True,
    input_names=["input_rgb", "input_ir"],
    output_names=["output"],
    verbose=False,
)

# model.to_onnx(onnx_file_name, x, export_params=True) #事实证明并不好用，误差很大，在检查那块。推荐使用pytorch的版本

# check onnx model
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model, full_check=True)

import onnxslim

onnx_model = onnxslim.slim(onnx_model)

metadata = {
    "author": "Ultralytics",
    "license": "AGPL-3.0 License (https://ultralytics.com/license)",
    "docs": "https://docs.ultralytics.com",
    "stride": int(max(model.stride)),
    "imgsz": imgsz,
    "names": model.names,
}
# Metadata
for k, v in metadata.items():
    meta = onnx_model.metadata_props.add()
    meta.key, meta.value = k, str(v)

onnx.save(onnx_model, onnx_path)

ort_session = onnxruntime.InferenceSession(onnx_path)

# compute ONNX Runtime output prediction
ort_inputs = {
    ort_session.get_inputs()[0].name: dummy_input1.cpu().numpy(),
    ort_session.get_inputs()[1].name: dummy_input2.cpu().numpy(),
}
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs[0].shape)  # , ort_outs)

# compute ONNX Runtime and Pytorch results
# assert_allclose: Raises an AssertionError if two objects are not equal up to desired tolerance.
np.testing.assert_allclose(torch_out[0].cpu().numpy(), ort_outs[0], rtol=1e-02, atol=1e-05)
