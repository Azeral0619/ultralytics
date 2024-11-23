import argparse
import json
import os
import time
import numpy as np
import cv2
import torch
from collections import OrderedDict, namedtuple
from models.experimental import attempt_load
from utils.general import check_img_size, scale_coords
from ultralytics.utils.ops import non_max_suppression
from ultralytics.utils.checks import check_requirements, check_version
from ultralytics.utils import ARM64, IS_JETSON, IS_RASPBERRYPI, LINUX, yaml_load
from utils.datasets import letterbox
from utils.torch_utils import select_device
import warnings
from pathlib import Path

import tensorrt as trt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")


def load_engine(engine_path):
    # TRT_LOGGER = trt.Logger(trt.Logger.WARNING)  # INFO
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


parser = argparse.ArgumentParser()
# 检测参数
parser.add_argument("--weights", default=r"runs/transformer_LLVIP/weights/best.pt", type=str, help="weights path")
parser.add_argument("--image_rgb", default=r"datasets/LLVIP/rgb/test", type=str, help="image_rgb")
parser.add_argument("--image_ir", default=r"datasets/LLVIP/ir/test", type=str, help="image_rgb")
parser.add_argument("--conf_thre", type=int, default=0.3, help="conf_thre")
parser.add_argument("--iou_thre", type=int, default=0.6, help="iou_thre")
parser.add_argument("--save_image", default=r"runs", type=str, help="save img or video path")
parser.add_argument("--device", type=str, default="0", help="use gpu or cpu")
parser.add_argument("--imgsz", type=int, default=640, help="image size")
parser.add_argument("--merge_nms", default=False, action="store_true", help="merge class")
parser.add_argument("--vis", default=False, action="store_true", help="visualize image")
parser.add_argument("--data", default=None, help="data info(yaml)")
opt = parser.parse_args()


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


class Detector:
    def __init__(self, device, model_path=r"./best.pt", imgsz=640, merge_nms=False):
        self.device = device
        self.dynamic = False
        self.fp16 = False
        self.cuda = True
        self.stride = 32
        if model_path.endswith("engine"):
            try:
                import tensorrt as trt  # noqa https://developer.nvidia.com/nvidia-tensorrt-download
            except ImportError:
                if LINUX:
                    check_requirements("tensorrt>7.0.0,!=10.1.0")
                import tensorrt as trt  # noqa
            check_version(trt.__version__, ">=7.0.0", hard=True)
            check_version(trt.__version__, "!=10.1.0", msg="https://github.com/ultralytics/ultralytics/pull/14239")
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.Logger(trt.Logger.INFO)
            # Read file
            with open(model_path, "rb") as f, trt.Runtime(logger) as runtime:
                try:
                    meta_len = int.from_bytes(f.read(4), byteorder="little")  # read metadata length
                    self.metadata = json.loads(f.read(meta_len).decode("utf-8"))  # read metadata
                except UnicodeDecodeError:
                    f.seek(0)  # engine file may lack embedded Ultralytics metadata
                model = runtime.deserialize_cuda_engine(f.read())  # read engine

            # Model context
            try:
                self.context = model.create_execution_context()
            except Exception as e:  # model is None
                raise e

            self.bindings = OrderedDict()
            self.output_names = []
            self.fp16 = False  # default updated below
            self.dynamic = False
            self.is_trt10 = not hasattr(model, "num_bindings")
            num = range(model.num_io_tensors) if self.is_trt10 else range(model.num_bindings)
            for i in num:
                if self.is_trt10:
                    name = model.get_tensor_name(i)
                    dtype = trt.nptype(model.get_tensor_dtype(name))
                    is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                    if is_input:
                        if -1 in tuple(model.get_tensor_shape(name)):
                            self.dynamic = True
                            self.context.set_input_shape(name, tuple(model.get_tensor_profile_shape(name, 0)[1]))
                        if dtype == np.float16:
                            self.fp16 = True
                    else:
                        self.output_names.append(name)
                    shape = tuple(self.context.get_tensor_shape(name))
                else:  # TensorRT < 10.0
                    name = model.get_binding_name(i)
                    dtype = trt.nptype(model.get_binding_dtype(i))
                    is_input = model.binding_is_input(i)
                    if model.binding_is_input(i):
                        if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                            self.dynamic = True
                            self.context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[1]))
                        if dtype == np.float16:
                            self.fp16 = True
                    else:
                        self.output_names.append(name)
                    shape = tuple(self.context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
            batch_size = self.bindings["input_rgb"].shape[0]  # if dynamic, this is instead max batch size
            if opt.data:
                data_info = yaml_load(opt.data)
                self.names = data_info["names"]
            else:
                raise "No class info"

        elif model_path.endswith("pt"):
            self.model = attempt_load(model_path, map_location=device)  # load FP32 model
            self.model.eval()
            self.names = self.model.names
            self.stride = max(int(self.model.stride.max()), 32)  # grid size (max stride)
            self.merge_nms = merge_nms

        elif model_path.endswith("onnx"):
            check_requirements(("onnx", "onnxruntime-gpu" if self.cuda else "onnxruntime"))
            if IS_RASPBERRYPI or IS_JETSON:
                # Fix 'numpy.linalg._umath_linalg' has no attribute '_ilp64' for TF SavedModel on RPi and Jetson
                check_requirements("numpy==1.23.5")
            import onnxruntime

            providers = onnxruntime.get_available_providers()
            if not self.cuda and "CUDAExecutionProvider" in providers:
                providers.remove("CUDAExecutionProvider")
            elif self.cuda and "CUDAExecutionProvider" not in providers:
                device = torch.device("cpu")
                self.cuda = False
            self.model = onnxruntime.InferenceSession(model_path, providers=providers)
            self.output_names = [x.name for x in self.model.get_outputs()]
            self.metadata = self.model.get_modelmeta().custom_metadata_map
            dynamic = isinstance(self.model.get_outputs()[0].shape[0], str)
            if not dynamic:
                self.io = self.model.io_binding()
                self.bindings = []
                for output in self.model.get_outputs():
                    y_tensor = torch.empty(output.shape, dtype=torch.float16 if self.fp16 else torch.float32).to(device)
                    self.io.bind_output(
                        name=output.name,
                        device_type=device.type,
                        device_id=device.index if self.cuda else 0,
                        element_type=np.float16 if self.fp16 else np.float32,
                        shape=tuple(y_tensor.shape),
                        buffer_ptr=y_tensor.data_ptr(),
                    )
                    self.bindings.append(y_tensor)
        self.imgsz = check_img_size(imgsz, s=self.stride)
        self.model_path = model_path

    def process_image(self, image, imgsz, stride, device):
        img = letterbox(image, imgsz, stride=stride, auto=False)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        im = img.float()  # uint8 to fp16/32
        im /= 255.0
        im = im[None]
        return im

    def from_numpy(self, x):
        """
        Convert a numpy array to a tensor.

        Args:
            x (np.ndarray): The array to be converted.

        Returns:
            (torch.Tensor): The converted tensor
        """
        return torch.tensor(x).to(self.device) if isinstance(x, np.ndarray) else x

    @torch.no_grad()
    def __call__(self, image_rgb: np.ndarray, image_ir: np.ndarray, conf, iou):
        img_vis = image_rgb.copy()
        img_vis_ir = image_ir.copy()
        img_rgb = self.process_image(image_rgb, self.imgsz, self.stride, device)
        img_ir = self.process_image(image_ir, self.imgsz, self.stride, device)

        if self.model_path.endswith("pt"):
            # inference
            pred = self.model(img_rgb, img_ir)
            print(len(pred), pred[0].shape)
            # Apply NMS
            pred = non_max_suppression(pred, conf_thres=conf, iou_thres=iou, classes=None, agnostic=self.merge_nms)
            print(len(pred), pred[0].shape)
        elif self.model_path.endswith("engine"):
            if self.dynamic:
                if self.is_trt10:
                    for name in self.output_names:
                        self.bindings[name].data.resize_(tuple(self.context.get_tensor_shape(name)))
                else:
                    for name in self.output_names:
                        i = self.model.get_binding_index(name)
                        self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))

            s = self.bindings["input_rgb"].shape
            assert (
                img_rgb.shape == s
            ), f"input size {img_rgb.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            s = self.bindings["input_ir"].shape
            assert (
                img_ir.shape == s
            ), f"input size {img_ir.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs["input_rgb"] = int(img_rgb.data_ptr())
            self.binding_addrs["input_ir"] = int(img_ir.data_ptr())

            self.context.execute_v2(list(self.binding_addrs.values()))
            pred = [self.bindings[x].data for x in sorted(self.output_names)][-1]
        elif self.model_path.endswith("onnx"):
            if self.dynamic:
                img_rgb = img_rgb.cpu().numpy()  # torch to numpy
                img_ir = img_ir.cpu().numpy()
                pred = self.model.run(
                    self.output_names,
                    {self.model.get_inputs()[0].name: img_rgb, self.model.get_inputs()[1].name: img_ir},
                )
            else:
                if not self.cuda:
                    img_rgb = img_rgb.cpu().numpy()  # torch to numpy
                    img_ir = img_ir.cpu().numpy()
                self.io.bind_input(
                    name=self.model.get_inputs()[0].name,
                    device_type=img_rgb.device.type,
                    device_id=img_rgb.device.index if img_rgb.device.type == "cuda" else 0,
                    element_type=np.float16 if self.fp16 else np.float32,
                    shape=tuple(img_rgb.shape),
                    buffer_ptr=img_rgb.data_ptr(),
                )
                self.io.bind_input(
                    name=self.model.get_inputs()[1].name,
                    device_type=img_rgb.device.type,
                    device_id=img_rgb.device.index if img_rgb.device.type == "cuda" else 0,
                    element_type=np.float16 if self.fp16 else np.float32,
                    shape=tuple(img_ir.shape),
                    buffer_ptr=img_ir.data_ptr(),
                )
                self.model.run_with_iobinding(self.io)
                pred = self.bindings[0]

        if isinstance(pred, (list, tuple)):
            pred = self.from_numpy(pred[0]) if len(pred) == 1 else [self.from_numpy(x) for x in pred]
        else:
            pred = self.from_numpy(pred)

        print(pred.shape)

        for i, det in enumerate(pred):  # detections per image
            det[:, :4] = scale_coords(img_rgb.shape[2:], det[:, :4], image_rgb.shape).round()
            for *xyxy, conf, cls in reversed(det):
                print(det.shape)
                xmin, ymin, xmax, ymax = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                cv2.rectangle(img_vis, (int(xmin), int(ymin)), (int(xmax), int(ymax)), get_color(int(cls) + 2), 2)
                cv2.putText(
                    img_vis,
                    f"{self.names[int(cls)]} {conf:.1f}",
                    (int(xmin), int(ymin - 5)),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    get_color(int(cls) + 2),
                    thickness=2,
                )
                cv2.rectangle(img_vis_ir, (int(xmin), int(ymin)), (int(xmax), int(ymax)), get_color(int(cls) + 2), 2)
                cv2.putText(
                    img_vis_ir,
                    f"{self.names[int(cls)]} {conf:.1f}",
                    (int(xmin), int(ymin - 5)),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    get_color(int(cls) + 2),
                    thickness=2,
                )
        return img_vis, img_vis_ir


if __name__ == "__main__":
    device = select_device(opt.device)
    detector = Detector(device=device, model_path=opt.weights, imgsz=opt.imgsz, merge_nms=opt.merge_nms)

    images_format = [".png", ".jpg", ".txt", ".jpeg"]
    image_names = [
        name for name in os.listdir(opt.image_rgb) for item in images_format if os.path.splitext(name)[1] == item
    ]

    for i, img_name in enumerate(image_names):
        rgb_path = os.path.join(opt.image_rgb, img_name)
        ir_path = os.path.join(opt.image_ir, img_name)
        img_rgb = cv2.imread(rgb_path)
        img_ir = cv2.imread(ir_path)
        cost = time.time()
        img_vi, img_ir = detector(img_rgb, img_ir, opt.conf_thre, opt.iou_thre)
        cost = time.time() - cost
        print(i, img_name, f"{cost:.2f} s")
        # 横向拼接img_vi和img_ir
        img_combined = cv2.hconcat([img_vi, img_ir])
        # 保存拼接后的图像
        cv2.imwrite(os.path.join(opt.save_image, img_name), img_combined)

        if opt.vis:
            cv2.imshow(img_name, img_combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
