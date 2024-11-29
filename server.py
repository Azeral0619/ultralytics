import asyncio
from concurrent.futures import ThreadPoolExecutor
import copy
import io
import os
import tempfile

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image
from torchvision.ops import nms

from ultralytics import YOLO
from ultralytics.utils.ops import nms_rotated

previous_model_id = None
model_rgb = None
model_ir = None
model_path = r"runs/weights"
task = "obb"


def late_fusion(results_rgb, results_ir, iou_threshold=0.7):
    shape = results_rgb[0].orig_img.shape
    if task != "obb":
        # 获取RGB和IR的检测结果
        boxes_rgb = results_rgb[0].boxes
        boxes_ir = results_ir[0].boxes

        # 获取RGB和IR的class names
        names_rgb = results_rgb[0].names
        names_ir = results_ir[0].names

        # 合并class names并创建映射
        merged_names = names_rgb
        name_to_index = {v: k for k, v in names_rgb.items()}
        index = len(name_to_index)

        # 处理RGB的names
        # for idx, name in names_rgb.items():
        #     if name not in name_to_index:
        #         name_to_index[name] = index
        #         merged_names[index] = name
        #         index += 1

        # 处理IR的names
        for idx, name in names_ir.items():
            if name not in name_to_index:
                name_to_index[name] = index
                merged_names[index] = name
                index += 1

        # 调整RGB检测框的cls索引
        # cls_rgb = boxes_rgb.cls.cpu().numpy()
        # cls_rgb_new = [name_to_index[names_rgb[int(c)]] for c in cls_rgb]
        # cls_rgb_new = torch.tensor(cls_rgb_new, device=boxes_rgb.cls.device)
        # boxes_rgb.cls = cls_rgb_new
        if boxes_ir is None or boxes_rgb is None:
            if boxes_ir is None:
                fused_results = copy.deepcopy(results_rgb)
                fused_results[0].names = merged_names  # 更新names
            return fused_results

        # 调整IR检测框的cls索引
        cls_ir = boxes_ir.cls.cpu().numpy()
        cls_ir_new = [name_to_index[names_ir[int(c)]] for c in cls_ir]
        cls_ir_new = torch.tensor(cls_ir_new, device=boxes_ir.cls.device)
        # boxes_ir.cls = cls_ir_new

        # 合并检测框、置信度和类别
        boxes_combined = torch.cat([boxes_rgb.xyxy, boxes_ir.xyxy], dim=0)
        scores_combined = torch.cat([boxes_rgb.conf, boxes_ir.conf], dim=0)
        classes_combined = torch.cat([boxes_rgb.cls, cls_ir_new], dim=0)

        # 使用NMS去除重复检测框
        indices = nms(boxes_combined, scores_combined, iou_threshold=iou_threshold)
        fused_boxes = boxes_combined[indices]
        fused_scores = scores_combined[indices]
        fused_classes = classes_combined[indices]

        # print(fused_boxes.shape, fused_scores.shape, fused_classes.shape)

        # 创建新的Boxes对象
        from ultralytics.engine.results import Boxes

        fused_boxes_obj = Boxes(
            boxes=torch.hstack((fused_boxes, fused_scores.unsqueeze(1), fused_classes.unsqueeze(1))),
            orig_shape=shape,
        )

        # 创建新的Results对象
        fused_results = copy.deepcopy(results_rgb)
        fused_results[0].boxes = fused_boxes_obj
        fused_results[0].names = merged_names  # 更新names

        return fused_results
    else:
        # 获取RGB和IR的检测结果
        obb_rgb = results_rgb[0].obb
        obb_ir = results_ir[0].obb

        # 获取RGB和IR的class names
        names_rgb = results_rgb[0].names
        names_ir = results_ir[0].names

        # 合并class names并创建映射
        merged_names = names_rgb
        name_to_index = {v: k for k, v in names_rgb.items()}
        index = len(name_to_index)

        # 处理RGB的names
        # for idx, name in names_rgb.items():
        #     if name not in name_to_index:
        #         name_to_index[name] = index
        #         merged_names[index] = name
        #         index += 1

        # 处理IR的names
        for idx, name in names_ir.items():
            if name not in name_to_index:
                name_to_index[name] = index
                merged_names[index] = name
                index += 1

        # 调整RGB检测框的cls索引
        # cls_rgb = obb_rgb.cls.cpu().numpy()
        # cls_rgb_new = [name_to_index[names_rgb[int(c)]] for c in cls_rgb]
        # cls_rgb_new = torch.tensor(cls_rgb_new, device=obb_rgb.cls.device)
        # obb_rgb.cls = cls_rgb_new
        if obb_ir is None or obb_rgb is None:
            if obb_ir is None:
                fused_results = copy.deepcopy(results_rgb)
                fused_results[0].names = merged_names  # 更新names
            return fused_results

        # 调整IR检测框的cls索引
        cls_ir = obb_ir.cls.cpu().numpy()
        cls_ir_new = [name_to_index[names_ir[int(c)]] for c in cls_ir]
        cls_ir_new = torch.tensor(cls_ir_new, device=obb_ir.cls.device)
        # obb_ir.cls = cls_ir_new

        # 合并检测框、置信度和类别
        obb_combined = torch.cat([obb_rgb.xywhr, obb_ir.xywhr], dim=0)
        scores_combined = torch.cat([obb_rgb.conf, obb_ir.conf], dim=0)
        classes_combined = torch.cat([obb_rgb.cls, cls_ir_new], dim=0)

        # 使用NMS去除重复检测框
        indices = nms_rotated(obb_combined, scores_combined, threshold=iou_threshold)
        fused_obb = obb_combined[indices]
        fused_scores = scores_combined[indices]
        fused_classes = classes_combined[indices]
        # print(fused_obb.shape, fused_scores.shape, fused_classes.shape)

        # 创建新的Boxes对象
        from ultralytics.engine.results import OBB

        fused_obb_obj = OBB(
            boxes=torch.hstack((fused_obb, fused_scores.unsqueeze(1), fused_classes.unsqueeze(1))),
            orig_shape=shape,
        )

        # 创建新的Results对象
        fused_results = copy.deepcopy(results_rgb)
        fused_results[0].obb = fused_obb_obj
        fused_results[0].names = merged_names  # 更新names

        return fused_results


def parallel_predict(model_rgb, model_ir, source_rgb, source_ir, conf_threshold):
    results_rgb = None
    results_ir = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        # 提交预测任务，返回 Future 对象
        future_rgb = executor.submit(model_rgb.predict, source=source_rgb, conf=conf_threshold)
        future_ir = executor.submit(model_ir.predict, source=source_ir, conf=conf_threshold)

        # 获取结果
        results_rgb = future_rgb.result()
        results_ir = future_ir.result()

    return results_rgb, results_ir


def yolo_inference(image_rgb, image_ir, video_rgb, video_ir, model_id, conf_threshold):
    global previous_model_id, model_rgb, model_ir
    if not (previous_model_id is not None and previous_model_id == model_id):
        if "obb" not in model_id:
            task = "detect"
        else:
            task = "obb"
        model_rgb = YOLO(f"{model_path}/{model_id}_RGB.engine", task=task)
        model_ir = YOLO(f"{model_path}/{model_id}_IR.engine", task=task)
    previous_model_id = model_id
    if image_rgb and image_ir:
        if image_rgb.size[0] != image_ir.size[0] or image_rgb.size[1] != image_ir.size[1]:
            raise ValueError("尺寸不匹配")
        image_ir, image_rgb = (
            cv2.cvtColor(np.array(image_ir), cv2.COLOR_RGB2BGR),
            cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR),
        )

        results_rgb, results_ir = parallel_predict(model_rgb, model_ir, image_rgb, image_ir, conf_threshold)
        results = late_fusion(results_rgb, results_ir)
        annotated_image = results[0].plot()
        # origin
        annotated_image_origin = results_rgb[0].plot()

        # hstack
        annotated_image = np.hstack((annotated_image_origin, annotated_image))

        return annotated_image[:, :, ::-1], None
    elif video_rgb and video_ir:
        video_path_rgb = tempfile.mktemp(suffix=".webm")
        with open(video_path_rgb, "wb") as f:
            with open(video_rgb, "rb") as g:
                f.write(g.read())

        cap_rgb = cv2.VideoCapture(video_path_rgb)
        fps_rgb = cap_rgb.get(cv2.CAP_PROP_FPS)
        frame_width_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_path_ir = tempfile.mktemp(suffix=".webm")
        with open(video_path_ir, "wb") as f:
            with open(video_ir, "rb") as g:
                f.write(g.read())

        cap_ir = cv2.VideoCapture(video_path_ir)
        fps_ir = cap_ir.get(cv2.CAP_PROP_FPS)
        frame_width_ir = int(cap_ir.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height_ir = int(cap_ir.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps_rgb != fps_ir or frame_width_rgb != frame_width_ir or frame_height_rgb != frame_height_ir:
            cap_rgb.release()
            cap_ir.release()
            os.remove(video_path_ir)
            os.remove(video_path_rgb)
            raise ValueError("视频的帧率、宽度或高度不匹配")

        fps, frame_width, frame_height = fps_ir, frame_width_ir, frame_height_ir

        output_video_path = tempfile.mktemp(suffix=".mp4")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width * 2, frame_height))

        while cap_ir.isOpened() and cap_rgb.isOpened():
            ret_ir, frame_ir = cap_ir.read()
            ret_rgb, frame_rgb = cap_rgb.read()
            if not ret_ir or not ret_rgb:
                break

            results_rgb, results_ir = parallel_predict(model_rgb, model_ir, frame_rgb, frame_ir, conf_threshold)
            results = late_fusion(results_rgb, results_ir)
            annotated_frame = results[0].plot()
            # origin
            annotated_frame_origin = results_rgb[0].plot()

            # hstack
            annotated_frame = np.hstack((annotated_frame_origin, annotated_image))

            out.write(annotated_frame)
        cap_ir.release()
        cap_rgb.release()
        out.release()

        os.remove(video_path_ir)
        os.remove(video_path_rgb)

        return None, output_video_path


app = FastAPI()
lock = asyncio.Lock()


@app.post("/inference/image")
async def inference_image(
    image_rgb: bytes = None, image_ir: bytes = None, model_id: str = "yolo11n-obb-zhcn", conf_threshold: float = 0.3
):
    async with lock:
        try:
            image_rgb_data = await image_rgb.read()
            image_ir_data = await image_ir.read()
            image_rgb = Image.open(io.BytesIO(image_rgb_data))
            image_ir = Image.open(io.BytesIO(image_ir_data))
            annotated_image, _ = yolo_inference(image_rgb, image_ir, None, None, model_id, conf_threshold)
            _, img_encoded = cv2.imencode(".jpg", annotated_image)
            return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")
        except Exception as e:
            return {"error": str(e)}


@app.post("/inference/video")
async def inference_video(
    video_rgb: UploadFile = File(...),
    video_ir: UploadFile = File(...),
    model_id: str = "yolo11n-obb",
    conf_threshold: float = 0.3,
):
    async with lock:
        # 进行推理
        _, output_video_path = yolo_inference(None, None, video_rgb, video_ir, model_id, conf_threshold)

        # 返回处理后的视频文件
        return FileResponse(output_video_path, media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
