import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import copy
import io
import os
import tempfile

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image

from ultralytics import YOLO

from utils import late_fusion, late_fusion_wbf, load_model_ids, modalities

previous_model_id = None
models = None
model_path = r"runs/weights"
model_ids = load_model_ids(model_path)
task = "obb"


def parallel_predict(models, source_rgb, source_ir, conf_threshold):
    results_rgb = None
    results_ir = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        # 提交预测任务，返回 Future 对象
        future_rgb = executor.submit(models["rgb"].predict, source=source_rgb, conf=conf_threshold)
        future_ir = executor.submit(models["ir"].predict, source=source_ir, conf=conf_threshold)

        # 获取结果
        results_rgb = future_rgb.result()
        results_ir = future_ir.result()

    return results_rgb, results_ir


def extract_results(index, results):
    """提取检测结果，包括坐标、置信度、类别信息，并返回一个列表格式的数据"""
    result_data = []
    if task == "obb":
        boxes = results[0].obb
        xywhr = boxes.xywhr.cpu().numpy()  # Oriented bounding boxes (center x, center y, width, height, rotation)
        conf = boxes.conf.cpu().numpy()  # Confidence scores
        cls = boxes.cls.cpu().numpy()  # Class indices
    else:
        boxes = results[0].boxes
        xywhr = boxes.xywh.cpu().numpy()  # Regular bounding boxes (center x, center y, width, height)
        conf = boxes.conf.cpu().numpy()  # Confidence scores
        cls = boxes.cls.cpu().numpy()  # Class indices

    # 将检测结果转化为二维列表，每行包含一个检测框的信息
    for i in range(len(xywhr)):
        # 保留坐标和置信度为一位小数
        rounded_coords = [round(coord, 1) for coord in xywhr[i]]
        rounded_conf = round(conf[i], 1)  # 保留置信度一位小数
        result_data.append(
            [
                index,  # Frame index
                rounded_coords,  # Coordinates with one decimal
                results[0].names[int(cls[i])],  # Class Name first
                rounded_conf,  # Confidence with one decimal
            ]
        )

    return result_data


def yolo_inference(image_rgb, image_ir, video_rgb, video_ir, model_id, conf_threshold, only_detection_info=False):
    global previous_model_id, model_ids, models, task
    if not (previous_model_id is not None and previous_model_id == model_id):
        task = model_ids[model_id]["task"]
        model_info = model_ids[model_id]
        models = {f"{modality}": YOLO(f"{model_info[modality]}", task=task) for modality in modalities}
    previous_model_id = model_id
    if image_rgb and image_ir:
        if image_rgb.size[0] != image_ir.size[0] or image_rgb.size[1] != image_ir.size[1]:
            raise ValueError("尺寸不匹配")
        image_ir, image_rgb = (
            cv2.cvtColor(np.array(image_ir), cv2.COLOR_RGB2BGR),
            cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR),
        )

        results_rgb, results_ir = parallel_predict(models, image_rgb, image_ir, conf_threshold)
        results = late_fusion_wbf(results_rgb, results_ir, task=task)
        if not only_detection_info:
            annotated_image = results[0].plot()
            # origin
            annotated_image_origin = results_rgb[0].plot()

            # hstack
            annotated_image = np.hstack((annotated_image_origin, annotated_image))

        output_text_list = extract_results(0, results)
        if not only_detection_info:
            return annotated_image[:, :, ::-1], None, output_text_list
        else:
            return None, None, output_text_list
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

        if not only_detection_info:
            output_video_path = tempfile.mktemp(suffix=".mp4")
            out = cv2.VideoWriter(
                output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width * 2, frame_height)
            )
            frame_index = 0
            output_text_list = []
            while cap_ir.isOpened() and cap_rgb.isOpened():
                ret_ir, frame_ir = cap_ir.read()
                ret_rgb, frame_rgb = cap_rgb.read()
                if not ret_ir or not ret_rgb:
                    break

                results_rgb, results_ir = parallel_predict(models, frame_rgb, frame_ir, conf_threshold)
                # results = late_fusion(results_rgb, results_ir)
                results = late_fusion_wbf(results_rgb, results_ir, task=task)
                annotated_frame = results[0].plot()
                # origin
                annotated_frame_origin = results_rgb[0].plot()

                output_text_list.extend(extract_results(frame_index, results))

                # hstack
                annotated_frame = np.hstack((annotated_frame_origin, annotated_frame))

                out.write(annotated_frame)
                frame_index += 1

            cap_ir.release()
            cap_rgb.release()
            out.release()

            os.remove(video_path_ir)
            os.remove(video_path_rgb)

            return None, output_video_path, output_text_list
        else:
            frame_index = 0
            output_text_list = []
            while cap_ir.isOpened() and cap_rgb.isOpened():
                ret_ir, frame_ir = cap_ir.read()
                ret_rgb, frame_rgb = cap_rgb.read()
                if not ret_ir or not ret_rgb:
                    break

                results_rgb, results_ir = parallel_predict(models, frame_rgb, frame_ir, conf_threshold)
                results = late_fusion_wbf(results_rgb, results_ir, task=task)
                output_text_list.extend(extract_results(frame_index, results))
                frame_index += 1

            cap_ir.release()
            cap_rgb.release()

            os.remove(video_path_ir)
            os.remove(video_path_rgb)

            return None, None, output_text_list


app = FastAPI()
lock = asyncio.Lock()


@app.post("/inference/image")
async def inference_image(
    image_rgb: bytes = None, image_ir: bytes = None, model_id: str = None, conf_threshold: float = 0.3
):
    async with lock:
        try:
            image_rgb_data = await image_rgb.read()
            image_ir_data = await image_ir.read()
            image_rgb = Image.open(io.BytesIO(image_rgb_data))
            image_ir = Image.open(io.BytesIO(image_ir_data))
            annotated_image, _, detection_info = yolo_inference(
                image_rgb, image_ir, None, None, model_id, conf_threshold
            )
            _, img_encoded = cv2.imencode(".jpg", annotated_image)
            img_base64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")
            # 返回处理后的图片和检测信息
            return {"image": img_base64, "detection_info": detection_info}
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
        try:
            # 进行推理
            _, output_video_path, detection_info = yolo_inference(
                None, None, video_rgb, video_ir, model_id, conf_threshold
            )
            if output_video_path:
                # 将视频文件编码为base64
                with open(output_video_path, "rb") as f:
                    video_bytes = f.read()
                video_base64 = base64.b64encode(video_bytes).decode("utf-8")

                # 返回视频文件和检测信息
                return {
                    "video": video_base64,  # base64编码的视频
                    "detection_info": detection_info,  # 检测信息
                }
            else:
                # 如果没有生成视频，只返回检测信息
                return {"detection_info": detection_info}
        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
