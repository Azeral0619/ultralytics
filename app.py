from concurrent.futures import ThreadPoolExecutor
import copy
import os
import gradio as gr
import cv2
import tempfile

import numpy as np
from ultralytics import YOLO
from torchvision.ops import nms
from ultralytics.utils.ops import nms_rotated
import torch

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
                results[0].names[int(cls[i])],  # Class Name first
                rounded_coords,  # Coordinates with one decimal
                rounded_conf,  # Confidence with one decimal
            ]
        )

    return result_data


def yolo_inference(image_rgb, image_ir, video_rgb, video_ir, model_id, conf_threshold):
    global previous_model_id, model_rgb, model_ir, task
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

        output_text_list = extract_results(0, results)

        return annotated_image[:, :, ::-1], None, output_text_list
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
        frame_index = 0
        output_text_list = []
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

            output_text_list.extend(extract_results(frame_index, results))

            out.write(annotated_frame)
            frame_index += 1

        cap_ir.release()
        cap_rgb.release()
        out.release()

        os.remove(video_path_ir)
        os.remove(video_path_rgb)

        return None, output_video_path, None


def yolo_inference_for_examples(image_rgb, image_ir, model_path, conf_threshold):
    annotated_image, _ = yolo_inference(image_rgb, image_ir, None, None, model_path, conf_threshold)
    return annotated_image


def app():
    with gr.Blocks():
        with gr.Row():
            image_rgb = gr.Image(type="pil", label="RGB图片", visible=True)
            image_ir = gr.Image(type="pil", label="红外图片", visible=True)

            video_RGB = gr.Video(label="RGB视频", visible=False)
            video_IR = gr.Video(label="红外视频", visible=False)

            with gr.Column():
                input_type = gr.Radio(
                    choices=["图片", "视频"],
                    value="图片",
                    label="输入类型",
                )
                model_id = gr.Dropdown(
                    label="模型",
                    choices=["yolo11n-obb-zhcn", "yolo11n-uav-zhcn"],
                    value="yolo11n-obb-zhcn",
                )
                conf_threshold = gr.Slider(
                    label="置信度阈值",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.25,
                )
                yolo_infer = gr.Button(value="检测物体")
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                output_image = gr.Image(type="numpy", label="标注图片", visible=True)
                output_video = gr.Video(label="标注视频", visible=False)
            with gr.Column():
                output_text = gr.DataFrame(
                    label="检测结果",
                    headers=["帧索引", "检测框(xywh(r))", "类别(cls)", "置信度(conf)"],
                    datatype=["str", "str"],
                    interactive=False,
                    height=output_image.height,
                    visible=True,
                )

        def update_visibility(input_type):
            image_rgb = gr.update(visible=True) if input_type == "图片" else gr.update(visible=False)
            image_ir = gr.update(visible=True) if input_type == "图片" else gr.update(visible=False)
            video_RGB = gr.update(visible=False) if input_type == "图片" else gr.update(visible=True)
            video_IR = gr.update(visible=False) if input_type == "图片" else gr.update(visible=True)
            output_image = gr.update(visible=True) if input_type == "图片" else gr.update(visible=False)
            output_video = gr.update(visible=False) if input_type == "图片" else gr.update(visible=True)
            output_text = gr.update(visible=False) if input_type == "图片" else gr.update(visible=True)

            return image_rgb, image_ir, video_RGB, video_IR, output_image, output_video, output_text

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image_rgb, image_ir, video_RGB, video_IR, output_image, output_video, output_text],
        )

        def run_inference(image_rgb, image_ir, video_rgb, video_ir, model_id, conf_threshold, input_type):
            if input_type == "图片":
                return yolo_inference(image_rgb, image_ir, None, None, model_id, conf_threshold)
            else:
                return yolo_inference(None, None, video_rgb, video_ir, model_id, conf_threshold)

        yolo_infer.click(
            fn=run_inference,
            inputs=[image_rgb, image_ir, video_RGB, video_IR, model_id, conf_threshold, input_type],
            outputs=[output_image, output_video, output_text],
        )
        """
        gr.Examples(
            examples=[
                [
                    "ultralytics/assets/LLVIP_RGB.jpg",
                    "ultralytics/assets/LLVIP_IR.jpg",
                    "detect",
                    0.25,
                ],
            ],
            fn=yolo_inference_for_examples,
            inputs=[
                image_rgb,
                image_ir,
                model_id,
                conf_threshold,
            ],
            outputs=[output_image],
            cache_examples="lazy",
        )
        """


gradio_app = gr.Blocks()
with gradio_app:
    with gr.Row():
        with gr.Column():
            app()
if __name__ == "__main__":
    gradio_app.launch(server_name="0.0.0.0")
