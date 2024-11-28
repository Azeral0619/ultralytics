import os
import gradio as gr
import cv2
import tempfile
import threading

import numpy as np
from ultralytics import YOLO
from torchvision.ops import nms
import torch

previous_model_id = None
model_rgb = None
model_ir = None
model_path = r"runs/weights"
task = "obb"


def late_fusion(results_rgb, results_ir, iou_threshold=0.7):
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

    # 调整IR检测框的cls索引
    cls_ir = boxes_ir.cls.cpu().numpy()
    cls_ir_new = [name_to_index[names_ir[int(c)]] for c in cls_ir]
    cls_ir_new = torch.tensor(cls_ir_new, device=boxes_ir.cls.device)
    boxes_ir.cls = cls_ir_new

    # 合并检测框、置信度和类别
    boxes_combined = torch.cat([boxes_rgb.xyxy, boxes_ir.xyxy], dim=0)
    scores_combined = torch.cat([boxes_rgb.conf, boxes_ir.conf], dim=0)
    classes_combined = torch.cat([boxes_rgb.cls, boxes_ir.cls], dim=0)

    # 使用NMS去除重复检测框
    indices = nms(boxes_combined, scores_combined, iou_threshold=iou_threshold)
    fused_boxes = boxes_combined[indices]
    fused_scores = scores_combined[indices]
    fused_classes = classes_combined[indices]

    # 创建新的Boxes对象
    from ultralytics.engine.results import Boxes

    fused_boxes_obj = Boxes(
        xyxy=fused_boxes,
        conf=fused_scores,
        cls=fused_classes,
    )

    # 创建新的Results对象
    fused_results = results_rgb.copy()
    fused_results[0].boxes = fused_boxes_obj
    fused_results[0].names = merged_names  # 更新names

    return fused_results


def parallel_predict(model_rgb, model_ir, source_rgb, source_ir, conf_threshold):
    results_rgb = None
    results_ir = None

    def predict_rgb():
        nonlocal results_rgb
        results_rgb = model_rgb.predict(source=source_rgb, conf=conf_threshold)

    def predict_ir():
        nonlocal results_ir
        results_ir = model_ir.predict(source=source_ir, conf=conf_threshold)

    # 创建线程
    thread_rgb = threading.Thread(target=predict_rgb)
    thread_ir = threading.Thread(target=predict_ir)

    # 启动线程
    thread_rgb.start()
    thread_ir.start()

    # 等待线程完成
    thread_rgb.join()
    thread_ir.join()

    return results_rgb, results_ir


def yolo_inference(image_rgb, image_ir, video_rgb, video_ir, model_id, conf_threshold):
    global previous_model_id, model_rgb, model_ir
    if not (previous_model_id is not None and previous_model_id == model_id):
        model_rgb = YOLO(f"{model_path}/{model_id}_RGB.engine", task=task)
        model_ir = YOLO(f"{model_path}/{model_id}_IR.engine", task=task)
    previous_model_id = model_id
    if image_rgb and image_ir:
        image_ir, image_rgb = (
            cv2.cvtColor(np.array(image_ir), cv2.COLOR_RGB2BGR),
            cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR),
        )

        results_rgb, results_ir = parallel_predict(model_rgb, model_ir, image_rgb, image_ir, conf_threshold)
        results = late_fusion(results_rgb, results_ir)
        annotated_image = results[0].plot()
        # origin
        annotated_frame_origin = results_rgb[0].plot()

        # hstack
        annotated_frame = np.hstack((annotated_frame_origin, annotated_image))

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


def yolo_inference_for_examples(image_rgb, image_ir, model_path, conf_threshold):
    annotated_image, _ = yolo_inference(image_rgb, image_ir, None, None, model_path, conf_threshold)
    return annotated_image


def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image_rgb = gr.Image(type="pil", label="Image_RGB", visible=True)
                image_ir = gr.Image(type="pil", label="Image_IR", visible=True)

                video_RGB = gr.Video(label="Video_RGB", visible=False)
                video_IR = gr.Video(label="Video_IR", visible=False)

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)
                output_video = gr.Video(label="Annotated Video", visible=False)

        with gr.Row():
            yolo_infer = gr.Button(value="Detect Objects")
            input_type = gr.Radio(
                choices=["Image", "Video"],
                value="Image",
                label="Input Type",
            )
            model_id = gr.Dropdown(
                label="Model",
                choices=[
                    "detect",
                ],
                value="detect",
            )
            conf_threshold = gr.Slider(
                label="Confidence Threshold",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.25,
            )

        def update_visibility(input_type):
            image_rgb = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            image_ir = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            video_RGB = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)
            video_IR = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)
            output_image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            output_video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)

            return image_rgb, image_ir, video_RGB, video_IR, output_image, output_video

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image_rgb, image_ir, video_RGB, video_IR, output_image, output_video],
        )

        def run_inference(image_rgb, image_ir, video_rgb, video_ir, model_id, conf_threshold, input_type):
            if input_type == "Image":
                return yolo_inference(image_rgb, image_ir, None, None, model_id, conf_threshold)
            else:
                return yolo_inference(None, None, video_rgb, video_ir, model_id, conf_threshold)

        yolo_infer.click(
            fn=run_inference,
            inputs=[image_rgb, image_ir, video_RGB, video_IR, model_id, conf_threshold, input_type],
            outputs=[output_image, output_video],
        )

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


gradio_app = gr.Blocks()
with gradio_app:
    with gr.Row():
        with gr.Column():
            app()
if __name__ == "__main__":
    gradio_app.launch(server_name="0.0.0.0")
