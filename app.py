import os
import gradio as gr
import cv2
import tempfile

import numpy as np
from ultralytics import YOLO

previous_model_id = None
model = None
model_path = r"runs/weights"
task = "mir"


def yolo_inference(image_rgb, image_ir, video_rgb, video_ir, model_id, conf_threshold):
    global previous_model_id, model
    if not (previous_model_id is not None and previous_model_id == model_id):
        model = YOLO(f"{model_path}/{model_id}.engine", task="mir")
    previous_model_id = model_id
    if image_rgb and image_ir:
        image_ir, image_rgb = (
            cv2.cvtColor(np.array(image_ir), cv2.COLOR_RGB2BGR),
            cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR),
        )
        image = cv2.merge((image_ir, image_rgb))
        results = model.predict(source=image, conf=conf_threshold)
        annotated_image = np.hstack((results[0][0].plot(), results[1][0].plot()))
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
            frame = cv2.merge((frame_ir, frame_rgb))
            results = model.predict(source=frame, conf=conf_threshold)
            annotated_frame = np.hstack((results[0][0].plot(), results[1][0].plot()))
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
                    "MIR",
                ],
                value="MIR",
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
                    "MIR",
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
