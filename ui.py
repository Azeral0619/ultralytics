import copy
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO

# 替换 Tkinter 相关导入
from PyQt5 import QtWidgets, QtGui, QtCore

# 全局变量
previous_model_id = None
model_rgb = None
model_ir = None
model_path = "runs/weights"
task = "obb"


def late_fusion(results_rgb, results_ir, iou_threshold=0.7):
    global task
    shape = results_rgb[0].orig_img.shape
    if task != "obb":
        # 获取RGB和IR的检测结果
        boxes_rgb = results_rgb[0].boxes
        boxes_ir = results_ir[0].boxes

        # 获取RGB和IR的class names
        names_rgb = results_rgb[0].names
        names_ir = results_ir[0].names

        # 合并class names并创建映射
        merged_names = names_rgb.copy()
        name_to_index = {v: k for k, v in names_rgb.items()}
        index = len(name_to_index)

        # 处理IR的names
        for idx, name in names_ir.items():
            if name not in name_to_index:
                name_to_index[name] = index
                merged_names[index] = name
                index += 1

        if boxes_ir is None or boxes_rgb is None:
            if boxes_ir is None:
                fused_results = copy.deepcopy(results_rgb)
                fused_results[0].names = merged_names  # 更新names
            return fused_results

        # 调整IR检测框的cls索引
        cls_ir = boxes_ir.cls.cpu().numpy()
        cls_ir_new = [name_to_index[names_ir[int(c)]] for c in cls_ir]
        cls_ir_new = torch.tensor(cls_ir_new, device=boxes_ir.cls.device)

        # 合并检测框、置信度和类别
        boxes_combined = torch.cat([boxes_rgb.xyxy, boxes_ir.xyxy], dim=0)
        scores_combined = torch.cat([boxes_rgb.conf, boxes_ir.conf], dim=0)
        classes_combined = torch.cat([boxes_rgb.cls, cls_ir_new], dim=0)

        # 使用NMS去除重复检测框
        from torchvision.ops import nms

        indices = nms(boxes_combined, scores_combined, iou_threshold=iou_threshold)
        fused_boxes = boxes_combined[indices]
        fused_scores = scores_combined[indices]
        fused_classes = classes_combined[indices]

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
        merged_names = names_rgb.copy()
        name_to_index = {v: k for k, v in names_rgb.items()}
        index = len(name_to_index)

        # 处理IR的names
        for idx, name in names_ir.items():
            if name not in name_to_index:
                name_to_index[name] = index
                merged_names[index] = name
                index += 1

        if obb_ir is None or obb_rgb is None:
            if obb_ir is None:
                fused_results = copy.deepcopy(results_rgb)
                fused_results[0].names = merged_names  # 更新names
            return fused_results

        # 调整IR检测框的cls索引
        cls_ir = obb_ir.cls.cpu().numpy()
        cls_ir_new = [name_to_index[names_ir[int(c)]] for c in cls_ir]
        cls_ir_new = torch.tensor(cls_ir_new, device=obb_ir.cls.device)

        # 合并检测框、置信度和类别
        obb_combined = torch.cat([obb_rgb.xywhr, obb_ir.xywhr], dim=0)
        scores_combined = torch.cat([obb_rgb.conf, obb_ir.conf], dim=0)
        classes_combined = torch.cat([obb_rgb.cls, cls_ir_new], dim=0)

        # 使用NMS旋转去除重复检测框
        from ultralytics.utils.ops import nms_rotated

        indices = nms_rotated(obb_combined, scores_combined, threshold=iou_threshold)
        fused_obb = obb_combined[indices]
        fused_scores = scores_combined[indices]
        fused_classes = classes_combined[indices]

        # 创建新的OBB对象
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
                index,  # 帧索引
                rounded_coords,  # 坐标
                results[0].names[int(cls[i])],  # 类别名称
                rounded_conf,  # 置信度
            ]
        )

    return result_data


def yolo_inference(image_rgb, image_ir, video_rgb, video_ir, model_id, conf_threshold, callback=None):
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
        image_ir = cv2.cvtColor(np.array(image_ir), cv2.COLOR_RGB2BGR)
        image_rgb = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)

        results_rgb, results_ir = parallel_predict(model_rgb, model_ir, image_rgb, image_ir, conf_threshold)
        results = late_fusion(results_rgb, results_ir)
        annotated_image = results[0].plot()
        # 原始 RGB 检测结果
        annotated_image_origin = results_rgb[0].plot()

        # 水平拼接
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

        # 用于实时��示
        def process_frames():
            nonlocal frame_index
            while cap_ir.isOpened() and cap_rgb.isOpened():
                ret_ir, frame_ir = cap_ir.read()
                ret_rgb, frame_rgb = cap_rgb.read()
                if not ret_ir or not ret_rgb:
                    break

                results_rgb, results_ir = parallel_predict(model_rgb, model_ir, frame_rgb, frame_ir, conf_threshold)
                results = late_fusion(results_rgb, results_ir)
                annotated_frame = results[0].plot()
                # 原始 RGB 检测结果
                annotated_frame_origin = results_rgb[0].plot()

                curr_list = extract_results(frame_index, results)
                output_text_list.extend(curr_list)

                # 水平拼接
                annotated_frame = np.hstack((annotated_frame_origin, annotated_frame))

                out.write(annotated_frame)

                # 回调更新GUI
                if callback:
                    callback(annotated_frame[:, :, ::-1], curr_list)

                frame_index += 1

            cap_ir.release()
            cap_rgb.release()
            out.release()

            os.remove(video_path_ir)
            os.remove(video_path_rgb)

            return output_video_path, output_text_list

        processing_thread = threading.Thread(target=process_frames)
        processing_thread.start()

        # 由于实时显示由回调处理，直接返回
        return None, output_video_path, output_text_list


def yolo_inference_for_examples(image_rgb, image_ir, model_path, conf_threshold):
    annotated_image, _ = yolo_inference(image_rgb, image_ir, None, None, model_path, conf_threshold)
    return annotated_image


class Application(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.default_font = QtGui.QFont("Noto Sans CJK SC", 10)
        self.setFont(self.default_font)
        self.setWindowTitle("YOLO 检测应用")
        self.resize(1200, 800)

        # 创建中心部件和布局
        central_widget = QtWidgets.QWidget()
        central_layout = QtWidgets.QVBoxLayout(central_widget)

        # 输入区域
        input_frame = QtWidgets.QWidget()
        input_layout = QtWidgets.QGridLayout(input_frame)

        # 输入类型
        self.input_type_combo = QtWidgets.QComboBox()
        self.input_type_combo.addItems(["图片", "视频"])
        self.input_type_combo.currentTextChanged.connect(self.update_visibility)
        input_layout.addWidget(QtWidgets.QLabel("输入类型:"), 0, 0)
        input_layout.addWidget(self.input_type_combo, 0, 1)

        # 选择 RGB 文件
        self.rgb_path = ""
        self.rgb_entry = QtWidgets.QLineEdit()
        self.rgb_button = QtWidgets.QPushButton("浏览")
        self.rgb_button.clicked.connect(self.browse_rgb)
        input_layout.addWidget(QtWidgets.QLabel("RGB 文件:"), 1, 0)
        input_layout.addWidget(self.rgb_entry, 1, 1)
        input_layout.addWidget(self.rgb_button, 1, 2)

        # 选择 IR 文件
        self.ir_path = ""
        self.ir_entry = QtWidgets.QLineEdit()
        self.ir_button = QtWidgets.QPushButton("浏览")
        self.ir_button.clicked.connect(self.browse_ir)
        input_layout.addWidget(QtWidgets.QLabel("红外文件:"), 2, 0)
        input_layout.addWidget(self.ir_entry, 2, 1)
        input_layout.addWidget(self.ir_button, 2, 2)

        # 模型选择
        self.model_dropdown = QtWidgets.QComboBox()
        self.model_dropdown.addItems(["yolo11n-obb-zhcn", "yolo11n-uav-zhcn"])
        input_layout.addWidget(QtWidgets.QLabel("模型:"), 3, 0)
        input_layout.addWidget(self.model_dropdown, 3, 1)

        # 置信度阈值
        self.conf_threshold = 0.25
        self.conf_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(int(self.conf_threshold * 100))
        self.conf_slider.valueChanged.connect(self.update_conf_threshold)
        input_layout.addWidget(QtWidgets.QLabel("置信度阈值:"), 4, 0)
        input_layout.addWidget(self.conf_slider, 4, 1)

        # 检测按钮
        self.detect_button = QtWidgets.QPushButton("检测物体")
        self.detect_button.clicked.connect(self.run_inference)
        input_layout.addWidget(self.detect_button, 5, 1)

        central_layout.addWidget(input_frame)

        # 输出区域
        output_frame = QtWidgets.QWidget()
        output_layout = QtWidgets.QVBoxLayout(output_frame)

        # 输出图片或视频
        self.output_image_label = QtWidgets.QLabel("标注图片:")
        self.output_image = QtWidgets.QLabel()
        output_layout.addWidget(self.output_image_label)
        output_layout.addWidget(self.output_image)

        self.output_video_label = QtWidgets.QLabel("标注视频:")
        self.output_video = QtWidgets.QLabel()
        output_layout.addWidget(self.output_video_label)
        output_layout.addWidget(self.output_video)

        # 检测结果
        self.result_tree = QtWidgets.QTableWidget()
        self.result_tree.setColumnCount(4)
        self.result_tree.setHorizontalHeaderLabels(["帧索引", "检测框(xywh(r))", "类别(cls)", "置信度(conf)"])
        output_layout.addWidget(self.result_tree)

        # 下载视频按钮
        self.download_button = QtWidgets.QPushButton("下载合成视频")
        self.download_button.setEnabled(False)
        self.download_button.clicked.connect(self.download_video)
        output_layout.addWidget(self.download_button)

        central_layout.addWidget(output_frame)
        self.setCentralWidget(central_widget)

        self.output_video_path = None  # 保存合成视频的路径

        self.update_visibility()

    def browse_rgb(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择RGB文件", "", "Image Files (*.png *.jpg *.jpeg);;Video Files (*.mp4 *.webm)"
        )
        if file_path:
            self.rgb_path = file_path
            self.rgb_entry.setText(file_path)

    def browse_ir(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择红外文件", "", "Image Files (*.png *.jpg *.jpeg);;Video Files (*.mp4 *.webm)"
        )
        if file_path:
            self.ir_path = file_path
            self.ir_entry.setText(file_path)

    def update_visibility(self):
        input_type = self.input_type_combo.currentText()
        if input_type == "图片":
            self.output_image_label.setText("标注图片:")
            self.output_image.show()
            self.output_video.hide()
            self.output_video_label.hide()
            self.download_button.hide()
        else:
            self.output_video_label.setText("标注视频:")
            self.output_video.show()
            self.output_image.hide()
            self.output_image_label.hide()
            self.download_button.hide()

    def run_inference(self):
        input_type = self.input_type_combo.currentText()
        rgb_path = self.rgb_entry.text()
        ir_path = self.ir_entry.text()
        model_id = self.model_dropdown.currentText()
        conf_threshold = self.conf_threshold

        if input_type == "图片":
            if not rgb_path or not ir_path:
                QtWidgets.QMessageBox.critical(self, "错误", "请选择RGB和红外图片文件。")
                return
            image_rgb = Image.open(rgb_path)
            image_ir = Image.open(ir_path)
            try:
                annotated_image, _, output_text_list = yolo_inference(
                    image_rgb, image_ir, None, None, model_id, conf_threshold
                )
                # 显示图片
                image = QtGui.QImage(
                    annotated_image.data, annotated_image.shape[1], annotated_image.shape[0], QtGui.QImage.Format_RGB888
                )
                pixmap = QtGui.QPixmap.fromImage(image)
                self.output_image.setPixmap(pixmap.scaled(600, 400, QtCore.Qt.KeepAspectRatio))

                # 清空之前的检测结果
                self.result_tree.setRowCount(0)

                # 插入新的检测结果
                for result in output_text_list:
                    row_position = self.result_tree.rowCount()
                    self.result_tree.insertRow(row_position)
                    for col, item in enumerate(result):
                        self.result_tree.setItem(row_position, col, QtWidgets.QTableWidgetItem(str(item)))

                self.download_button.setEnabled(False)  # 不需要下载按钮
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "错误", str(e))
        else:
            if not rgb_path or not ir_path:
                QtWidgets.QMessageBox.critical(self, "错误", "请选择RGB和红外视频文件。")
                return
            try:
                # 禁用检测按钮以防重复点击
                self.detect_button.setEnabled(False)
                # 清空之前的检测结果
                self.result_tree.setRowCount(0)
                self.output_video.setText("处理中...")
                self.output_video.repaint()

                # 启动视频处理线程
                threading.Thread(target=self.process_video, args=(rgb_path, ir_path, model_id, conf_threshold)).start()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "错误", str(e))
                self.detect_button.setEnabled(True)

    def process_video(self, rgb_path, ir_path, model_id, conf_threshold):
        try:

            def update_frame(frame, results):
                # 转换为 QImage 格式
                image = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(image)
                # 更新GUI
                self.output_image.setPixmap(pixmap)
                # 更新结果表格
                for result in results:
                    row_position = self.result_tree.rowCount()
                    self.result_tree.insertRow(row_position)
                    for col, item in enumerate(result):
                        self.result_tree.setItem(row_position, col, QtWidgets.QTableWidgetItem(str(item)))

            _, output_video_path, output_text_list = yolo_inference(
                None, None, rgb_path, ir_path, model_id, conf_threshold, callback=update_frame
            )

            self.output_video_path = output_video_path
            self.output_video.setText(f"视频已处理并保存到: {output_video_path}")
            self.download_button.setEnabled(True)
            self.download_button.show()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", str(e))
        finally:
            self.detect_button.setEnabled(True)

    def download_video(self):
        if not self.output_video_path:
            QtWidgets.QMessageBox.critical(self, "错误", "没有可供下载的视频。")
            return
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存合成视频", "", "MP4 Files (*.mp4);;All Files (*.*)"
        )
        if save_path:
            try:
                os.rename(self.output_video_path, save_path)
                QtWidgets.QMessageBox.information(self, "成功", f"视频已保存到: {save_path}")
                self.output_video_path = None
                self.download_button.setEnabled(False)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")

    def update_conf_threshold(self, value):
        self.conf_threshold = value / 100.0


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = Application()
    window.show()
    sys.exit(app.exec_())
