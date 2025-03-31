import copy
import os

import torch

from ultralytics.utils.ops import nms_rotated
from torchvision.ops import nms

modalities = ["rgb", "ir"]


def load_model_ids(model_path):
    """
    "prefix_suffix": {
        "modality": "model_file"
        "task": "detect" or "obb"
    }
    """
    files = os.listdir(model_path)

    model_files = [f for f in files if f.endswith(".engine") or f.endswith(".pt")]

    model_ids = {}
    for model_file in model_files:
        for modality in modalities:
            if f"_{modality}" in model_file:
                prefix = model_file.split(f"_{modality}")[0]
                suffix = model_file.split(".")[-1]
                prefix_suffix = prefix + "_" + suffix
                if prefix_suffix not in model_ids:
                    model_ids[prefix_suffix] = {modality: os.path.join(model_path, model_file)}
                else:
                    model_ids[prefix_suffix][modality] = os.path.join(model_path, model_file)
    valid_model_ids = {}
    for prefix_suffix in model_ids:
        if len(model_ids[prefix_suffix]) == 2:
            valid_model_ids[prefix_suffix] = model_ids[prefix_suffix]

    for prefix_suffix in valid_model_ids:
        if "obb" in prefix_suffix.lower():
            valid_model_ids[prefix_suffix]["task"] = "obb"
        else:
            valid_model_ids[prefix_suffix]["task"] = "detect"

    return valid_model_ids


def late_fusion(results_rgb, results_ir, task="detect", iou_threshold=0.7):
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


def late_fusion_wbf(results_rgb, results_ir, task="detect", iou_threshold=0.7, conf_type="avg", weights=None):
    """
    使用 ensemble-boxes 库中的 Weighted Boxes Fusion (WBF) 方法融合 RGB 和 IR 的检测结果

    Args:
        results_rgb: RGB 图像的检测结果
        results_ir: IR 图像的检测结果
        task: 任务类型，'detect' 或 'obb'
        iou_threshold: IOU 阈值，用于确定重叠检测框
        conf_type: 置信度融合类型，'avg'(平均)或'max'(取最大值)
        weights: 各模型的权重，默认为 None (等权重)

    Returns:
        融合后的检测结果
    """
    import numpy as np
    from ensemble_boxes import weighted_boxes_fusion

    shape = results_rgb[0].orig_img.shape

    from ultralytics.utils.ops import nms_rotated

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

        # 处理空检测结果的情况
        if boxes_ir is None or boxes_rgb is None:
            if boxes_ir is None:
                fused_results = copy.deepcopy(results_rgb)
                fused_results[0].names = merged_names  # 更新names
            else:
                fused_results = copy.deepcopy(results_ir)
                # 调整类别索引
                cls_ir = boxes_ir.cls.cpu().numpy()
                cls_ir_new = [name_to_index[names_ir[int(c)]] for c in cls_ir]
                cls_ir_new = torch.tensor(cls_ir_new, device=boxes_ir.cls.device)
                boxes_ir.cls = cls_ir_new
                fused_results[0].names = merged_names
            return fused_results

        # 准备 ensemble-boxes 库需要的格式
        # WBF 期望的格式：boxes_list = [boxes_model1, boxes_model2, ...],
        # 其中每个 boxes_modelN 是格式为 [x1/w, y1/h, x2/w, y2/h] 的标准化坐标

        # 调整IR检测框的cls索引
        cls_ir = boxes_ir.cls.cpu().numpy()
        cls_ir_new = [name_to_index[names_ir[int(c)]] for c in cls_ir]

        # 获取图像尺寸，用于归一化
        height, width = shape[:2]

        # 准备 RGB 模型的数据
        boxes_rgb_xyxy = boxes_rgb.xyxy.cpu().numpy()
        scores_rgb = boxes_rgb.conf.cpu().numpy()
        classes_rgb = boxes_rgb.cls.cpu().numpy()

        # 准备 IR 模型的数据
        boxes_ir_xyxy = boxes_ir.xyxy.cpu().numpy()
        scores_ir = boxes_ir.conf.cpu().numpy()
        classes_ir = np.array(cls_ir_new)

        # 归一化坐标
        boxes_rgb_norm = boxes_rgb_xyxy.copy()
        boxes_rgb_norm[:, 0] /= width
        boxes_rgb_norm[:, 2] /= width
        boxes_rgb_norm[:, 1] /= height
        boxes_rgb_norm[:, 3] /= height

        boxes_ir_norm = boxes_ir_xyxy.copy()
        boxes_ir_norm[:, 0] /= width
        boxes_ir_norm[:, 2] /= width
        boxes_ir_norm[:, 1] /= height
        boxes_ir_norm[:, 3] /= height

        # 构建模型列表
        boxes_list = [boxes_rgb_norm, boxes_ir_norm]
        scores_list = [scores_rgb, scores_ir]
        labels_list = [classes_rgb, classes_ir]

        # 设置权重
        if weights is None:
            weights = [1, 1]  # 等权重

        # 执行 weighted boxes fusion
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_threshold,
            skip_box_thr=0.0,
            conf_type=conf_type,
        )

        # 反归一化坐标
        fused_boxes[:, 0] *= width
        fused_boxes[:, 2] *= width
        fused_boxes[:, 1] *= height
        fused_boxes[:, 3] *= height

        # 转换为 torch 张量
        fused_boxes = torch.tensor(fused_boxes, device=boxes_rgb.xyxy.device)
        fused_scores = torch.tensor(fused_scores, device=boxes_rgb.conf.device)
        fused_classes = torch.tensor(fused_labels, device=boxes_rgb.cls.device)

        # 创建新的 Boxes 对象
        from ultralytics.engine.results import Boxes

        fused_boxes_obj = Boxes(
            boxes=torch.hstack((fused_boxes, fused_scores.unsqueeze(1), fused_classes.unsqueeze(1))),
            orig_shape=shape,
        )

        # 创建新的 Results 对象
        fused_results = copy.deepcopy(results_rgb)
        fused_results[0].boxes = fused_boxes_obj
        fused_results[0].names = merged_names  # 更新 names

        return fused_results

    else:  # 旋转边界框 (OBB)
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

        # 处理空检测结果的情况
        if obb_ir is None or obb_rgb is None:
            if obb_ir is None:
                fused_results = copy.deepcopy(results_rgb)
                fused_results[0].names = merged_names  # 更新names
            else:
                fused_results = copy.deepcopy(results_ir)
                # 调整类别索引
                cls_ir = obb_ir.cls.cpu().numpy()
                cls_ir_new = [name_to_index[names_ir[int(c)]] for c in cls_ir]
                cls_ir_new = torch.tensor(cls_ir_new, device=obb_ir.cls.device)
                obb_ir.cls = cls_ir_new
                fused_results[0].names = merged_names
            return fused_results

        # 如果有旋转框的WBF实现
        # 如果没有旋转框的WBF实现，退回到使用NMS
        # 调整IR检测框的cls索引
        cls_ir = obb_ir.cls.cpu().numpy()
        cls_ir_new = [name_to_index[names_ir[int(c)]] for c in cls_ir]
        cls_ir_new = torch.tensor(cls_ir_new, device=obb_ir.cls.device)

        # 合并检测框、置信度和类别
        obb_combined = torch.cat([obb_rgb.xywhr, obb_ir.xywhr], dim=0)
        scores_combined = torch.cat([obb_rgb.conf, obb_ir.conf], dim=0)
        classes_combined = torch.cat([obb_rgb.cls, cls_ir_new], dim=0)

        # 使用NMS去除重复检测框
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
