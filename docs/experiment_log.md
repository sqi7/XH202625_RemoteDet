# XH-202625 实验记录

## 总目标
赛题：基于不均衡小样本学习的光学遥感卫星陆上目标检测识别

核心指标：
- Recall >= 85%
- 虚警率 <= 20%，暂按 FP / (TP + FP) = 1 - Precision
- Precision >= 80%
- 单幅 10000x10000 图像推理时间 <= 20s

## 实验路线
1. YOLOv8m baseline
2. YOLOv8s 速度保底
3. YOLOv8m + 切片推理
4. YOLOv8m + 类别均衡 / Copy-Paste
5. YOLOv8m + 类别阈值搜索
6. YOLOv8l 或 YOLO11m 冲精度
7. RT-DETR 重新训练一版
8. 比较 YOLO 与 RT-DETR，决定是否融合

## E00 环境测试
- 日期：
- 环境：remote_det
- GPU：NVIDIA H800 PCIe
- PyTorch：2.6.0+cu124
- Ultralytics：8.4.67
- 测试模型：YOLOv8m
- 测试图片：bus.jpg
- 结果：成功

## E01 YOLOv8m baseline
- 日期：
- 数据版本：
- 模型：yolov8m.pt
- imgsz：
- batch：
- epochs：
- 训练命令：
- Precision：
- Recall：
- 虚警率：
- mAP50：
- mAP50-95：
- 推理时间：
- 备注：

## E02 YOLOv8s speed baseline
- 日期：
- 数据版本：
- 模型：yolov8s.pt
- imgsz：
- batch：
- epochs：
- 训练命令：
- Precision：
- Recall：
- 虚警率：
- mAP50：
- mAP50-95：
- 推理时间：
- 备注：
