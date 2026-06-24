XH-202625 实验记录

1. 总目标

赛题：基于不均衡小样本学习的光学遥感卫星陆上目标检测识别

核心任务：对光学遥感图像中的 25 类目标进行检测识别，目标类别包含舰船、飞机和车辆三大类。

核心指标：

* 整体 Recall ≥ 85%
* 整体虚警率 FDR ≤ 20%
* 单幅 10000×10000 图像推理时间 ≤ 20s
* 评估时车辆目标 IoU 阈值为 0.35，其他目标 IoU 阈值为 0.50
* 预测框按 score 从高到低匹配，每个 GT 最多匹配一次，重复检测算 FP
* 虚警率计算公式：FDR = FP / (TP + FP)

补充说明：

* 当前 train/val 为从官方有标签数据中自行划分得到，仅用于本地验证和方案选择。
* 最终比赛结果以官方 test 集评测为准。
* 当前评估脚本已按 25 个子类严格匹配，即预测子类必须与 GT 子类完全一致才算 TP，再汇总到 ship / aircraft / vehicle 三大类。

⸻

2. 数据版本

官方有标签数据已划分为：

数据集	图片数量
train	3585
val	896
total	4481

数据路径：

official_data/prepared/xh202625_yolo/
├── images/train
├── images/val
├── labels/train
├── labels/val
└── dataset.yaml

过采样配置：

configs/xh202625_oversample.yaml

⸻

3. 实验路线

当前基础路线：

1. E00：环境测试
2. E01：YOLOv8m baseline
3. E02：YOLOv8s speed baseline
4. E03：YOLOv8m + 切片推理 + 类别阈值校准
5. E04：YOLOv8m + 少样本过采样
6. 10000×10000 模拟大图推理速度测试
7. COCO JSON 导出测试
8. 后续创新：类别均衡难例挖掘 / DIOR 预训练 / 更强模型对比

当前基础主方案暂定：

模型：E01 YOLOv8m baseline
权重：runs/yolo/E01_yolov8m_baseline/weights/best.pt
推理方式：切片推理
后处理：default=0.25, 0=0.10, 1=0.10, 24=0.12
结果目录：runs/sliced_infer/E03_v3_balance_24_012

⸻

4. E00 环境测试

* 日期：
* 环境：remote_det
* GPU：NVIDIA H800 PCIe 80GB
* CUDA：12.4
* PyTorch：2.6.0+cu124
* Ultralytics：8.4.67
* 测试模型：YOLOv8m
* 测试图片：bus.jpg
* 结果：推理成功

结论：环境可正常运行 Ultralytics YOLO 检测流程。

⸻

5. E01 YOLOv8m baseline

* 实验名称：E01_yolov8m_baseline
* 日期：待补
* 数据版本：official train/val split，3585 train / 896 val
* 模型：yolov8m.pt
* imgsz：640
* batch：16
* epochs：100
* best epoch：58
* 权重路径：

runs/yolo/E01_yolov8m_baseline/weights/best.pt

YOLO 原生验证结果：

指标	数值
Precision	0.9457
Recall	0.9605
FDR	0.0543
mAP50	0.9755
mAP50-95	0.7895

结论：

E01 是当前综合表现最稳定的主模型。后续切片推理、类别阈值搜索和模拟大图测试均以 E01 作为主要基础模型。

⸻

6. E02 YOLOv8s speed baseline

* 实验名称：E02_yolov8s_speed_b64_nocache
* 日期：
* 数据版本：official train/val split，3585 train / 896 val
* 模型：yolov8s.pt
* imgsz：1024
* batch：64
* epochs：100
* best epoch：59
* 权重路径：

runs/yolo/E02_yolov8s_speed_b64_nocache/weights/best.pt

YOLO 原生验证结果：

指标	数值
Precision	0.9614
Recall	0.9317
FDR	0.0385
mAP50	0.9688
mAP50-95	0.7840

结论：

E02 的 Precision 更高、FDR 更低，但 Recall 低于 E01。该模型可作为速度保底或轻量化备用方案，当前不作为主方案。

⸻

7. E03 YOLOv8m 切片推理

* 实验名称：E03_val_yolov8m_conf005
* 基础模型：E01 YOLOv8m baseline best.pt
* 推理对象：val 集 896 张图
* 推理方式：滑窗切片推理
* tile size：1024
* overlap：0.2
* conf：0.05
* iou：0.5
* 输出路径：

runs/sliced_infer/E03_val_yolov8m_conf005

切片推理统计：

项目	数值
images	896
total_time	26.111s
avg_time_per_image	0.029s
total_tiles	904
used_tiles	903
total_detections	4638

结论：

切片推理流程可正常运行，输出格式为：

cls score x1 y1 x2 y2

该结果作为后续类别阈值过滤的原始预测目录。

⸻

8. E03 类别阈值校准

8.1 Recall 优先版：E03_v2_recall_24_010

阈值设置：

default=0.25
0=0.10
1=0.10
24=0.10

结果目录：

runs/sliced_infer/E03_v2_recall_24_010

严格 25 子类 + 三大类评估结果：

指标	Overall	ship	aircraft	vehicle
Recall	0.9741	0.8822	0.9887	0.8875
FDR	0.0686	0.1804	0.0459	0.3107

结论：

该方案整体 Recall 最高，vehicle Recall 较高，但 vehicle FDR 偏高。适合作为 Recall 优先备用策略。

⸻

8.2 平衡版：E03_v3_balance_24_012

阈值设置：

default=0.25
0=0.10
1=0.10
24=0.12

结果目录：

runs/sliced_infer/E03_v3_balance_24_012

严格 25 子类 + 三大类评估结果：

指标	Overall	ship	aircraft	vehicle
Recall	0.9734	0.8822	0.9887	0.8500
FDR	0.0678	0.1804	0.0459	0.2917

结论：

该方案整体 Recall 仍然很高，整体 FDR 较低，vehicle Recall 刚好达到 85%。当前作为基础主策略。

⸻

9. 类别 24 FSC 阈值搜索

搜索类别：24 FSC
IoU 阈值：0.35

阈值	TP	FP	FN	Precision	Recall	FDR
0.08	71	38	9	0.6514	0.8875	0.3486
0.10	71	32	9	0.6893	0.8875	0.3107
0.12	68	28	12	0.7083	0.8500	0.2917
0.14	66	27	14	0.7097	0.8250	0.2903
0.16	66	26	14	0.7174	0.8250	0.2826
0.18	65	24	15	0.7303	0.8125	0.2697
0.20	65	22	15	0.7471	0.8125	0.2529
0.22	64	22	16	0.7442	0.8000	0.2558
0.24	63	21	17	0.7500	0.7875	0.2500

结论：

类别 24 单纯通过提高阈值无法同时实现高 Recall 和低 FDR。阈值提高后 FP 会下降，但 Recall 也明显下降。当前选择 24=0.12 作为平衡策略，24=0.10 作为 Recall 优先策略。

⸻

10. E04 YOLOv8m oversample

* 实验名称：E04_yolov8m_oversample
* 日期：待补
* 数据版本：official train/val split + oversample train list
* 模型：yolov8m.pt
* 配置：

configs/xh202625_oversample.yaml

* 权重路径：

runs/yolo/E04_yolov8m_oversample/weights/best.pt

训练结果：

* EarlyStopping 触发
* 共训练 74 epoch
* best epoch：44
* best.pt 已保存

10.1 E04 平衡版：24=0.12

阈值设置：

default=0.25
0=0.10
1=0.10
24=0.12

结果目录：

runs/sliced_infer/E04_v1_balance_24_012

指标	Overall	ship	aircraft	vehicle
Recall	0.9697	0.8781	0.9827	0.9500
FDR	0.0808	0.1889	0.0527	0.4286

10.2 E04 Recall 优先版：24=0.10

阈值设置：

default=0.25
0=0.10
1=0.10
24=0.10

结果目录：

runs/sliced_infer/E04_v2_recall_24_010

指标	Overall	ship	aircraft	vehicle
Recall	0.9697	0.8781	0.9827	0.9500
FDR	0.0818	0.1889	0.0527	0.4493

结论：

E04 过采样显著提高 vehicle Recall，但同时大幅增加 vehicle FP，导致 vehicle FDR 明显升高。简单过采样能够增强模型对 FSC 的敏感性，但会带来更多虚警。当前 E04 不作为基础主方案，但可作为后续类别均衡难例挖掘的依据。

⸻

11. 10000×10000 模拟大图生成

生成路径：

official_data/prepared/mosaic_10000/images
official_data/prepared/mosaic_10000/labels

生成设置：

num=10
tile_size=1000
mosaic_size=10000
grid=10x10

生成统计：

项目	数值
source_images	896
generated_mosaics	10
used_tiles	1000
bad_tiles	0
total_objects	4630

结论：

已成功生成 10 张 10000×10000 模拟大图，用于验证大图切片推理、坐标还原、速度统计和提交格式导出流程。

⸻

12. 10000×10000 大图推理速度测试

使用模型：

runs/yolo/E01_yolov8m_baseline/weights/best.pt

推理设置：

tile=1024
overlap=0.2
conf=0.05
iou=0.5

输出目录：

runs/sliced_infer/E01_mosaic_conf005

速度结果：

项目	数值
images	10
total_time	22.357s
avg_time_per_image	1.508s
max_time_per_image	2.827s
total_tiles	1440
used_tiles	1440
total_detections	7126

结论：

10000×10000 模拟大图平均推理时间为 1.508s，最大推理时间为 2.827s，远低于 20s 要求，说明当前切片推理流程满足工程时效性要求。

⸻

13. 模拟大图阈值测试

13.1 大图平衡版

阈值设置：

default=0.25
0=0.10
1=0.10
24=0.12

结果目录：

runs/sliced_infer/E01_mosaic_v3_balance_24_012

指标	Overall	ship	aircraft	vehicle
Recall	0.9592	0.8549	0.9723	0.8462
FDR	0.2593	0.4006	0.2387	0.4444

⸻

13.2 大图 conservative 版

阈值设置：

default=0.35
0=0.15
1=0.15
24=0.20

结果目录：

runs/sliced_infer/E01_mosaic_conservative

指标	Overall	ship	aircraft	vehicle
Recall	0.9572	0.8527	0.9704	0.8462
FDR	0.2300	0.3644	0.2125	0.3293

⸻

13.3 大图 more conservative 版

阈值设置：

default=0.45
0=0.20
1=0.20
24=0.30

结果目录：

runs/sliced_infer/E01_mosaic_more_conservative

指标	Overall	ship	aircraft	vehicle
Recall	0.9557	0.8504	0.9694	0.8154
FDR	0.2077	0.3280	0.1932	0.2535

结论：

在模拟大图上提高阈值可以显著降低 FDR，Overall FDR 从 0.2593 降至 0.2077，同时 Recall 仅从 0.9592 降至 0.9557。但由于模拟大图由多张小图拼接而成，存在拼接边界和非自然背景，FDR 偏高不能完全代表真实官方 test 表现。该实验主要用于验证大图推理流程、速度约束和备用阈值策略。

⸻

14. COCO JSON 导出测试

已成功导出：

results/E01_mosaic_v3_balance_submission.json

导出统计：

项目	数值
images	10
images_with_detections	10
raw_predictions	5996
exported_predictions	5996
invalid_boxes	0

结论：

预测 txt 到 COCO JSON 的转换链路已打通，可用于后续官方 test 结果导出。

待补导出：

results/E01_mosaic_more_conservative_submission.json

⸻

15. 当前结论

当前基础主方案：

模型：E01 YOLOv8m baseline
权重：runs/yolo/E01_yolov8m_baseline/weights/best.pt
推理方式：1024 tile + 0.2 overlap 切片推理
主阈值策略：default=0.25, 0=0.10, 1=0.10, 24=0.12

当前备用方案：

Recall 优先：default=0.25, 0=0.10, 1=0.10, 24=0.10
大图保守：default=0.45, 0=0.20, 1=0.20, 24=0.30

基础实验结论：

1. E01 YOLOv8m baseline 是当前最稳定的基础主模型。
2. E02 YOLOv8s 适合作为速度保底方案。
3. E04 简单过采样能提高 FSC 召回，但也会显著增加车辆类虚警，暂不作为主方案。
4. 类别阈值校准能在 Recall 和 FDR 之间提供灵活权衡。
5. 切片推理流程能处理 10000×10000 大图，速度满足 20s 要求。
6. COCO JSON 导出流程已验证成功。
7. 后续创新应重点围绕类别均衡难例挖掘、负样本抑制和更强遥感预训练展开。

⸻

16. 下一步计划

近期基础工作：

1. 导出 E01_mosaic_more_conservative_submission.json
2. 检查 tools/final_infer_test.py 是否可一键完成 test 推理
3. 整理基础版实验表格
4. 准备基础版技术报告框架

后续创新方向：

1. E05：Class-Balanced Hard Example Mining，类别均衡难例挖掘训练策略
2. DIOR 遥感公开数据预训练 + 官方数据微调
3. YOLOv8l / YOLO11m 更强模型对比
4. 必要时尝试 RT-DETR 作为对照模型


## final_infer_test.py 一键推理流程测试

测试模型：runs/yolo/E01_yolov8m_baseline/weights/best.pt  
测试图片：official_data/prepared/mosaic_10000/images  
输出目录：runs/final_test/test_E01_mosaic_balance  
提交文件：results/test_E01_mosaic_balance_submission.json  

阈值策略：
default=0.25, 0=0.10, 1=0.10, 24=0.12

速度结果：
images=10
avg_time_per_image=1.492s
max_time_per_image=2.991s
total_tiles=1440
used_tiles=1440
total_detections=7126

导出结果：
exported_predictions=5996
invalid_boxes=0

结论：
一键推理流程已跑通，可完成切片推理、类别阈值过滤和 COCO JSON 导出。10000×10000 模拟大图最大推理时间为 2.991s，满足单图 20s 内推理要求。