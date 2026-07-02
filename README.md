# XH202625 RemoteDet 使用说明

## 1. 项目说明

本项目用于 XH-202625 光学遥感卫星陆上目标检测识别比赛。

当前主方案：

- 主模型：YOLOv8m baseline
- 权重：`runs/yolo/E01_yolov8m_baseline/weights/best.pt`
- 推理方式：10000×10000 大图切片推理
- 主阈值策略：`default=0.25, 0=0.10, 1=0.10, 24=0.12`
- 一键推理脚本：`tools/final_infer_test.py`

当前最稳结果来自 E01，不使用 E06 DIOR 微调模型作为最终主模型。

---

## 2. 目录说明

XH202625_RemoteDet/
├── configs/          数据集配置文件
├── docs/             实验记录和实验计划
├── tools/            数据处理、推理、评估、导出脚本
├── official_data/    官方比赛数据，不上传 GitHub
├── public_data/      外部公开数据，如 DIOR，不上传 GitHub
├── runs/             训练和推理结果，不上传 GitHub
├── results/          导出的 JSON 结果，不上传 GitHub
├── weights/          YOLO 预训练权重，不上传 GitHub
├── transfer_to_3090/ 换 3090 电脑用的迁移包目录，不上传 GitHub
├── train_*.py        各实验训练脚本
├── README.md         项目说明
├── requirements.txt  Python 依赖
└── .gitignore        Git 忽略规则