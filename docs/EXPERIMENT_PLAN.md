# XH202625_RemoteDet 实验计划

## 当前进度

比赛任务是检测 25 类陆上目标，评测重点包括整体 Recall、FDR、10000x10000 大图推理速度，以及船、飞机、车辆三个大类的 Recall/FDR 排名。

已完成内容：

- 官方数据划分：3585 train / 896 val。
- YOLOv8m baseline：P=0.9457，R=0.9605，FDR=0.0543。
- YOLOv8s speed baseline：P=0.9614，R=0.9317，FDR=0.0385。
- 切片推理工具：`tools/infer_sliced_yolo_batch.py`。
- 整体/类别评估工具：`tools/eval_sliced_txt.py`。
- 当前最佳后处理：切片推理 + 类别自适应阈值，默认阈值 0.25，类别 24 FSC 阈值 0.10；指标 P=0.9309，R=0.9736，FDR=0.0691。
- 已准备过采样配置：`configs/xh202625_oversample.yaml`，下一步训练 `E04_yolov8m_oversample`。

## 后续路线

1. 训练 `E04_yolov8m_oversample`，重点观察 HM、LQS、KC-10、FSC 等少样本或关键类别的召回变化。
2. 使用验证集小图拼接 10000x10000 模拟大图，验证大图切片推理流程。
3. 在模拟大图上检查切片坐标还原、全图 NMS、速度统计、预测 txt 输出和 COCO JSON 导出。
4. 用三大类评估脚本分别统计船、飞机、车辆的 TP、FP、FN、Recall、FDR。
5. 在 Recall >= 85%、FDR <= 20%、单图推理 <= 20s 的硬约束内，优先保持 Recall，再通过类别阈值控制 FDR。

## 大图验证命令

生成模拟 10000x10000 大图：

```bash
python tools/make_mosaic_10000.py \
  --image_dir official_data/prepared/xh202625_yolo/images/val \
  --label_dir official_data/prepared/xh202625_yolo/labels/val \
  --out_image_dir official_data/prepared/mosaic_10000/images \
  --out_label_dir official_data/prepared/mosaic_10000/labels \
  --num 10 \
  --tile_size 1000 \
  --mosaic_size 10000
```

参数说明：当前脚本使用 `--num` 控制生成大图数量，使用 `--mosaic_size` 控制大图边长；不支持 `--grid` 或 `--num_mosaics` 参数。

切片推理：

```bash
python tools/infer_sliced_yolo_batch.py \
  --model runs/yolo/E04_yolov8m_oversample/weights/best.pt \
  --source_dir official_data/prepared/mosaic_10000/images \
  --out runs/sliced/mosaic_pred_raw \
  --tile 1024 \
  --overlap 0.2 \
  --conf 0.05 \
  --iou 0.5 \
  --device 0
```

类别自适应阈值：

```bash
python tools/filter_pred_by_class_conf.py \
  --src runs/sliced/mosaic_pred_raw \
  --dst runs/sliced/mosaic_pred_thr \
  --default 0.25 \
  --class_thr 24:0.10
```

整体评估：

```bash
python tools/eval_sliced_txt.py \
  --image_dir official_data/prepared/mosaic_10000/images \
  --label_dir official_data/prepared/mosaic_10000/labels \
  --pred_dir runs/sliced/mosaic_pred_thr \
  --iou 0.5
```

三大类评估：

```bash
python tools/eval_group_metrics.py \
  --image_dir official_data/prepared/mosaic_10000/images \
  --label_dir official_data/prepared/mosaic_10000/labels \
  --pred_dir runs/sliced/mosaic_pred_thr
```

默认会一次输出 Overall、ship、aircraft、vehicle 指标，并使用分组 IoU：ship=0.50、aircraft=0.50、vehicle=0.35。Overall 也按各目标所属大类的 IoU 阈值匹配后汇总。

如需显式指定分组和 IoU，可运行：

```bash
python tools/eval_group_metrics.py \
  --image_dir official_data/prepared/mosaic_10000/images \
  --label_dir official_data/prepared/mosaic_10000/labels \
  --pred_dir runs/sliced/mosaic_pred_thr \
  --groups ship:0-3 aircraft:4-23 vehicle:24 \
  --group_iou ship:0.5 aircraft:0.5 vehicle:0.35
```

导出 COCO detection JSON：

```bash
python tools/export_coco_json.py \
  --image_dir official_data/prepared/mosaic_10000/images \
  --pred_dir runs/sliced/mosaic_pred_thr \
  --out_json runs/sliced/mosaic_pred_coco.json \
  --category_id_offset 0
```

如果官方要求类别编号为 1-25，导出时改用：

```bash
python tools/export_coco_json.py \
  --image_dir official_data/prepared/mosaic_10000/images \
  --pred_dir runs/sliced/mosaic_pred_thr \
  --out_json runs/sliced/mosaic_pred_coco_1based.json \
  --category_id_offset 1
```

## 评估关注点

- 整体指标：Recall、FDR、单幅 10000x10000 大图推理时间。
- 大类指标：船类 0-3、飞机类 4-23、车辆类 24。
- 三大类评估脚本默认按 ship=0.50、aircraft=0.50、vehicle=0.35 的 IoU 阈值一次完成匹配和汇总。
- 速度指标：记录 `tools/infer_sliced_yolo_batch.py` 打印的 `avg_time_per_image`，确认是否满足 20s 限制。
- 导出格式：预测 txt 保留内部分析，COCO JSON 用于适配可能的官方提交格式。
