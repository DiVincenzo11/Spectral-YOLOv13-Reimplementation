# 数据接口

本仓库没有原CR-Mix++。合成数据仅用于检查程序。所有路径支持相对路径；检测YAML相对自身目录，时序manifest相对自身目录。

## 检测

采用标准YOLO检测数据布局：

```text
data/coral/
  images/train/*.jpg
  images/val/*.jpg
  images/test/*.jpg
  labels/train/*.txt
  labels/val/*.txt
  labels/test/*.txt
```

每行 `class_id x_center y_center width height`，坐标归一化至[0,1]。类序固定为0 Healthy、1 Sub-healthy、2 Bleached、3 Dead。图片与标注同名。确认没有目标的图片使用空txt。

使用 `configs/data.example.yaml` 指向自己的根目录。不要把其他数据集的二分类标签直接改名为四类健康标注。

## 预测与联合训练

JSONL每行一个样本。输入帧为RGB图；三个未来target为float32二维`.npy`健康热图，统一定义在[0,1]，0=死亡、1=健康。所有图必须完成空间配准，代表同一观测区域。输入帧按时间排序。

```json
{"sample_id":"reef_a_0001","site_id":"reef_a","sequence_id":"tracked_patch_01","split":"train","input_paths":["images/t0.jpg","images/t1.jpg","images/t2.jpg","images/t3.jpg"],"input_times_days":[0,1,2,3],"target_paths":["maps/day10.npy","maps/day17.npy","maps/day31.npy"],"target_times_days":[10,17,31],"last_label_path":"labels/t3.txt"}
```

- `last_label_path`为当前帧检测标注；joint模式用它计算检测损失。forecast模式仍要求该字段以保持统一接口，空txt可表示没有检测目标。
- 单个batch内序列长度、目标图尺寸必须一致；变长序列可用batch1，或自行按长度分桶。
- `sequence_id`代表同一固定区域的观测流，不可每个重叠窗口换新ID规避检查。不同划分的观测及目标窗口必须分离；不同独立场地可按site划分。
- 未来目标固定为当前帧之后7/14/28天，不允许直接拿过去或当前的热图充当未来目标。
- 输入图在时序训练中直接缩放到模型尺寸；不做破坏时间对齐的mosaic。检测单帧训练使用上游标准letterbox/YOLO增强。
- 数据协议比论文含糊的周段描述更显式。若有真实采样协议，应据其调整配置与验证器，报告差异。

检查命令：

```bash
python scripts/check_dataset.py --data configs/data.example.yaml --manifest data/coral/temporal.jsonl
```

检测工具扫描损坏图片、标签范围等；时序检查加载所有图/热图，检查时间、文件复用与跨split时间重叠。它不能替代海洋生物学标注验证，也不能保证不同文件名下不存在重复图片。
