# Spectral-YOLOv13 — 论文重建版 v0.1.0

> **版本状态：非官方论文方法重建 / 软件流程验证。** 原始数据与权重缺失，尚未复现论文的真实珊瑚实验指标。合成数据结果仅用于程序验证，不能用作论文性能证据。版本详情见 [v0.1.0 发布说明](docs/RELEASE_v0.1.0.md)。
>
> **Unofficial paper-based reimplementation.** Original experimental data and checkpoints are unavailable. Synthetic pipeline checks do not establish reproduction of the published coral benchmarks.

依据 **Spectral-YOLOv13: A Dual-Domain Vision-Mamba Sensing Framework for Fine-Grained Coral Health Assessment and Continuous Ecological Forecasting**, Sensors 2026, 26(10), 3265 重写的研究代码。

**这是原代码丢失后的重新实现，不是恢复的原始实验仓库。原CR-Mix++、原始权重和训练日志均不在本仓库。没有宣称复现53.8% mAP、26.8%预测误差降低或论文FPS。**

论文：https://doi.org/10.3390/s26103265

本版包括真实YOLOv13 backbone、Db4两级WIO-Neck、EMA原型CP-Head、Mamba-1 selective SSM预测器、检测/预测/联合训练、评测与推理工具。未定义的细节及论文矛盾点见 [实现说明](docs/IMPLEMENTATION.md)，不能把本版自行补全的选择当作原论文已明确给出的参数。

## 已验证的范围

- Windows / CPU环境下，17项数值与集成测试通过；包括小波重建、原型EMA、Mamba因果性、梯度、空目标和数据泄漏检查。
- 官方YOLOv13-L backbone、640x640输入、P3-P7五尺度前向检查通过。
- 合成几何图片上的检测与联合训练、checkpoint保存加载、检测AP和时序MAE/RMSE计算已执行。
- 672步selective scan检查通过；这不等于在真实672帧珊瑚序列上完成了预测验证。
- CUDA fused scan、GPU AMP、真实珊瑚准确率及Jetson部署尚未验证。

实测记录位于 [artifacts/verification](artifacts/verification)。合成数据不具有海洋生物学意义。

## 安装

建议Python 3.11或3.12，在本目录运行：

```bash
python -m venv .venv
```

Windows PowerShell无需激活，直接：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe scripts/smoke_test.py
```

Linux/macOS使用 `.venv/bin/python`。下文的 `python` 均指已安装这些依赖的环境。不要另行 `pip install ultralytics`：本项目会加载vendor中的指定官方YOLOv13版本。

如需GPU训练，先按 [PyTorch官方安装说明](https://pytorch.org/get-started/locally/) 安装与显卡兼容的torch/torchvision，再安装其余依赖。本次没有修改系统CUDA或现有学习环境；测试使用工作目录中的隔离环境。

`pytorch-wavelets`依赖旧的pkg_resources，因此固定 `setuptools<81`。导入时出现其弃用提示不代表测试失败。

## 一次跑通软件流程

```bash
python scripts/run_demo.py --out runs/synthetic_demo
```

该脚本生成带明确来源标记的矩形图片和简单热图，短程训练检测、联合分支，运行评估与图片输出，生成 `demo_summary.json`。它不会下载真实珊瑚数据，也不会生成伪造论文分数。输出图片会明确注明重建版；未训练足够的模型可能没有检测框。

## 使用真实数据训练

先按 [数据格式](docs/DATA_FORMAT.md) 准备YOLO格式图片及四类健康标注，修改 `configs/data.example.yaml`。

检测训练：

```bash
python -m spectral_yolo.train --config configs/paper.yaml --task detect --data configs/data.example.yaml --device cuda:0 --out runs/detect
```

本机内存不足时，可调 `--batch 2 --image-size 320`；这属于调整实验条件，不能继续声称与论文训练设置一致。默认L/640/300轮需要充分训练资源。

有真实连续观测和未来健康图后，优先从检测checkpoint继续训练预测分支：

```bash
python -m spectral_yolo.train --config configs/paper.yaml --task forecast --checkpoint runs/detect/last.pt --manifest data/coral/temporal.jsonl --device cuda:0 --out runs/forecast
```

或者训练检测+预测联合目标：

```bash
python -m spectral_yolo.train --config configs/paper.yaml --task joint --checkpoint runs/detect/last.pt --manifest data/coral/temporal.jsonl --device cuda:0 --out runs/joint
```

`--checkpoint`表示加载模型权重后重新开始优化，不是恢复原优化器/随机数状态的断点续训。预测单独训练默认冻结检测特征，未先训练检测器时它们是随机特征，只适合软件检查。

`configs/paper.yaml`按可确定的论文设置给出默认值；`configs/smoke.yaml`使用nano/64和小尺寸隐空间，仅用于快速验证。时序默认使用可移植reference scan，长序列大batch会较慢且占用反向传播内存。

## 评测与推理

```bash
python -m spectral_yolo.evaluate --checkpoint runs/detect/last.pt --data configs/data.example.yaml --split test --device cuda:0 --out runs/detection_test.json
python -m spectral_yolo.evaluate --task forecast --checkpoint runs/joint/last.pt --manifest data/coral/temporal.jsonl --split test --device cuda:0 --out runs/forecast_test.json
python -m spectral_yolo.predict --checkpoint runs/detect/last.pt --image your_image.jpg --out runs/prediction.png --device cuda:0
python scripts/visualize_embeddings.py --checkpoint runs/detect/last.pt --data configs/data.example.yaml --out runs/embeddings.png
```

检测报告包括mAP@.5、mAP@.5:.95及每类AP；没有GT的类别标为null并从均值中排除。预测报告按1/2/4周输出MAE和RMSE。PCA是本版的解释工具，不冒充原论文t-SNE图。

评测和推理不会使用验证集更新原型。检测专用checkpoint没有训练过预测器，工具会拒绝据其评估预测能力；反之同理。

## 消融与可配置项

检测命令支持 `--no-wio`、`--no-cp`、`--no-forecast`；例如：

```bash
python -m spectral_yolo.train --config configs/paper.yaml --data configs/data.example.yaml --no-wio --out runs/no_wio
```

这些是同一重建框架内的受控消融。关闭WIO/CP不会自动变成完全相同的原始YOLOv13官方基线；若比较官方基线，应另外使用固定提交的原始模型及同一数据协议。

- `model.pyramid_levels`：3或5，处理论文尺度口径差异。
- `temporal.detach_backbone`：默认true，对应§3.5.2；false可研究真正端到端梯度。
- `temporal.backend`：reference或cuda。cuda需安装兼容mamba-ssm selective scan，不会失败后悄悄替换为其他模型。
- 输入T可变，未来目标时间明确；不从不存在的观测自动制造672帧序列。

## 目录

```text
spectral_yolo/     backbone组合、小波、原型、Mamba、损失、数据、训练/评测/推理
configs/           论文默认配置、快速检查配置、真实数据模板
scripts/           合成数据、完整demo、数值检查、数据检查、嵌入可视化
tests/             数值与集成回归测试
docs/              实现假设、数据格式、验证报告
artifacts/         本次真实执行的软件验证记录
vendor/yolov13/    固定提交的官方上游代码与原许可
```

## 来源、作者与许可

论文作者：Litian Yang, Wenkun Chen, Zhuoyue Mo, Xin Gao, Minzhi Mo, Chunlei Xia, Liankuan Zhang。正式论文将Wenkun Chen列为第二作者，并列有Methodology、Software贡献。本仓库的重建工作不改变原论文作者贡献，也不表示本次重新运行过当年的实验。

上游YOLOv13/Ultralytics代码保留AGPL-3.0许可；本仓库组合实现按同一许可提供，见LICENSE与 [THIRD_PARTY.md](THIRD_PARTY.md)。未再分发原始论文PDF、真实数据或预训练权重。

仓库：https://github.com/DiVincenzo11/Spectral-YOLOv13-Reimplementation
