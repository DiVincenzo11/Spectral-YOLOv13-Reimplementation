# v0.1.0 — Paper-based reimplementation

首次公开发布 Spectral-YOLOv13 论文方法重建版。

**状态：非官方方法重建与软件流程验证；尚未复现论文真实珊瑚实验指标。**

## 包含内容

- 固定上游提交的 YOLOv13 backbone、WIO-Neck、CP-Head 和 Bio-Mamba。
- 检测、预测与联合训练，权重保存加载，评测、推理和数据接口。
- 中文运行说明、实现假设、数据格式、第三方来源及许可说明。

## 软件验证范围

- Windows / CPU 上 17 项数值与集成测试通过。
- YOLOv13-L / 640×640 / P3–P7 前向检查通过。
- 合成数据上的训练、权重保存加载、评测和推理流程已执行。
- 随附验证记录位于 `artifacts/verification/`，详见 `docs/VALIDATION.md`。

## 已知限制

- 原始 CR-Mix++ 数据、官方权重、原始训练日志及数据划分不可用。
- 论文未说明或存在矛盾的细节按 `docs/IMPLEMENTATION.md` 中的假设实现。
- 合成数据结果仅证明软件流程可运行，不代表真实珊瑚识别、生态预测或论文指标复现。
- GPU AMP、CUDA fused scan、Jetson/TensorRT 部署及论文速度尚未验证。

This is an unofficial paper-based reimplementation, not a recovered original experimental repository. Original data and checkpoints are unavailable. Synthetic software checks do not establish reproduction of the published results. See the implementation and validation documents for assumptions and limitations.

License: AGPL-3.0; third-party notices are retained in `THIRD_PARTY.md` and `vendor/`.
