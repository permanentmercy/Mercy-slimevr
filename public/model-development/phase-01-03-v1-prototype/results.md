# Phase 1–3 执行结果

> **结论**：v1 模型训练、数学验收和 ONNX 工程链路已有记录并通过；没有真实追踪精度或实机端到端验收。

## 已完成

- AMASS / SMPL-H 流水线处理 38 个动作子集，记录为 16,785 条有效序列、165,229 个窗口、约 330.5 万帧。
- Hidden 96 CSTN 有约 0.97M 参数；全量合成训练记录的残差角误差为 0.1533°、合成接触 F1 为 95.80%。
- 工程验收覆盖未来帧隔离、流式等价、四元数归一与限幅、残差乘法方向、门控旁路、梯度、检查点等项目。
- ONNX 报告记录 8 帧递归对齐，`atol=2e-5`；紧凑模型约 3.99 MB，单线程 CPU 均值约 0.397 ms、P95 约 0.610 ms（工程冒烟权重）。
- 12 条序列小样本实验使用 9/3 序列拆分和 72/24 窗口；验证残差角 0.1138°，相对 0.1168° 的合成目标基线约改善 2.6%。

## 原始证据

- [工程验证说明](../../../ai_corrector/corrector/VALIDATION.md)
- [小样本训练记录](../../../ai_corrector/corrector/SUBSET_TRAINING_REPORT.md)
- [工程验收代码](../../../ai_corrector/corrector/acceptance.py)
- [ONNX 紧凑模型报告](../../../ai_corrector/corrector/onnx-compact-report.json)

## 限制

这些数字衡量合成标签拟合、工程性能或合成代理分类，不证明对真实 IMU 长期偏航漂移、真实足底接触或 SlimeVR 全链路有改善。后续 v2 序列隔离评测结果见 [Phase 6](../phase-06-accuracy-baseline/results.md)。
