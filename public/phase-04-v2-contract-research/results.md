# Phase 4 执行结果

> **结论**：v2 124 维输入契约、四个下肢残差输出和基本坐标转换测试已落到代码；实际服务端特征映射仍需按契约核验，见 Phase 5。

## 已完成

- [`contract_v2.py`](../../ai_corrector/corrector/contract_v2.py) 定义 8 路输入切片、总维度 124、输出关节顺序及旋转转换工具。
- 代码包含维度覆盖测试和已知姿态转换测试（T-Pose、90° Yaw、坐姿）。
- 调研报告已归档到本阶段目录；其中方法建议是待验证假设，不是已确认的模型缺陷或收益。

## 尚未由此阶段证明

- 不能由上身 VD/HMD 参考推导出下肢 IMU 偏航绝对准确。
- Python 契约测试不等于 Kotlin 服务端适配器逐通道一致，也不等于在线 ONNX 推理成功。
- 调研稿中关于 Mean Pooling 必然丢失信息、SparsePoser 支持 FK 损失、PIP 可直接根除脚滑及量化收益的表述需要谨慎看待；具体证据边界列在 [Phase 6 子计划](../phase-06-accuracy-baseline/plan.md)。
