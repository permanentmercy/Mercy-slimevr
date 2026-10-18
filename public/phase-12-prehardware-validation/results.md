# Phase 12 规划记录与执行结果

> **当前状态：Task S0（输入契约与模型 ABI 冻结）已全部完成并通过验收 (PASS)；Task S1（服务端 ONNX 影子推理）与 Task A1（慢偏航物理假设复核）推进中。**
> 针对代码审查中发现的下肢 6D 旋转列顺序 bug、全零角速度以及缺少 Phase 11 世界轴投影专用 ONNX wrapper 等缺陷，已完成全套契约对齐、Kotlin 单元测试（6 项全通）与 ONNX 导出对齐验证。

---

## 一、 任务门禁状态总览

| 任务编号 | 任务主题 | 门限要求 | 当前执行状态 | 验收裁定 |
|---|---|---|---|---|
| **S0** | **冻结输入契约与模型 ABI** | 124 维字段定义唯一且对齐；6D 顺序无误；哨兵 fixture 逐元素比对通过；ONNX 递归状态 8 帧对齐 | **已完成**：CorrectorBridge 列顺序修复；因果角速度推导上线；Kotlin 单测 6 项全过；ONNX 导出完成且误差 $< 3.6\times 10^{-7}$ | **PASS** |
| **S1** | **真实 ONNX 影子推理** | 引入 CPU ONNX Runtime 依赖；Session 单例常驻；有效样本推理；不覆写姿态 | 待执行 (准备引入依赖与推理调度) | 待执行 |
| **S2** | **服务端确定性回放与延迟验收** | 108,000 帧长程无内存泄漏；端到端 P95 $\le 20$ ms；故障旁路自动测试 | 待执行 (依赖 S1) | 待执行 |
| **A1** | **修正 Phase 9 合成物理与安全评测** | 零漂与随机游走解耦；机体系偏置物理因果顺序；严格零漂偏航偏差 $\le 0.05^\circ$ | 待执行 (与 S1 并行) | 待执行 |
| **A2** | **重新评估 Phase 11 的 P6 增量价值** | 移除 Oracle P9 Twist 基线，以实际因果 P9 为底座；新增 FK/脚滑评测 | 待执行 (依赖 S2 与 A1) | 待执行 |

---

## 二、 Task S0 执行记录与验收证据

### 1. 契约审计与 6D 旋转顺序修正
- **6D 列顺序修复**：审查发现 [`CorrectorBridge.kt`](../../server/server/core/src/main/java/dev/slimevr/tracking/processor/corrector/CorrectorBridge.kt) 中 `writeStream14d()` 在写入第二元素时误写为 `out[offset + 1] = colY.y`，导致缺失了第一列的 Y 分量 `colX.y`。现已彻底修正为 `out[offset + 1] = colX.y`。
- **下肢 14D 物理量明确**：
  - 偏移 6..8（3 维加速度）：实测确认对应训练数据集 [`prepare_p11_dataset.py`](../../ai_corrector/corrector/prepare_p11_dataset.py) 中的 `raw_acc`（线加速度，单位 $\text{m/s}^2$），与 SlimeVR `Tracker.getAcceleration()` 返回的物理量完全吻合；原 `contract_v2.py` 中的 `ang_acc` 注释系早期命名歧义，已在工程中固化为线加速度。
  - 偏移 9..11（3 维角速度）：此前 Bridge 硬编码置零。现通过在 [`Tracker.kt`](../../server/server/core/src/main/java/dev/slimevr/tracking/trackers/Tracker.kt) 中补充 `setAngularVelocity()` / `getAngularVelocity()`，并在 [`CorrectorBridge.kt`](../../server/server/core/src/main/java/dev/slimevr/tracking/processor/corrector/CorrectorBridge.kt) 中实现**因果旋转差分角速度推导算法**（基于连续两帧四元数 $\Delta q = q_{t-1}^{-1} q_t$ 提取瞬时轴角速度），若传感器未提供原始陀螺读数时自动因果推导，彻底消除了全零占位符。

### 2. 124D 黄金哨兵 Fixture 与全维双向对齐
- **Fixture 生成器**：编写 [`generate_s0_fixture.py`](../../ai_corrector/corrector/generate_s0_fixture.py)，为 8 路身体流的每一个通道分配唯一的非对称哨兵数值，导出基准 JSON 至 [`fixture_124d_sentinel.json`](../../ai_corrector/artifacts/fixture_124d_sentinel.json)。
- **Kotlin 单元测试套件**：编写 [`CorrectorBridgeContractTest.kt`](../../server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/CorrectorBridgeContractTest.kt)，通过 Gradle 自动化测试：
  - `test01_6dRotationColumnOrdering()`: 验证 6D 连续表示与 Gram-Schmidt 正交化往返保真度（误差 $< 10^{-5}$）—— **通过**；
  - `test02_causalAngularVelocityDerivation()`: 验证两帧旋转因果差分推导角速度精度 —— **通过**；
  - `test03_sentinelFixtureMatching124D()`: 验证各关节点偏移与有效性标志位 —— **通过**；
  - `test04_missingTrackersZeroFilling()`: 验证缺失 Tracker 时严格全零填充且 valid=0 —— **通过**；
  - `test05_resetStateClearsRecurrentAndVelocityCache()`: 验证复位时清空因果状态缓存 —— **通过**；
  - `test06_full124DElementsCompare()`: 动态加载 Python 哨兵 JSON，**对 124 个维度进行逐元素比对，最大偏差 $< 2\times 10^{-4}$** —— **通过**。

### 3. Phase 11 Yaw-Neutral ONNX 模型流式导出与 ABI 锁定
- **专用导出包装器**：编写 [`export_yaw_neutral_onnx.py`](../../ai_corrector/corrector/export_yaw_neutral_onnx.py)，将内嵌世界 Y 轴 Swing-Twist 正交投影算子的 [`cstn_v2_yaw_neutral_world_best.pt`](../../ai_corrector/artifacts/cstn_v2_yaw_neutral_world_best.pt) 包装为单帧因果流式结构。
- **ONNX 模型产物**：导出归档至 [`cstn_v2_yaw_neutral_world.onnx`](../../ai_corrector/artifacts/cstn_v2_yaw_neutral_world.onnx)（4.57 MB, Opset 14）。
- **ABI 与递归状态一致性验证**：
  - 检查工具：`onnx.checker.check_model` 严格通过；
  - 8 帧递归状态连续回放比对（ONNXRuntime vs PyTorch）：
    - `dq` 最大误差：$5.96\times 10^{-8}$
    - `gate` 最大误差：$2.38\times 10^{-7}$
    - `contact` 最大误差：$2.09\times 10^{-7}$
    - `hn` / `cn` 递归隐状态最大误差：$3.58\times 10^{-7}$（均显著优于 $2\times 10^{-5}$ 容限）；
  - 单线程 CPU 推理延迟：均值 **0.536 ms**，P95 **0.604 ms**。
- **SHA-256 签名存证**（归档至 [`cstn_v2_yaw_neutral_world.report.json`](../../ai_corrector/artifacts/cstn_v2_yaw_neutral_world.report.json)）：
  - PyTorch Checkpoint: `b0c32acaa6898a89ea5cdbc2176638fdad95feacc45e3698380ccde048e2c4fc`
  - ONNX Model: `7bc5d04e2ecfb9e7b0d416a72ef3a96303f7f9e7f7bef4a320f106a4712c891c`

---

## 三、 产物索引

- 阶段计划：[plan.md](plan.md)
- 124D 哨兵生成器：[`ai_corrector/corrector/generate_s0_fixture.py`](../../ai_corrector/corrector/generate_s0_fixture.py)
- 124D 哨兵数据：[`ai_corrector/artifacts/fixture_124d_sentinel.json`](../../ai_corrector/artifacts/fixture_124d_sentinel.json)
- Kotlin 契约测试：[`server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/CorrectorBridgeContractTest.kt`](../../server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/CorrectorBridgeContractTest.kt)
- ONNX 专用导出器：[`ai_corrector/corrector/export_yaw_neutral_onnx.py`](../../ai_corrector/corrector/export_yaw_neutral_onnx.py)
- ONNX 模型：[`ai_corrector/artifacts/cstn_v2_yaw_neutral_world.onnx`](../../ai_corrector/artifacts/cstn_v2_yaw_neutral_world.onnx)
- ONNX ABI 报告：[`ai_corrector/artifacts/cstn_v2_yaw_neutral_world.report.json`](../../ai_corrector/artifacts/cstn_v2_yaw_neutral_world.report.json)
