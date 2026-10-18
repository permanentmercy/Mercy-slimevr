# Phase 12 规划记录与执行结果

> **当前状态：Phase 12 全阶段 5 项核心任务（S0、S1、S2、A1、A2）已全部执行完毕并通过严格门禁验收 (ALL PASS)！**  
> 本阶段完成了 SlimeVR Kotlin 服务端底层流式 ONNX 运行时接入与长程无泄漏回放验证，修正了输入契约缺陷与角速度因果差分推导；同时在算法侧彻底排除了真值 Oracle 作弊基线，基于**真实因果慢偏航在线估计器**完成了 Phase 9 物理假设复核与 Phase 11 CSTN 增量价值的 SMPL-H 前向动力学（FK）及脚滑评测。

---

## 一、 任务门禁状态总览

| 任务编号 | 任务主题 | 门限要求 | 执行实测数据 | 验收裁定 |
|---|---|---|---|---|
| **S0** | **冻结输入契约与模型 ABI** | 124 维字段定义唯一且对齐；6D 列顺序修复；哨兵 fixture 逐元素对齐；ONNX 递归状态 8 帧对齐 | 修复 `colX.y` 漏写；实现因果差分角速度；Kotlin 单测 6 项全过；ONNX 误差 $< 3.6\times 10^{-7}$ | **PASS** |
| **S1** | **真实 ONNX 影子推理** | 引入 CPU ONNX Runtime 依赖；Session 单例常驻；有效样本推理；NaN/Inf 安全旁路；严格零姿态篡改 | 引入 ORT 1.18.0；单例生命周期管理；Kotlin 6 项集成单测全过；姿态篡改数为 0 | **PASS** |
| **S2** | **服务端确定性回放与延迟验收** | 108,000 帧长程无内存泄漏；开发机端到端 P95 $\le 20$ ms；故障旁路自动测试 | 108,000 帧逻辑 30 分钟连续回放通过；内存 Delta -0.60 MB；端到端 **P95 仅 0.7301 ms** (门限 20 ms) | **PASS** |
| **A1** | **修正 Phase 9 合成物理与安全评测** | 零漂与随机游走解耦；机体系偏置物理因果顺序；严格零漂偏航偏差 $\le 0.05^\circ$；主场景 95% CI 下界 $> 0$ | 140 条会话多核评测通过；严格零漂四路合计偏差 **0.008977°** (门限 0.05°)；主场景 95% CI 下界 $> +339^\circ$ | **PASS** |
| **A2** | **重新评估 Phase 11 的 P6 增量价值** | 彻底移除 Oracle P9 Twist 基线，以实际因果 P9 为底座；新增 SMPL-H FK 与脚滑评测；Swing 95% CI 下界 $> 0$ | 190 独立序列评测通过；Swing 增益 **+1.35°** (95% CI: [+1.20°, +1.49°])；FK 误差改善 **+12.4 mm**；脚滑改善 **+11.5 mm/s** | **PASS** |

---

## 二、 Task S0 执行记录与验收证据

1. **6D 列顺序修复**：
   - 审查定位 [`CorrectorBridge.kt`](../../server/server/core/src/main/java/dev/slimevr/tracking/processor/corrector/CorrectorBridge.kt) 中 `writeStream14d()` 在写入第二元素时误写为 `out[offset + 1] = colY.y`，已彻底修正为 `out[offset + 1] = colX.y`。
2. **下肢 14D 物理量明确与因果差分推导**：
   - 偏移 6..8（3 维加速度）：实测确认对应训练数据集 [`prepare_p11_dataset.py`](../../ai_corrector/corrector/prepare_p11_dataset.py) 中的 `raw_acc`（线加速度，单位 $\text{m/s}^2$），与 SlimeVR `Tracker.getAcceleration()` 物理读数一致。
   - 偏移 9..11（3 维角速度）：在 [`Tracker.kt`](../../server/server/core/src/main/java/dev/slimevr/tracking/trackers/Tracker.kt) 中补充 `setAngularVelocity()` / `getAngularVelocity()`，并在 [`CorrectorBridge.kt`](../../server/server/core/src/main/java/dev/slimevr/tracking/processor/corrector/CorrectorBridge.kt) 中实现基于连续四元数 $\Delta q = q_{t-1}^{-1} q_t$ 的因果差分角速度推导算法，彻底消除全零占位符。
3. **124D 黄金哨兵 Fixture 与 Kotlin 契约单测**：
   - 编写 [`generate_s0_fixture.py`](../../ai_corrector/corrector/generate_s0_fixture.py) 生成非对称哨兵 JSON [`fixture_124d_sentinel.json`](../../ai_corrector/artifacts/fixture_124d_sentinel.json)。
   - 编写 Kotlin 测试套件 [`CorrectorBridgeContractTest.kt`](../../server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/CorrectorBridgeContractTest.kt)，6 项测试全通，124 维逐元素最大误差 $< 2\times 10^{-4}$。
4. **Phase 11 Yaw-Neutral ONNX 流式导出与 ABI 锁定**：
   - 编写 [`export_yaw_neutral_onnx.py`](../../ai_corrector/corrector/export_yaw_neutral_onnx.py)，将包含世界 Y 轴 Swing-Twist 正交投影算子的模型包装导出至 [`cstn_v2_yaw_neutral_world.onnx`](../../ai_corrector/artifacts/cstn_v2_yaw_neutral_world.onnx)（Opset 14, 4.57 MB）。
   - 8 帧递归状态与 PyTorch 相比，隐状态最大误差 $< 3.6\times 10^{-7}$，哈希固化至 [`cstn_v2_yaw_neutral_world.report.json`](../../ai_corrector/artifacts/cstn_v2_yaw_neutral_world.report.json)。

---

## 三、 Task S1 执行记录与验收证据

1. **ONNX Runtime 核心集成**：
   - 在 [`server/server/core/build.gradle.kts`](../../server/server/core/build.gradle.kts) 中引入官方稳定依赖 `com.microsoft.onnxruntime:onnxruntime:1.18.0`。
   - 编写 [`OnnxCorrectorEngine.kt`](../../server/server/core/src/main/java/dev/slimevr/tracking/processor/corrector/OnnxCorrectorEngine.kt)，实现 `CorrectorInferenceEngine` 抽象接口。Session 在 Bridge 初始化时加载常驻，在服务关闭时通过 `AutoCloseable` 显式释放。
2. **递归状态因果流式维护与安全旁路**：
   - 隐状态 $h, c$ 按 $[8, 1, 1, 96]$ 常驻内存，帧间因果递推；遇到 `resetState()` 时完全清零。
   - 输入与输出实施严格的异常值检测：若检测到 `NaN`、`Inf` 或维度不匹配，引擎自动安全旁路，重置隐状态并返回空结果，避免数值污染。
3. **严格影子模式与测试验收**：
   - 编写集成测试套件 [`CorrectorBridgeInferenceTest.kt`](../../server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/CorrectorBridgeInferenceTest.kt)，涵盖：
     - 测试 1：依赖注入与 Mock 测试替身；
     - 测试 2：真实 ONNX 模型加载与单帧流式推断；
     - 测试 3：隐状态跨帧因果传递与 reset 幂等复原；
     - 测试 4：NaN / Inf 输入安全旁路与状态自动自愈；
     - 测试 5：严格影子模式下，原始 Tracker 物理读数与骨骼姿态逐位绝对不变（0 姿态篡改）。
   - Gradle 测试全绿通过（`BUILD SUCCESSFUL in 11s`）。

---

## 四、 Task S2 执行记录与验收证据

1. **高压回放场景与鲁棒性验证**：
   - 编写 [`CorrectorBridgeReplayBenchmarkTest.kt`](../../server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/CorrectorBridgeReplayBenchmarkTest.kt)，自动注入并测试了 6 种极端边界工况：
     - A: 60 Hz 正常运动流；
     - B: 胸部与双臂 VD 视觉追踪失效/缺失；
     - C: 单路腿部 Tracker 掉线/缺失；
     - D: 采样间隔跳变（$dt = 0.5\text{s}$ 或 $0.001\text{s}$）；
     - E: 会话中途微重置与快速标定；
     - F: 连续异常输入（加速度 NaN 注入安全旁路）。
2. **逻辑 30 分钟连续回放（108,000 帧）长程稳定性**：
   - 模拟连续执行 108,000 帧，期间触发 59 次微重置与状态刷新。
   - 内存变化量 Delta 为 **-0.60 MB**（垃圾回收正常，隐状态完全有界），未发生内存泄漏或数值发散。
3. **P50/P95/P99 延迟基准实测**：
   - 实测 3,000 帧高频统计（归档于 [`s2_replay_benchmark_report.json`](../../ai_corrector/artifacts/s2_replay_benchmark_report.json)）：
     - **端到端 Bridge 延迟**：均值 **0.5719 ms**，P50 **0.5512 ms**，**P95 0.7301 ms**（门限 $\le 20.0$ ms，大幅优于指标），P99 **0.8447 ms**，最大值 **1.3687 ms**；
     - **ONNX 推理计算延迟**：P50 **0.5484 ms**，P95 **0.7241 ms**；
     - **特征组装耗时**：P50 **2.20 $\mu\text{s}$**，P95 **7.00 $\mu\text{s}$**。
   - 裁定：**Task S2 PASS**。

---

## 五、 Task A1 执行记录与验收证据

1. **物理模型解耦与因果约束**：
   - 审查确立：严格零漂与随机游走完全解耦（零漂用例必须关闭随机游走）；
   - 传感器角速度因果模型：$\omega_{\text{raw}} = \omega_{\text{motion}} + b_{\text{body}} + \text{noise}$，严禁在线访问真值漂移曲线或未来端点。
2. **多场景横向对比与样本规模**：
   - 编写高性能并行评测脚本 [`evaluate_phase12_a1_physics_audit.py`](../../ai_corrector/corrector/evaluate_phase12_a1_physics_audit.py)，评测了 140 条 30 分钟长会话（涵盖 8 类工况，4 种策略：Raw, ZARU-only, Gait-only, Full）：
     - **标称正漂移 (+3°/min, N=30)**：Raw 误差 $361.25^\circ$，Full 误差降至 **$5.84^\circ$**，净收益 **$+355.41^\circ$**（95% CI: $[+340.72^\circ, +369.80^\circ]$），胜率 100%；
     - **标称负漂移 (-3°/min, N=30)**：Raw 误差 $361.28^\circ$，Full 误差降至 **$6.20^\circ$**，净收益 **$+355.08^\circ$**（95% CI: $[+339.30^\circ, +369.69^\circ]$），胜率 100%；
     - **高速漂移 (±10°/min, N=30)**：Raw 误差 $302.27^\circ$，Full 误差降至 **$4.43^\circ$**，净收益 **$+297.84^\circ$**（95% CI: $[+254.64^\circ, +336.84^\circ]$），胜率 100%；
     - **纯静态端坐 (All Sitting, N=10)**：Full 误差降至 **$0.34^\circ$**，净收益 **$+342.60^\circ$**，证明静态姿态无冲刷退化；
     - **上身 VD 缺失 (VD Missing, N=10)**：纯下肢独立运行，Full 误差降至 **$6.91^\circ$**，净收益 **$+354.02^\circ$**。
3. **严格零漂安全否决项**：
   - 在 10 条严格零漂会话（$0.0^\circ/\text{min}$ 且无随机游走）中：
     - 实测四路合计端点偏航最大偏移量为 **0.008977°**，远小于安全否决上限 **0.05°**！
   - 裁定：**Task A1 PASS**（归档于 [`phase12_a1_physics_audit_report.json`](../../ai_corrector/artifacts/phase12_a1_physics_audit_report.json)）。

---

## 六、 Task A2 执行记录与验收证据

1. **排除 Oracle 作弊基线**：
   - 彻底废除 Phase 11 早期采用真值残差拆解出来的“Oracle P9 Twist”作弊基准；
   - 建立基于**真实因果在线估计器 [`DeployableBiasEstimatorV3`](../../ai_corrector/corrector/bias_estimator_v3.py)** 的基线：输入 124 维连续流，单向实时输出偏航估计角 $\hat{e}_i(t)$，施加世界左乘 $R_y(-\hat{e}_i(t)) R_{\text{raw}}$ 作为实际 P9-only 姿态。
2. **SMPL-H 骨骼前向动力学（FK）与脚滑代理**：
   - 编写 [`evaluate_phase12_a2_p6_incremental_value.py`](../../ai_corrector/corrector/evaluate_phase12_a2_p6_incremental_value.py)，在 190 条独立 FinalConfirm 序列（57,000 帧）上全面评测四种策略（Raw、P9-only、P6-only、P9+P6）：
     - **世界系 Swing 测地误差**：
       - Raw: $2.05^\circ$
       - Actual Causal P9: $2.05^\circ$（慢偏航只校正 Twist，不改变高频 Swing）
       - **Hybrid (P9+P6)**: **0.70°**
       - **净收益**: **+1.35°**，95% Bootstrap 置信区间 **[+1.20°, +1.49°]**，下界严格大于 0（胜率 100%）！
     - **SMPL-H 下肢关节 FK 位置误差**（左膝、右膝、左踝、右踝）：
       - Raw: $21.6\text{ mm}$
       - Actual Causal P9: $21.6\text{ mm}$
       - **Hybrid (P9+P6)**: **9.2 mm**（误差降低 **12.4 mm**，提升 **57.4%**！逐序列最差恶化量为 $-0.0\text{ mm}$，无任何序列退化）！
     - **支撑期脚滑代理速度**（Foot Sliding Velocity）：
       - Raw: $443.4\text{ mm/s}$
       - Actual Causal P9: $443.4\text{ mm/s}$
       - **Hybrid (P9+P6)**: **431.8 mm/s**（脚滑平均减少 **11.5 mm/s**）！
3. **参数化鲁棒性与压力测试**：
   - 在高斯噪声、2 帧延迟、VD 完全缺失及安装偏差扰动下，P9+P6 相比 P9-only 的 Swing 改善均稳定保持在 **$+1.35^\circ$**，FK 改善均在 **$+12.3 \sim +12.4\text{ mm}$**，脚滑减少 **$+10.1 \sim +11.6\text{ mm/s}$**。
   - 裁定：**Task A2 PASS**（归档于 [`phase12_a2_p6_incremental_report.json`](../../ai_corrector/artifacts/phase12_a2_p6_incremental_report.json)）。

---

## 七、 产物与测试索引

- **阶段文档**：
  - [plan.md](plan.md)
  - [results.md](results.md)
- **Task S0 产物**：
  - 124D 哨兵生成器：[`ai_corrector/corrector/generate_s0_fixture.py`](../../ai_corrector/corrector/generate_s0_fixture.py)
  - 124D 哨兵基准：[`ai_corrector/artifacts/fixture_124d_sentinel.json`](../../ai_corrector/artifacts/fixture_124d_sentinel.json)
  - Kotlin 契约测试：[`server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/CorrectorBridgeContractTest.kt`](../../server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/CorrectorBridgeContractTest.kt)
  - ONNX 专用导出器：[`ai_corrector/corrector/export_yaw_neutral_onnx.py`](../../ai_corrector/corrector/export_yaw_neutral_onnx.py)
  - ONNX 模型：[`ai_corrector/artifacts/cstn_v2_yaw_neutral_world.onnx`](../../ai_corrector/artifacts/cstn_v2_yaw_neutral_world.onnx)
  - ONNX ABI 报告：[`ai_corrector/artifacts/cstn_v2_yaw_neutral_world.report.json`](../../ai_corrector/artifacts/cstn_v2_yaw_neutral_world.report.json)
- **Task S1/S2 产物**：
  - Kotlin 推理引擎：[`server/server/core/src/main/java/dev/slimevr/tracking/processor/corrector/OnnxCorrectorEngine.kt`](../../server/server/core/src/main/java/dev/slimevr/tracking/processor/corrector/OnnxCorrectorEngine.kt)
  - Kotlin 服务端桥接：[`server/server/core/src/main/java/dev/slimevr/tracking/processor/corrector/CorrectorBridge.kt`](../../server/server/core/src/main/java/dev/slimevr/tracking/processor/corrector/CorrectorBridge.kt)
  - Kotlin 推理单测：[`server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/CorrectorBridgeInferenceTest.kt`](../../server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/CorrectorBridgeInferenceTest.kt)
  - Kotlin 回放基准单测：[`server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/CorrectorBridgeReplayBenchmarkTest.kt`](../../server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/CorrectorBridgeReplayBenchmarkTest.kt)
  - S2 延迟与回放报告：[`ai_corrector/artifacts/s2_replay_benchmark_report.json`](../../ai_corrector/artifacts/s2_replay_benchmark_report.json)
- **Task A1 产物**：
  - 物理复核评测脚本：[`ai_corrector/corrector/evaluate_phase12_a1_physics_audit.py`](../../ai_corrector/corrector/evaluate_phase12_a1_physics_audit.py)
  - A1 物理审计报告：[`ai_corrector/artifacts/phase12_a1_physics_audit_report.json`](../../ai_corrector/artifacts/phase12_a1_physics_audit_report.json)
- **Task A2 产物**：
  - 非 Oracle 增量价值评测脚本：[`ai_corrector/corrector/evaluate_phase12_a2_p6_incremental_value.py`](../../ai_corrector/corrector/evaluate_phase12_a2_p6_incremental_value.py)
  - A2 增量评测报告：[`ai_corrector/artifacts/phase12_a2_p6_incremental_report.json`](../../ai_corrector/artifacts/phase12_a2_p6_incremental_report.json)
