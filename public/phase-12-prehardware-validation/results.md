# Phase 12 规划记录与执行结果

> **复核结论：Phase 12 尚未通过整体验收。** 2026-10-18 对代码、机器报告和测试进行复核后，S0 契约测试通过；S1/S2 本地测试通过但 Session 关闭、完整服务端路径和内存门槛仍有缺口；A1 的静态姿势门禁被硬编码为通过且仿真角速度来源与运行时不一致；A2 复用了 Phase 11 已用过的 FinalConfirm，不能作为确认性结果。A1/A2 原始数字仅保留为合成探索结果。

---

## 一、 任务门禁状态总览

| 任务编号 | 任务主题 | 门限要求 | 执行实测数据 | 验收裁定 |
|---|---|---|---|---|
| **S0** | **冻结输入契约与模型 ABI** | 124 维字段定义唯一且对齐；哨兵 fixture 逐元素对齐；ONNX 递归状态对齐 | 本机契约 6 项通过；服务端角速度目前由姿态差分生成，生产代码未发现原始陀螺读数写入点 | **本机通过；产物需在干净环境复核** |
| **S1** | **真实 ONNX 影子推理** | CPU ONNX Runtime、状态递推/重置、异常旁路、影子姿态不变、Session 释放 | 本机 5 项推理测试通过；`VRServer` 未调用 `correctorBridge.close()`，Session 释放未接入 | **部分完成** |
| **S2** | **服务端确定性回放与延迟验收** | 108,000 帧连续回放；P95 $\le 20$ ms；内存有界；故障旁路 | 13 项指定 Kotlin 测试通过；逐帧 ONNX 推理 108,000 次；组装+推理 P95 **0.6571 ms**；GC 后 heap delta **-0.517 MB**（无内存门限；未计完整 `VRServer.tick()`） | **部分通过** |
| **A1** | **修正 Phase 9 合成物理与安全评测** | 独立静态保持门禁；输入物理量与服务端来源一致；主要场景 CI 下界 $>0$ | 严格零漂旧用例测得 0.008977°；但 `sitting_passed` 未被更新，且仿真直接提供带偏置角速度、服务端使用姿态差分 | **不接受原 PASS** |
| **A2** | **重新评估 Phase 11 的 P6 增量价值** | 新 source-level 隔离集；实际 P9 基线；零漂/静态安全；有依据的 FK/脚滑非劣门禁 | 190 条旧 FinalConfirm 被复用；P9-only 与 Raw 完全相同；FK/脚滑为简化骨长代理，压力条件未进入最终门禁 | **探索结果，不是确认性通过** |

服务端代码位于独立嵌套仓库 `server/`；该仓库当前包含本阶段修改及已有的本机前端/构建调整，相关改动尚未提交。主仓库按既定 `.gitignore` 忽略 `/server/`，因此根仓库提交不会携带这些代码。模型、哨兵 fixture 等本地大产物也被忽略；干净环境需要按报告中的导出步骤准备模型。

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
   - 编写 [`OnnxCorrectorEngine.kt`](../../server/server/core/src/main/java/dev/slimevr/tracking/processor/corrector/OnnxCorrectorEngine.kt)，实现 `CorrectorInferenceEngine` 抽象接口。Engine 提供 `AutoCloseable`，但复核未找到 `VRServer` 生命周期调用 `correctorBridge.close()`；Session 释放尚未接入服务关闭流程。
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
   - 独立复验中契约 6 项、推理 5 项、回放 2 项，共 13 项 Kotlin 测试通过。测试使用相对路径定位本机忽略的 ONNX 产物；干净 checkout 需先提供模型文件。

---

## 四、Task S2 本机回放与性能结果（范围有限）

1. **鲁棒性用例**：`CorrectorBridgeReplayBenchmarkTest` 覆盖正常流、VD 缺失、腿部 tracker 缺失、dt 跳变、reset、NaN 输入等场景；本次 JUnit XML 显示该类 2 项测试通过。
2. **连续回放**：修正后测试对 108,000 个逻辑帧逐帧执行特征组装和 ONNX 推理，共 108,000 次模型调用，执行 59 次状态重置。该项比原先每 30 帧调用一次更接近 60 Hz 流式推理。
3. **延迟**：最新一次 3,000 样本测得特征组装加推理延迟均值 0.5410 ms、P50 0.5234 ms、P95 **0.6571 ms**、P99 0.7698 ms、最大 1.3693 ms；低于 20 ms 目标。测量没有覆盖真实 `VRServer.tick()` 内的 tracker 刷新和调度开销，因此不称完整服务端端到端延迟。
4. **内存**：强制 GC 前后 heap delta 为 **-0.517 MB**。这是单次观测，没有预先定义上限或自动断言，故只能说未观察到 GC 后堆增长，不能据此宣称“无内存泄漏已通过门禁”。
5. **机器报告**：[`s2_replay_benchmark_report.json`](../../ai_corrector/artifacts/s2_replay_benchmark_report.json)。当前工作区已有 Kotlin 13 项测试全部通过记录；S2 回放类在最后一次逐帧改动后单独重跑通过。

**裁定：S2 的逐帧推理与延迟目标通过；完整 VRServer 路径和内存有界门禁仍未完成。**

## 五、Task A1 复核：零漂数字可保留，静态安全验收无效

旧报告在严格零漂、无随机游走的 10 个合成会话中测得四路合计端点估计最大值 **0.008977°**，低于 0.05° 数值门限。这一项只支持该组仿真输入下的终点结果。

不能据此接受 A1：[`evaluate_phase12_a1_physics_audit.py`](../../ai_corrector/corrector/evaluate_phase12_a1_physics_audit.py) 将 `sitting_passed` 初始化为 `True`，之后只打印 `all_sitting` 的漂移收益，从未计算或更新静态保持门禁。`all_sitting` 场景本身还注入了 3°/min 漂移，测的是漂移补偿，不是零漂下不误改静态姿势。

另一个关键不一致是输入来源：[`session_simulator_v3.py`](../../ai_corrector/corrector/session_simulator_v3.py) 将机体系陀螺偏置与噪声直接写入角速度字段；服务端目前没有生产路径调用 `Tracker.setAngularVelocity()`，[`CorrectorBridge.kt`](../../server/server/core/src/main/java/dev/slimevr/tracking/processor/corrector/CorrectorBridge.kt) 因此用融合姿态四元数差分生成角速度。两者不是同一观测链路，当前 A1 增益不能代表服务端可部署输入下的收益。

**裁定：A1 不接受原 PASS。** 在补上独立零漂静态姿势门禁、统一角速度定义并重跑前，表中漂移收益只作为仿真探索结果。全部原始数字见 [`phase12_a1_physics_audit_report.json`](../../ai_corrector/artifacts/phase12_a1_physics_audit_report.json)。

## 六、Task A2 复核：探索性结果不能证明组合增益

旧报告的数值仍有参考价值，但不能作为 A2 验收：评测读取 `p11_continuous/final_confirm_continuous.npz` 中的 190 条序列（57,000 帧），这些是 Phase 11 已使用过的 FinalConfirm；本阶段计划明确要求新建隔离 manifest，并禁止重复作为调参后确认集。报告没有新 manifest，也没有逐条 source ID / 数据哈希。

原始名义条件测得 Swing 收益 **+1.352°**（95% bootstrap CI `[+1.200°, +1.491°]`，序列胜率 **96.84%**），固定骨长位置代理改善 **12.37 mm**，接触期脚踝速度代理改善 **11.51 mm/s**。但 P9-only 的 Swing、位置与速度数值与 Raw 完全相同；P9+P6 又与 P6-only 完全相同。因此这次数据展示的是 P6 在该合成扰动上的单模型收益，没有展示 P9 与 P6 融合后的额外收益。

实现还存在验收覆盖缺口：最终门禁只裁定名义条件的 Swing CI 与均值非劣阈值；噪声、延迟、缺失和安装偏差结果没有进入 PASS 判定，也没有零漂/静态姿势安全集。所谓“SMPL-H FK”实际调用 [`compute_lower_body_fk`](../../ai_corrector/corrector/evaluate_baseline.py)，使用固定 0.42 m 腿段与简化髋偏移，不是 SMPL-H 网格或个体骨长；脚滑计算只用这些局部关节位置，没有叠加 root 世界位移，应称为几何代理，不能当作世界足滑指标。

**裁定：A2 仅为探索结果，不是确认性通过，也不证明 P9+P6 融合有效。** 下一轮需使用此前未参与训练、选模或 Phase 11 评测的新 source-level/subject-level 清单；若不存在新的独立源数据，应将 A2 标为数据受限而非复用旧 FinalConfirm。报告见 [`phase12_a2_p6_incremental_report.json`](../../ai_corrector/artifacts/phase12_a2_p6_incremental_report.json)。
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
