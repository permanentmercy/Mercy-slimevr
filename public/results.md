# SlimeVR CSTN AI Corrector 项目总成果与指标汇总

> **文档定位**：本项目所有阶段的**总体成果、核心精度数据与工程性能指标的总汇总看板**。  
> **更新时间**：2026-10-18  
> **数据版本**：Phase 1 至 Phase 12 全阶段实测与复核基准  

---

## 一、 项目全阶段状态与交付结论总览

| 阶段 | 阶段主题 | 核心职责 | 交付成果与关键指标 | 当前状态与裁定 |
|---|---|---|---|---|
| **Phase 1–3** | **v1 原型与 ONNX 工程基线** | 60Hz 采样契约建立，初始 112 维特征流定义与模型骨架打通 | 完成初始数据流定义与 PyTorch 到 ONNX 导出基础链路 | [已完成并归档](phase-01-03-v1-prototype/results.md) |
| **Phase 4** | **v2 数据契约与调研** | 确立 124 维分层流式契约（上身 4 路视觉 68 维 + 下肢 4 路 IMU 56 维） | 锁定 124 维流式特征协议，定义 6D 连续旋转与传感器元数据 | [已完成契约锁定](phase-04-v2-contract-research/results.md) |
| **Phase 5** | **服务端影子桥接工程** | 在 SlimeVR Kotlin 服务端 `server:core` 实现采样拦截与桥接骨架 | 搭建 `CorrectorBridge`，打通 `Tracker.tick()` 与姿态解算链路 | [核心由 Phase 12 承接落地](phase-05-server-shadow-mode/results.md) |
| **Phase 6** | **序列隔离与补偿精度基线** | 建立 796 条时序集对旧 CSTN 全量残差模型审计 | 揭示旧模型世界偏航局部右乘导致的物理退化，确立正交分频解耦路线 | [已完成审计复核](phase-06-accuracy-baseline/results.md) |
| **Phase 7** | **合成回放与离线验收** | 构建 60Hz 合成时序回放机制，制定延迟与稳定性评测规范 | 建立 10.8 万帧连续回放评测标准，由 Phase 12 落地实测 | [由 Phase 12 承接落地](phase-07-replay-validation/results.md) |
| **Phase 8** | **实机采集与在线迁移** | 规划实机硬件日志采集协议、独立动捕真值环境对照与在线姿态应用 | 规划硬件部署、光学动捕对照方案与在线姿态解算 | [待硬件就绪后启动](phase-08-device-validation/plan.md) |
| **Phase 9** | **四路偏航状态模型重构** | 设计可部署的显式偏航状态卡尔曼滤波器 `DeployableBiasEstimatorV3` | 4 路 2x2 EKF 慢偏航估计器，30 分钟合成序列漂移抑制达 96.6% | [合成信号积极；待实机闭环](phase-09-stateful-bias-model/results.md) |
| **Phase 10** | **README 与阶段成果同步** | 规范化多阶段 `public/` 文档管理与团队协作流程 | 统一全阶段双层文档架构与自动化索引规范 | [已完成](phase-10-readme-sync/results.md) |
| **Phase 11** | **Phase 6/9 正交组合验证** | 提出世界 Y 轴偏航中立约束（Yaw-Neutral CSTN），消除跨频段偏航串扰 | 4 项离线门禁 PASS；Oracle 状态下 Swing 测地误差降低 66.7% | [正交解耦通过；增量由 Phase 12 复测](phase-11-hybrid-compensation/results.md) |
| **Phase 12** | **无实机影子推理、回放与算法复验** | 契约修复、服务端真实 ONNX 流式接入、10.8 万帧回放、严格零漂安全否决、SMPL-H FK 与脚滑评测 | **S0/S1/S2 工程单测全通，回放 P95=0.657~0.73ms；A1/A2 离线探索指标积极** | [阶段复核报告](phase-12-prehardware-validation/results.md) |

---

## 二、 算法架构与解耦理论体系

针对 IMU 动作捕捉中存在的“低频积分发散漂移”与“高频动作失真残差”混合交织的问题，项目确立了**双分支正交分频解耦补偿架构**：

```
                原始传感器读数 (R_raw, omega, acc) + 上身视觉上下文 (HMD, VD Chest)
                                         │
                                         ▼
                                 124 维标准时序流
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │                                               │
                 ▼                                               ▼
     【慢分支：物理状态估计】                         【快分支：神经运动学修正】
  Phase 9 DeployableBiasEstimatorV3              Phase 11/12 Yaw-Neutral CSTN (ONNX)
  - 4 路独立 ZARU 静止频带/方差零偏检测            - 8 路因果流式 LSTM (Hidden Dim=96)
  - 可信直线步态矢状摆动面航向分析                 - 针对下肢 4 关节输出 3D 旋转残差
  - 四路 2x2 状态协方差卡尔曼滤波 (EKF)            - 内嵌世界 Y 轴 Swing-Twist 正交投影
  - 估计四路世界竖直轴慢偏航误差 e_i(t)            - 严格剥离世界偏航 Twist，仅保留高频 Swing
                 │                                               │
                 ▼                                               ▼
         世界逆偏航左乘修正                              局部右乘 Swing 补偿
       R_P9 = R_y(-e_i) @ R_raw                        R_hybrid = R_P9 @ dq_swing_gated
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                            最终修正姿态 (供求解器输入)
                         世界偏航串扰恒为 0，高频动作细节还原
```

### 正交分频数学保证
世界竖直轴偏航分解为四元数 Swing-Twist：
$$q = q_{\text{swing}} \otimes q_{\text{twist}},\quad q_{\text{twist}} = \frac{[w, 0, y, 0]^T}{\sqrt{w^2 + y^2}}$$
$$q_{\text{neutral}} = q \otimes q_{\text{twist}}^*$$
输出残差 $\Delta q$ 在世界系下经正交投影剥离世界 Y 轴旋转量，确保快分支网络无法产生世界偏航扰动（实测偏航串扰 $< 10^{-6\circ}$）。

---

## 三、 全阶段核心量化指标汇总看板

### 1. 服务端工程与运行时性能指标 (Phase 12 Task S0–S2 实测)

所有测试均在本地 Java 17 + Gradle 8.11 环境下，通过真实 ONNX Runtime 1.18.0 CPU 引擎驱动执行。

| 性能与工程指标 | 验收目标 / 门限 | 实际测量值 | 优于门限倍数 / 状态 | 测试用例与证据源 |
|---|---|---|---|---|
| **Kotlin 单元测试通过率** | 100% 全部通过 | **13 / 13 全通 (100%)** | 全部绿灯 | `CorrectorBridgeContractTest` 等 3 个套件 |
| **124D 契约哨兵逐元素比对** | 最大绝对误差 $< 10^{-3}$ | **$< 1.83 \times 10^{-4}$** | 优于门限 5.4 倍 | `fixture_124d_sentinel.json` 契约比对 |
| **长程连续回放稳定性** | 连续 30 分钟无崩溃 | **108,000 帧全通 (30分钟)** | 0 崩溃，59 次 reset 幂等恢复 | `CorrectorBridgeReplayBenchmarkTest` |
| **端到端 Bridge 处理延迟 P95** | $\le 20.0$ ms | **0.6571 ms ~ 0.7301 ms** | **优于门限 27.4 ~ 30.4 倍** | `s2_replay_benchmark_report.json` |
| **端到端 Bridge 处理延迟 P50** | 实时流畅 | **0.5234 ms ~ 0.5512 ms** | 极低系统开销 | 3,000 样本实测基准 |
| **端到端 Bridge 处理延迟 P99** | $\le 20.0$ ms | **0.7698 ms ~ 0.8447 ms** | 优于门限 23.7 倍 | 3,000 样本实测基准 |
| **端到端 Bridge 最大处理延迟** | $\le 50.0$ ms | **1.3687 ms ~ 1.3693 ms** | 远低于 5ms 服务端超时防抖 | 3,000 样本实测基准 |
| **ONNX 模型前向推理延迟 P95** | - | **0.6501 ms ~ 0.7241 ms** | 占总耗时 99.1% | ONNX Runtime CPU 单线程 |
| **124D 特征提取与正交化 P95** | - | **7.00 $\mu\text{s}$ (0.007 ms)** | 仅占总耗时 0.9% | 纯标量优化 Gram-Schmidt |
| **长程内存稳定性 (GC 堆差值)** | 堆内存严格有界 | **Delta = -0.517 MB ~ -0.60 MB** | 内存平稳，无递增泄漏 | 108,000 帧前后强制 Full GC 观测 |
| **异常输入安全旁路测试** | 正常流 0 误触，NaN 100% 旁路 | **正常 0/3000，异常 100% 隔离** | 状态自愈，零污染后续帧 | NaN / Inf 注入测试套件 |
| **影子模式姿态保真度** | 姿态 0 篡改 | **原始 Tracker 姿态逐位恒等** | 严格只读旁路 | `shadowMode = true` 验证 |

> [!NOTE]
> **服务端集成边界说明**：
> - 当前测量的延迟覆盖从原始 Tracker 数据提取 124 维特征流、执行正交投影并调用 ONNX 前向推理的全过程；未覆盖完整的 `VRServer.tick()` 外围调度。
> - `CorrectorBridge` 已实现 `AutoCloseable` 接口，服务退出时的生命周期挂载（`VRServer.close()` 调用）待在后续工程阶段接入。

---

### 2. 慢分支长程偏航物理估计指标 (Phase 9 / Phase 12 Task A1 实测)

评测基于 140 条独立连续 30 分钟会话（252 万帧合成时序），覆盖 8 类运动工况；采用纯在线因果特征与纯标量优化求解器。

| 工况场景 | 测试样本量 | Raw 累计偏航误差 | Full 修正后误差 | 净收益 $G$ | 95% Bootstrap CI | 序列胜率 | 偏航抑制率 |
|---|---|---|---|---|---|---|---|
| **标称正偏航 (+3°/min)** | 30 条会话 | 361.25° | **5.84°** | **+355.41°** | `[+340.72°, +369.80°]` | 100.0% | **98.4%** |
| **标称负偏航 (-3°/min)** | 30 条会话 | 361.28° | **6.20°** | **+355.08°** | `[+339.30°, +369.69°]` | 100.0% | **98.3%** |
| **高偏航漂移 (±10°/min)** | 30 条会话 | 302.27° | **4.43°** | **+297.84°** | `[+254.64°, +336.84°]` | 100.0% | **98.5%** |
| **纯静态端坐 (All Sitting)** | 10 条会话 | 342.93° | **0.34°** | **+342.60°** | `[+322.18°, +362.35°]` | 100.0% | **99.9%** |
| **上身 VD 丢失 (VD Missing)**| 10 条会话 | 360.93° | **6.91°** | **+354.02°** | `[+334.25°, +374.69°]` | 100.0% | **98.1%** |
| **持续转身漫步 (No Still)**  | 10 条会话 | 362.68° | 348.83° | +13.85° | `[+5.11°, +22.67°]` | 70.0% | 步态分析防发散 |
| **纯随机游走 (零均值偏置)**   | 10 条会话 | 4.98° | 2.82° | +2.16° | `[+1.38°, +2.99°]` | 100.0% | 自适应噪声平滑 |

- **严格零漂安全否决项 (Strict Zero-Drift Veto)**：
  - **测试条件**：`drift_tier = 0.0 deg/min` 且关闭随机游走（`enable_random_walk = False`），连续运行 30 分钟；
  - **门限要求**：四路下肢 IMU 累计端点偏航改变量绝对值之和 $\le 0.05^\circ$；
  - **实测数值**：四路合计最大端点偏航改变量为 **0.008977°**（仅占安全上限的 17.9%）；
  - **安全裁定**：实测数值通过安全否决门槛；审计指出静态姿势保持门禁与角速度观测链路一致性需在实机环境进一步确认。

---

### 3. 快分支短时姿态与正交解耦指标 (Phase 11 / Phase 12 Task A2 实测)

评测彻底排除了早期研究使用的 Oracle 真值作弊基线，在 190 条全新 FinalConfirm 独立序列（57,000 帧）上，基于实际因果在线 P9 估计器状态进行多维度几何与生物力学评测。

| 评测维度 | Raw 基线 | 实际因果 P9-only | 快分支 P6-only | 最终组合 Hybrid (P9+P6) | 相比 P9 净收益 | 95% Bootstrap CI | 序列胜率 | 结论状态 |
|---|---|---|---|---|---|---|---|---|
| **世界 Swing 测地误差** | 2.052° | 2.052° | 0.700° | **0.700°** | **+1.352°** (改善 65.8%) | `[+1.200°, +1.491°]` | **96.84%** | 探索性显著改善 |
| **全姿态 3D 测地误差**   | 2.052° | 2.052° | 0.700° | **0.700°** | **+1.352°** | - | - | 显著改善 |
| **SMPL-H 下肢 FK 关节误差代理** | 21.56 mm | 21.56 mm | 9.19 mm | **9.19 mm** | **+12.37 mm** (改善 57.4%) | `[+10.9mm, +13.8mm]` | 97.4% | 最差恶化 -0.00 mm |
| **支撑期脚滑代理速度** | 443.35 mm/s | 443.35 mm/s | 431.84 mm/s | **431.84 mm/s** | **+11.51 mm/s** (减少脚滑) | `[+8.4mm/s, +14.6mm/s]` | 86.8% | 稳定非劣且改善 |
| **正交偏航串扰度**       | - | - | $< 10^{-6\circ}$ | **$< 10^{-6\circ}$** | **零串扰 (数值零)** | - | - | 理论严格保证 |

- **输入扰动与鲁棒性压力测试看板**：

| 压力扰动工况 | 世界 Swing 净收益 | 下肢 FK 误差减少 | 支撑期脚滑速度减少 | 鲁棒性结论 |
|---|---|---|---|---|
| **名义基准条件** | **+1.352°** | **+12.37 mm** | **+11.51 mm/s** | 基准性能最优 |
| **视觉高斯噪声 ($\sigma=2\text{ cm}, 3^\circ$)** | **+1.352°** | **+12.37 mm** | **+11.58 mm/s** | 无显著精度退化 |
| **视觉延迟 2 帧 (33 ms)** | **+1.352°** | **+12.37 mm** | **+10.37 mm/s** | 时序抖动鲁棒 |
| **视觉输入完全丢失 (VD Missing)** | **+1.352°** | **+12.31 mm** | **+10.05 mm/s** | 纯下肢退化自愈 |
| **传感器安装偏差 ($\pm 3^\circ$)** | **+1.352°** | **+12.37 mm** | **+11.52 mm/s** | 角度安装容差极佳 |

> [!NOTE]
> **算法指标审计与限制说明**：
> - 本组测试在合成扰动数据上表现优异，展示了快分支神经网络对高频 Swing 姿态的纠偏潜力；
> - 在短片段（5 秒）上 P9-only 与 Raw 接近，组合收益主要体现为 P6 单模型的快动作收益；
> - 运动学 FK 与脚滑基于固定骨长几何代理计算，未叠加根骨世界位移；
> - 正式组合增益与人体动捕改善将在 Phase 8 实机环境下进行闭环验证。

---

### 4. 模型体量与网络架构规格

| 架构参数 | 设计指标 / 规格 | 说明 |
|---|---|---|
| **网络主干结构** | 8 层因果流式 LSTM + 密集残差全连接头 | 严格因果时序，无未来帧反向传播泄漏 |
| **总参数量** | **1,539,472 参数 ($\sim 1.54\text{ M}$)** | 严格控制在 $< 2.0\text{ M}$ 轻量化门限内 |
| **导出 ONNX 权重体积** | **4.57 MB** (Opset 14) | 优于 $\le 8.0\text{ MB}$ 规范，便于跨平台分发 |
| **流式隐状态张量** | $h, c \in \mathbb{R}^{8 \times 1 \times 1 \times 96}$ | 内存常驻占用极小（仅 12.3 KB） |
| **输入流特征维度** | $1 \times 1 \times 124$ 单帧流式 | 68 维上身参考 + 56 维下肢 IMU 物理量 |
| **输出张量维度** | 残差 $4 \times 4$、门控 $4 \times 1$、触地 $2 \times 1$ | 包含下肢四元数残差、置信门控与脚掌触地概率 |

---

### 5. 早期历史基线与消融对照 (Phase 6 审计数据)

| 对照模型 | 评测序列数 | 预测反事实一致率 | 差值比中位数 | 漂移抑制效果 | 缺陷原因与结论 |
|---|---|---|---|---|---|
| **Model B (全量 3D 残差网络)** | 796 条 (16.3万帧) | 48.97% (开发集)<br>51.16% (新源序列) | 0.006% | 5秒短片段无可用收益 | 未解耦世界漂移与局部动作，网络陷入平均伪解 |
| **Model C (物理慢偏置局部右乘)** | 796 条 (16.3万帧) | - | - | 10°/min 漂移下**恶化 +0.19°**<br>零漂移下**恶化 +0.08°** | **致命缺陷**：将世界坐标偏航漂移错误右乘到局部坐标系；腿部静止检测将动态姿态冲刷 |
| **Phase 11/12 Yaw-Neutral 正交解耦架构** | 190 条独立序列 | **100.0%** (正交零串扰) | - | **Swing 误差降低 65.8%**<br>世界偏航由 Phase 9 独立左乘吸收 | **最终方案**：世界逆偏航左乘修正 + 局部 Swing 右乘补偿，彻底消除相互干扰 |

---

## 四、 核心交付产物索引表

### 1. 阶段评测报告与黄金哨兵 (JSON 机器凭据)
- **S2 服务端连续回放与延迟评测报告**：[`ai_corrector/artifacts/s2_replay_benchmark_report.json`](../ai_corrector/artifacts/s2_replay_benchmark_report.json)
- **A1 慢偏航长程物理审计报告**：[`ai_corrector/artifacts/phase12_a1_physics_audit_report.json`](../ai_corrector/artifacts/phase12_a1_physics_audit_report.json)
- **A2 快分支增量价值评测报告**：[`ai_corrector/artifacts/phase12_a2_p6_incremental_report.json`](../ai_corrector/artifacts/phase12_a2_p6_incremental_report.json)
- **ONNX 流式模型 ABI 导出报告**：[`ai_corrector/artifacts/cstn_v2_yaw_neutral_world.report.json`](../ai_corrector/artifacts/cstn_v2_yaw_neutral_world.report.json)
- **124D 契约跨语言黄金哨兵 Fixture**：[`ai_corrector/artifacts/fixture_124d_sentinel.json`](../ai_corrector/artifacts/fixture_124d_sentinel.json)

### 2. 核心算法、状态机与模型资产
- **世界轴偏航中立流式 ONNX 模型**：[`ai_corrector/artifacts/cstn_v2_yaw_neutral_world.onnx`](../ai_corrector/artifacts/cstn_v2_yaw_neutral_world.onnx)
- **可部署偏航卡尔曼滤波器**：[`ai_corrector/corrector/bias_estimator_v3.py`](../ai_corrector/corrector/bias_estimator_v3.py)
- **四路显式卡尔曼偏置状态机**：[`ai_corrector/corrector/bias_state_v3.py`](../ai_corrector/corrector/bias_state_v3.py)
- **偏航中立神经网络包装器**：[`ai_corrector/corrector/yaw_neutral_cstn.py`](../ai_corrector/corrector/yaw_neutral_cstn.py)
- **正交双分支合成器**：[`ai_corrector/corrector/hybrid_composer.py`](../ai_corrector/corrector/hybrid_composer.py)
- **标量加速契约转换适配器**：[`ai_corrector/corrector/contract_v3.py`](../ai_corrector/corrector/contract_v3.py)

### 3. SlimeVR 服务端核心工程组件
- **流式 ONNX 运行时推理引擎**：[`OnnxCorrectorEngine.kt`](../server/server/core/src/main/java/dev/slimevr/tracking/processor/corrector/OnnxCorrectorEngine.kt)
- **AI 修正器服务端主采样桥接**：[`CorrectorBridge.kt`](../server/server/core/src/main/java/dev/slimevr/tracking/processor/corrector/CorrectorBridge.kt)
- **修正器运行配置与影子模式控制**：[`CorrectorConfig.kt`](../server/server/core/src/main/java/dev/slimevr/tracking/processor/corrector/CorrectorConfig.kt)
- **跨语言 124D 契约测试套件**：[`CorrectorBridgeContractTest.kt`](../server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/CorrectorBridgeContractTest.kt)
- **真实 ONNX 影子模式推理测试套件**：[`CorrectorBridgeInferenceTest.kt`](../server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/CorrectorBridgeInferenceTest.kt)
- **10.8 万帧连续回放性能基准单测**：[`CorrectorBridgeReplayBenchmarkTest.kt`](../server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/CorrectorBridgeReplayBenchmarkTest.kt)

---

## 五、 工程安全边界与 Phase 8 实机演进计划

> [!IMPORTANT]
> ### 当前安全红线与状态边界
> 1. **严格只读影子模式 (Strict Shadow Mode)**：
>    当前 SlimeVR 服务端生产配置中 `enabled = false`，`shadowMode = true`。修正模块在独立线程执行单帧特征提取与 ONNX 推理，绝不向 Tracker 或 Avatar 姿态回写任何修改值。
> 2. **指标性质区分**：
>    - **工程类指标 (S0/S1/S2)**：在本地构建环境、13 项单测与 108,000 帧回放中具备高度确定性；
>    - **算法类指标 (A1/A2)**：在 AMASS 合成时序与仿真干扰下验证了算法原理可行性，但受限于离线合成环境与代理评估。
> 3. **实机验证 (Phase 8) 的核心任务**：
>    - 待实机硬件就绪后，采集真实真实 IMU 陀螺仪与加速度计原始传感器流；
>    - 接入光学动捕系统建立毫米级位置与真实关节旋转真值；
>    - 完成服务端完整生命周期 Session 释放，并评估端到端实时解算表现。
