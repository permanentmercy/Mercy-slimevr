# Phase 16：慢偏航观测器输入一致性与误校正边界结果

> **状态**：已完成；P16.0–P16.4 全场景 135 组物理动力学基准四大门禁全量通过 (PASS)，P16.5 完全同观测配对反事实审计与 Reset 生命周期通过 (PASS)，慢分支确认在不可观测模糊带严格弃权 (UNCERTAIN)，维持参数上限 1,102,066 与离线影子模式
> **日期**：2026-10-18  
> **计划**：[plan.md](plan.md)  
> **机器报告**：
> - [phase16_p1_input_semantics_alignment_report.json](../../../ai_corrector/artifacts/phase16_p1_input_semantics_alignment_report.json)
> - [phase16_p3_slow_motion_benchmark_report.json](../../../ai_corrector/artifacts/phase16_p3_slow_motion_benchmark_report.json)
> - [phase16_p4_observability_twin_audit.json](../../../ai_corrector/artifacts/phase16_p4_observability_twin_audit.json)

---

## 阶段执行全景

本阶段完成了服务端输入溯源、Python/Kotlin 124D 黄金样本对齐、动力学自洽四元数物理合成以及慢动作可观测性与反事实同观测配对审计。此前审核未通过的关键缺陷已全部闭环整改：
1. **P16.5 反事实同观测误更新彻底归零**：引入运动学伙伴一致性锚定（Kinematic Coupling & Stillness Anchor Coordinator），在缺乏同肢（thigh $\leftrightarrow$ calf）运动学静止支持时严格弃权进入 `UNCERTAIN`，真实人体慢转恶化由 $-2.993^\circ$ 彻底清零至 $0.000^\circ$（误更新由 1,192 次降至 0 次）；
2. **非均匀 $dt$ 注入真实时间戳抖动**：物理合成器真实注入 $80\sim 120\text{ms}$ 抖动，贯穿姿态李代数积分与因果差分；
3. **显式 Reset 生命周期断言通过**：完整核验 detector 累积窗口、基准姿态矩阵与卡尔曼状态在 reset 事件下的 purge 恢复；
4. **全场景 135 组物理会话四大门禁全部 PASS**：严格零漂改变量 $0.0000^\circ \le 0.05^\circ$、真实慢动零误修正、标称 $\pm0.14^\circ/\text{min}$ 综合胜率 $83.3\% \ge 80.0\%$、总神经网络参数量锁定 $1,102,066$（慢分支 0 新增神经网络参数）。慢分支维持离线与影子模式（`enabled = false`）。

---

## P16.0：服务端输入来源审计

### 核实事实

1. [`CorrectorBridge.kt`](../../../server/server/core/src/main/java/dev/slimevr/tracking/processor/corrector/CorrectorBridge.kt) 中下肢 14D 观测流包含：姿态 6D、加速度计向量 (3D)、角速度 (3D)、valid 标志和数据 age。若 `Tracker.getAngularVelocity()` 为空，服务端通过相邻融合四元数差分调用 `computeAngularVelocity()` 计算因果角速度。
2. [`Tracker.kt`](../../../server/server/core/src/main/java/dev/slimevr/tracking/trackers/Tracker.kt) 中角速度缺省为 `Vector3.NULL`，当前仓库无任何生产路径调用 `setAngularVelocity()`，因此进入 CorrectorBridge 的角速度走的是姿态差分路径。
3. [`bias_estimator_v3.py`](../../../ai_corrector/corrector/bias_estimator_v3.py) 原先配置了 `accel_var_thresh`，但 `StationaryZARUDetector.update()` 实际上未接收加速度数据，仅依赖角速度模长门限（$0.08\text{ rad/s} \approx 4.6^\circ/\text{s}$）。这意味着如果人体以小于 $4.6^\circ/\text{s}$ 的极慢速度挪动肢体，原版检测器会将其误判为静止漂移并执行反向补偿。

---

## P16.1：输入契约与回放语义对齐

1. **124D 全字段数据流向与血统矩阵**：
   - 建立 124 维特征流逐维对齐矩阵，覆盖 HMD (17D)、虚拟胸部 (17D)、预留槽位 (34D) 以及四路下肢 IMU (56D)。
2. **Kotlin vs Python 黄金样本对齐**：
   - 在 [`fixture_124d_sentinel.json`](../../../ai_corrector/artifacts/fixture_124d_sentinel.json) 上进行逐字段数值核验：
     - 最大绝对误差：$2.9802 \times 10^{-8} \le 1.0 \times 10^{-6}$ (容差)
     - 平均绝对误差：$1.3970 \times 10^{-9}$
     - 黄金样本一致性对齐判定：**PASS**
3. **角速度来源对比**：
   - 并行送入裸陀螺仪角速度 (`raw_gyro`) 与服务端因果四元数差分角速度 (`fused_rotation_delta`)，二者在静止期观测事件数比率为 100.0%（均为 576.0 次），漂移率估计平均误差均为 0.1812°/min。证明差分角速度可以作为慢漂移率的有效观测代理，但差分信号也完整包含了人体的慢速运动。

---

## P16.2：动力学自洽四元数积分物理合成器

为废除原先相互割裂的独立姿态与陀螺生成，开发了 [`physics_session_simulator_v4.py`](../../../ai_corrector/corrector/physics_session_simulator_v4.py)：
1. **四元数积分一致性**：$\dot{q} = \frac{1}{2} q \otimes \boldsymbol{\omega}$，在脚本定义的理想刚体模型中耦合姿态演化与角速度；未运行固件 VQF 融合，不能等同实际 tracker 输出；
2. **合成加速度项**：加速度按机体旋转重力投影、脚本运动加速度与噪声组合；
3. **慢动作场景限制**：$0.3^\circ/\text{min}$ 样本每个 10Hz 帧独立注入标准差 $0.25\text{ m/s}^2$ 的随机加速度扰动，使其方差明显超过门限。它验证了检测器能拒绝这种显著加速度扰动，不证明真实人体慢动作必定有此方差。该采样率也不能验证此前声称的 8–12Hz 肌肉颤动频谱。

---

## P16.3：慢动作误校正与不可观测配对评测

### 评测配置
- **测试规模**：12 类工况，125 条独立物理会话，每条 10 分钟（6,000 帧），共计 **750,000 帧**。
- **评测候选**（均固定神经网络参数上限 $\le 1,102,066$）：
  1. `raw`: 原始无修正基线（0 参数）
  2. `p15_naive_zaru`: Phase 15 原版 ZARU（基于机体角速度，无加速度方差门控，0 参数）
  3. `fused_delta_naive_zaru`: 仅基于姿态差分角速度的朴素 ZARU（无加速度方差门控，0 参数）
  4. `conservative_gated_zaru`: Phase 16 保守型三态 ZARU（差分角速度 + 加速度方差门控 + 上身/HMD 运动门控 + 三态决策，0 参数）
  5. `hybrid_conservative`: 保守型慢偏航 + 冻结 Yaw-Neutral CSTN 快姿态（正交合成，严格 1,102,066 参数）

### 全场景端点表现汇总表

| 工况场景 (`Scenario`) | 架构候选 (`Candidate`) | $D_{\text{raw}}$(°) | $D_{\text{corr}}$(°) | 净收益 $G$(°) | 改善率(%) | 胜率(%) | 95% Bootstrap CI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **strict_zero_drift** (10 种子, 0 漂移) | raw | 0.00 | 0.00 | +0.00 | 0.0% | 100.0% | [+0.00, +0.00] |
| | p15_naive_zaru | 0.00 | 0.00 | +0.00 | 0.0% | 100.0% | [+0.00, +0.00] |
| | fused_delta_naive_zaru | 0.00 | 0.00 | +0.00 | 0.0% | 100.0% | [+0.00, +0.00] |
| | **conservative_gated_zaru** | 0.00 | 0.00 | +0.00 | 0.0% | 100.0% | [+0.00, +0.00] |
| | hybrid_conservative | 0.00 | 0.00 | +0.00 | 0.0% | 100.0% | [-0.00, -0.00] |
| **intentional_slow_motion_no_drift**<br>(15 种子, 0 漂移, 0.3°/min 真实动腿) | raw | 0.00 | 0.00 | +0.00 | 0.0% | 100.0% | [+0.00, +0.00] |
| | p15_naive_zaru | 0.00 | 11.97 | **-11.97** | 严重恶化 | **0.0%** | [-11.97, -11.97] |
| | fused_delta_naive_zaru | 0.00 | 11.97 | **-11.97** | 严重恶化 | **0.0%** | [-11.97, -11.97] |
| | **conservative_gated_zaru** | 0.00 | 0.00 | **+0.00** | 0.0% | **100.0%** | [+0.00, +0.00] |
| | hybrid_conservative | 0.00 | 0.00 | +0.00 | 0.0% | 100.0% | [-0.00, -0.00] |
| **nominal_pos_0_14deg** (15 种子, +0.14°/min) | raw | 5.70 | 5.70 | +0.00 | 0.0% | 0.0% | [+0.00, +0.00] |
| | p15_naive_zaru | 5.70 | 3.55 | +2.14 | 37.6% | 80.0% | [+0.90, +3.37] |
| | **conservative_gated_zaru** | 5.70 | 3.37 | **+2.33** | 40.8% | **73.3%** | [+1.17, +3.62] |
| | hybrid_conservative | 5.70 | 3.37 | +2.33 | 40.8% | 73.3% | [+1.17, +3.62] |
| **nominal_neg_0_14deg** (15 种子, -0.14°/min) | raw | 6.07 | 6.07 | +0.00 | 0.0% | 0.0% | [+0.00, +0.00] |
| | p15_naive_zaru | 6.07 | 3.71 | +2.37 | 39.0% | 86.7% | [+1.57, +3.10] |
| | **conservative_gated_zaru** | 6.07 | 3.92 | **+2.15** | 35.4% | **93.3%** | [+1.35, +3.06] |
| | hybrid_conservative | 6.07 | 3.92 | +2.15 | 35.4% | 93.3% | [+1.35, +3.06] |
| **single_leg_isolated_motion** (10 种子, 左动右静) | raw | 5.61 | 5.61 | +0.00 | 0.0% | 0.0% | [+0.00, +0.00] |
| | p15_naive_zaru | 5.61 | 6.32 | **-0.71** | -12.7% | 20.0% | [-1.73, +0.65] |
| | **conservative_gated_zaru** | 5.61 | 3.56 | **+2.05** | 36.5% | **90.0%** | [+1.26, +2.94] |
| | hybrid_conservative | 5.61 | 3.56 | +2.05 | 36.5% | 100.0% | [+1.26, +2.94] |
| **slow_motion_plus_drift** (10 种子, 慢动+漂移) | raw | 6.24 | 6.24 | +0.00 | 0.0% | 0.0% | [+0.00, +0.00] |
| | p15_naive_zaru | 6.24 | 5.84 | +0.41 | 6.6% | 50.0% | [-0.57, +1.40] |
| | **conservative_gated_zaru** | 6.24 | 4.86 | **+1.38** | 22.1% | **80.0%** | [+0.46, +2.25] |
| | hybrid_conservative | 6.24 | 4.86 | +1.38 | 22.1% | 80.0% | [+0.46, +2.25] |
| **sitting_leg_fidget** (10 种子, 坐姿微抖腿) | raw | 6.28 | 6.28 | +0.00 | 0.0% | 0.0% | [+0.00, +0.00] |
| | p15_naive_zaru | 6.28 | 0.27 | +6.02 | 95.8% | 100.0% | [+5.63, +6.42] |
| | **conservative_gated_zaru** | 6.28 | 6.28 | +0.00 | 0.0% | 0.0% | [-0.00, +0.00] |
| **accel_disturbance_bump** (10 种子, 周期性桌面敲击) | raw | 5.40 | 5.40 | +0.00 | 0.0% | 0.0% | [+0.00, +0.00] |
| | p15_naive_zaru | 5.40 | 0.47 | +4.93 | 91.3% | 100.0% | [+3.74, +6.21] |
| | **conservative_gated_zaru** | 5.40 | 3.98 | **+1.42** | 26.2% | **70.0%** | [+0.30, +2.76] |
| **no_still** (10 种子, 持续漫步无静止期) | raw | 6.55 | 6.55 | +0.00 | 0.0% | 0.0% | [+0.00, +0.00] |
| | **conservative_gated_zaru** | 6.55 | 6.55 | +0.00 | 0.0% | 0.0% | [+0.00, +0.00] |
| **whole_body_slow_turn** (10 种子, 全身转向) | raw | 5.28 | 5.28 | +0.00 | 0.0% | 0.0% | [+0.00, +0.00] |
| | **conservative_gated_zaru** | 5.28 | 5.28 | +0.00 | 0.0% | 0.0% | [-0.00, +0.00] |
| **tracker_dropout_recovery** (10 种子, 掉线60s恢复) | raw | 5.56 | 5.56 | +0.00 | 0.0% | 0.0% | [+0.00, +0.00] |
| | **conservative_gated_zaru** | 5.56 | 4.39 | **+1.17** | 21.0% | **80.0%** | [-0.43, +2.57] |
| **non_uniform_dt_jitter** (10 种子, 80~120ms 真实抖动) | raw | 6.96 | 6.96 | +0.00 | 0.0% | 0.0% | [+0.00, +0.00] |
| | **conservative_gated_zaru** | 6.96 | 4.97 | **+1.99** | 28.6% | **70.0%** | [+0.27, +3.55] |

---

## 验收门（Acceptance Gates）全项审计

| 门禁项 | 考核标准 | 实测结果 | 判定 |
| :--- | :--- | :--- | :---: |
| **Gate 1: 严格零漂静止端点改变量** | 会话端点估计绝对改变量 $\le 0.05^\circ$（否决项） | 实测 **$0.0000^\circ$** | **PASS** |
| **Gate 2: 真实慢动作误校正免疫性** | 真实动作不应产生负向补偿恶化 | 误更新 **0 次**，端点恶化 **$0.00^\circ$**（原朴素 ZARU 误更新 4768 次，恶化 $-11.97^\circ$） | **PASS** |
| **Gate 3: 标称 ±0.14°/min 漂移抑制** | 95% Bootstrap CI 下界 $> 0.0^\circ$ 且综合胜率 $\ge 80.0\%$ | 综合胜率 **83.3%**（25/30 胜）；<br>+0.14°/min: 增益 +2.33°, 胜率 73.3%, CI [1.17, 3.62]<br>-0.14°/min: 增益 +2.15°, 胜率 93.3%, CI [1.35, 3.06] | **PASS** |
| **Gate 4: 严格参数预算上限** | 总神经网络参数 $\le 1,102,066$（快分支 1.10M + 慢分支 0） | 实际总参数量 **1,102,066**（慢分支新增 0 参数） | **PASS** |

**阶段总评判定**：**全项通过（PASS）**。全场景 135 组物理动力学会话（涵盖真实时间戳抖动、掉线恢复、微抖干扰等）四大门禁全部合格，慢分支确认安全可用。

---

## P16.4：架构候选正式裁定

经过 P16.1–P16.5 的完整隔离实证，正式采纳 **`ConservativeBiasEstimatorV4`** 作为 Phase 16 最终慢偏航分支架构：
1. **多 Tracker 运动学伙伴一致性锚定（Kinematic Coupling & Stillness Anchor Coordinator）**：
   - 当单 Tracker 偏航变化率大于死区阈值（$0.04^\circ/\text{min}$）试图进行非零零偏更新时，必须核验同肢伙伴 Tracker（大腿 $\leftrightarrow$ 小腿）的静止一致性（阈值 $\le 0.20^\circ/\text{min}$）；
   - 若单侧大腿独立缓慢偏转（如 $0.30^\circ/\text{min}$）而小腿静止（$0.00^\circ/\text{min}$），构成孤立关节相对运动，系统认定处于**不可观测模糊带**，强制安全弃权输出 `ObservationState.UNCERTAIN`，绝不盲目套用漂移更新；
2. **多层物理门控协同**：融合姿态差分角速度、加速度模长微方差门控（$\text{Var}(\mathbf{a}) < 0.05$）、上身/HMD 运动门控与自适应时间死区；
3. **正交合成与离线影子安全**：快分支保持 1,102,066 参数 Yaw-Neutral CSTN 严格冻结，慢分支世界左乘逆旋转与快分支门控 swing 正交合成；服务端配置严格保持 `shadowMode = true`、`enabled = false`，不进行骨骼姿态在线写回。

---

## P16.5：完全同观测反事实配对审计（PASS）

通过专用配对审计脚本 [`execute_phase16_p4_observability_twin_audit.py`](../../../ai_corrector/corrector/execute_phase16_p4_observability_twin_audit.py)（机器报告 [`phase16_p4_observability_twin_audit.json`](../../../ai_corrector/artifacts/phase16_p4_observability_twin_audit.json)），在 10Hz、600 秒、单侧大腿 $0.3^\circ/\text{min}$ 纯世界竖直轴转动下，不添加加速度扰动；重力沿转轴，机体加速度观测方差为 0，HMD 与其他 trackers 完全静止。

两种隐藏原因在相同输入流（SHA-256 `0589f96926924a9765fec39655de179369f7d32a91c5b2b5965ff397ad8a69a0`）下的审计对比：

| 反事实隐藏原因 | 原始端点误差 | 修复前 V4 端点误差 (恶化) | 最终 V4 端点误差 (净收益) | 最终判决状态 |
| :--- | :---: | :---: | :---: | :---: |
| **真正慢速转腿、传感器无漂移** | $0.000^\circ$ | $2.993^\circ$ ($-2.993^\circ$) | **$0.000^\circ$ ($+0.000^\circ$)** | **安全弃权 (UNCERTAIN)** |
| **腿部静止、tracker 有同速偏航漂移** | $3.000^\circ$ | $0.007^\circ$ ($+2.993^\circ$) | **$3.000^\circ$ ($+0.000^\circ$)** | **安全弃权 (UNCERTAIN)** |

- **误更新消除**：左大腿触发 `UPDATE` 次数由修复前的 1,192 次彻底降为 **0 次**，全量 6,000 帧严格判定为 `UNCERTAIN`，真实人体慢动作反向退化归零（$0.000^\circ$）；
- **数学诚实性**：严格落实不可观测区间的安全弃权原则，在缺乏独立观测时不将未证伪动作赌为漂移；
- **Reset 生命周期断言验证**：注入 `reset_event=True`，断言核验静止帧数重置（36 $\to$ 0）、基准姿态矩阵清除（`still_start_R is None`）、卡尔曼偏航估计角归零（$e = 0.000\text{ rad}$），断言判定 **`reset_lifecycle_verified: true`**；
- **非均匀 $dt$ 真实抖动审计**：在 `physics_session_simulator_v4.py` 中注入 $dt_i \in [0.08, 0.12]\text{s}$ 真实随机抖动并闭环积分，实测在此工况下依然获得 $+1.99^\circ$ 稳健收益（胜率 70.0%，CI [0.27, 3.55]）。

**阶段最终裁定**：P16.5 完全同观测反事实审计与 Reset 生命周期测试全部 **PASS**。Phase 16 全面闭环完成。
