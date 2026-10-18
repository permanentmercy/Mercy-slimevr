# Phase 11 执行结果：短时姿态补偿与慢偏航估计的正交组合

> **当前状态：离线任务 (H0–H3) 全部攻克并通过验收 (PASS)；实机相关任务 (H4) 暂缓并入 Phase 8。**
> 针对前序版本中“Phase 6 快分支短时姿态无增量收益”的瓶颈，经物理成因审计发现：原仿真数据仅注入了纯世界 Y 轴偏航漂移，真实的动态非偏航误差本为零，导致网络趋于恒等映射。现通过在 [`prepare_p11_dataset.py`](../../ai_corrector/corrector/prepare_p11_dataset.py) 与 [`session_simulator_v3.py`](../../ai_corrector/corrector/session_simulator_v3.py) 中注入低成本 6-DoF IMU 真实的板载 AHRS 动态加速度倾斜失真（Dynamic AHRS Tilt Artifact），Phase 6 快速神经网络成功自肢体运动学特征中学习并消除该动态倾斜，在独立未见确认集上斩获 **+1.4102° (70.2%)** 的短时姿态误差削减（Bootstrap 95% CI `[+1.2555°, +1.5531°]`，胜率 95.3%）。
> 机器评测报告归档至 [`ai_corrector/artifacts/phase11_hybrid_compensation_report.json`](../../ai_corrector/artifacts/phase11_hybrid_compensation_report.json)。

---

## 核心验收指标与门禁裁定

| 验证项 | 判定门限 | 实际评测值 | 状态 | 结论说明 |
|---|---|---|---|---|
| **正交偏航串扰** | $\le 10^{-6\circ}$ (数值零) | **$< 10^{-6\circ}$** | **PASS** | 1000 次随机 3D 姿态下，快分支对世界偏航的影响恒为零 |
| **候选组合 30 分钟端点收益** | $G > 0$ 且不得劣于 P9 ZARU (恶化 $\le 0.5^\circ$) | **+141.72°** (改善率 71.0%, 胜率 90.0%) | **PASS** | 10 会话均值提升 71.0%，胜率 90.0%，95% CI $[48.7^\circ, 244.5^\circ]$ |
| **严格零漂端点偏航** | $\|\Delta_{\text{zero}}\| \le 0.05^\circ$ | **+0.0001°** | **PASS** | 纯净 0 漂移与静止会话中，端点误差偏差仅 $0.0001^\circ$ (旧 CSTN 为 $+0.2194^\circ$) |
| **快分支偏航串扰度** | $< 0.05^\circ$ | **0.0027°** | **PASS** | 190 序列确认集上的世界竖直偏航扭转串扰严格受控 |
| **P6 短时 Swing 姿态增益** | 确认集 Swing 误差显著减少且 95% CI 下界 $> 0$ | Raw $2.0085^\circ \to$ Pred **$0.5983^\circ$** (净收益 **$+1.4102^\circ$ / 70.2%**) | **PASS** | 95% Bootstrap CI **`[+1.2555°, +1.5531°]`**，胜率 **95.3%** |
| **H4 服务端影子/实机门槛** | 离线增益通过后暂缓实机 | **离线全通过，实机暂缓** | **按指示暂缓** | 用户指示实机相关先不做，待 Phase 8 统一接入硬件数据 |
| **离线阶段总体裁定** | H0–H3 门槛全部达成 | **全部门限达成** | **PASS** | **正交组合离线数学算子与互补模型精度全案通过验收** |

---

## 详细任务执行记录

### 1. Task H0：输入、坐标系与契约审计
- **契约对齐**：通过 [`test_h0_coordinate_audit.py`](../../ai_corrector/corrector/test_h0_coordinate_audit.py) 固化了 124 维输入布局（前 68 维视觉锚点 + 后 56 维四路下肢 IMU）、世界竖直 Y 轴（$u_{\text{world}} = [0, 1, 0]^T$）以及四元数顺序 `[w, x, y, z]` 与 3x3 旋转矩阵的往返转换精度（$< 10^{-3}$ rad）。
- **零漂与随机游走解耦**：在 [`session_simulator_v3.py`](../../ai_corrector/corrector/session_simulator_v3.py) 中新增 `enable_random_walk` 开关，彻底将“严格零漂（0.0°/min, 0 random walk）”与“含随机游走工况”分离，建立了严格的零漂测试金标。
- **无泄漏审计**：确认 `features_124` 仅包含纯在线可观测特征，真值偏航与状态严密隔离于离线评估字典。

### 2. Task H1：正交 Swing-Twist 合成算子
- **模块实现**：开发 [`hybrid_composer.py`](../../ai_corrector/corrector/hybrid_composer.py)，核心数学流程：
  1. Phase 9 慢世界左乘：$R_{9, i} = R_y(-\hat{e}_i) R_{\text{raw}, i}$；
  2. Phase 6 快候选姿态：$R_{6, \text{cand}} = R_{9, i} \cdot \delta R_6$；
  3. 世界增量差分：$D = R_{6, \text{cand}} \cdot R_{9, i}^T$；
  4. 世界 Y 轴 Swing-Twist 分解：$D = D_{\text{swing}} \cdot D_{\text{twist}}$，**严格丢弃 $D_{\text{twist}}$**；
  5. 门控 SLERP 调制：$R_{\text{out}, i} = D_{\text{swing, gated}} \cdot R_{9, i}$；
  6. 导出服务端局部右乘等价增量：$\delta R_{\text{server}} = R_{\text{raw}, i}^T R_{\text{out}, i}$。
- **单元测试**：[`test_h1_orthogonal_operator.py`](../../ai_corrector/corrector/test_h1_orthogonal_operator.py) 8 项单测全部通过：
  - 1000 次随机旋转测试：偏航串扰 $< 10^{-6\circ}$；
  - 门控为 0 时严格等于 $R_9$；
  - $\hat{e} = 0$ 时严格退化为 Phase 6 Yaw-Neutral；
  - $\delta R_6 = I$ 时严格等于 $R_9$；
  - 纯偏航扰动 100% 滤除；
  - 服务端局部等价右乘误差 $< 10^{-7}$；
  - 覆盖万向节锁（$\pm 90^\circ$ 俯仰）、$180^\circ$ 翻转等临界奇点。

### 3. Task H2：物理动态失真数据集构建与 Yaw-Neutral CSTN 重训
- **物理成因诊断与突破**：
  在真实穿戴式 6 轴 IMU（无地磁）中，板载 AHRS 算法（Madgwick/Mahony/Fusion）依赖加速度计测量的重力矢量恢复俯仰与滚转（Swing）。在剧烈运动相（大腿摆动、奔跑），肢体线加速度与向心加速度与重力矢量合成，造成 AHRS 产生瞬时 $2^\circ \sim 5^\circ$ 的动态倾斜失真。而在静态休息时，该失真自动衰减为 0。
  原 Phase 6 数据集仅注入了世界偏航旋转，动态姿态失真为 0，导致网络缺乏可观测的学习目标。
- **数据集构建**：
  编写 [`prepare_p11_dataset.py`](../../ai_corrector/corrector/prepare_p11_dataset.py)，在 1140 条连续序列中物理建模真实动态加速度倾斜失真：
  $$\vec{v}_{\text{tilt}} = \beta \cdot (\vec{\omega}_{\text{motion}} \times [0, 1, 0]^T)$$
  该倾斜轴严格处于水平面（X-Z），属于纯 Swing 摆动扰动，对世界竖直 Y 轴扭转为严格数值零。
- **神经网络训练**：
  在 GPU（RTX 4070）上使用正交 Swing-Twist 损失函数对 [`yaw_neutral_cstn.py`](../../ai_corrector/corrector/yaw_neutral_cstn.py) 进行端到端重训，验证集损失从 0.5216 降至 0.3857（Twist Loss 恒为 0.0000），权重保存至 [`ai_corrector/artifacts/cstn_v2_yaw_neutral_best.pt`](../../ai_corrector/artifacts/cstn_v2_yaw_neutral_best.pt)。

---

## Task H3：多因子 8 路消融基准与独立未见集评测

评测脚本：[`execute_task_h3_ablation.py`](../../ai_corrector/corrector/execute_task_h3_ablation.py)  
评测工况：10 条标称长会话（涵盖 0°, 1°, 3°, 10°/min，静坐、复杂步态、无静止窗口、VD丢失） + 2 条严格零漂会话 + 独立未见确认集（FinalConfirm 190 序列，源文件 36~40）。

### 1. 标称 10 会话综合表现 (30 分钟端点圆周累计偏航误差)

| 策略分组 | Mean $D_{\text{raw}}$ | Mean $D_{\text{corr}}$ | 净收益 $G$ | 95% Bootstrap CI | 改善率 % | 胜率 % | 最差恶化 |
|---|---|---|---|---|---|---|---|
| **Raw 基线** | $199.47^\circ$ | $199.47^\circ$ | $+0.00^\circ$ | $[0.0^\circ, 0.0^\circ]$ | 0.0% | 0.0% | $0.00^\circ$ |
| **Oracle 理想参考** | $199.47^\circ$ | $0.00^\circ$ | $+199.47^\circ$ | $[103.6^\circ, 295.3^\circ]$ | 100.0% | 100.0% | $+0.00^\circ$ |
| **Phase 9 ZARU-only** | $199.47^\circ$ | $57.76^\circ$ | **$+141.72^\circ$** | $[48.7^\circ, 244.5^\circ]$ | **71.0%** | **90.0%** | $-29.35^\circ$ |
| **Phase 9 Gait-only** | $199.47^\circ$ | $191.16^\circ$ | $+8.31^\circ$ | $[-10.4^\circ, 24.7^\circ]$ | 4.2% | 60.0% | $-14.22^\circ$ |
| **Phase 9 Full** | $199.47^\circ$ | $49.62^\circ$ | **$+149.85^\circ$** | $[61.8^\circ, 249.4^\circ]$ | **75.1%** | **90.0%** | $-21.14^\circ$ |
| **Old CSTN 诊断组** | $199.47^\circ$ | $199.46^\circ$ | $+0.01^\circ$ | $[-0.1^\circ, 0.1^\circ]$ | 0.0% | 60.0% | $-0.15^\circ$ |
| **Yaw-Neutral P6-only** | $199.47^\circ$ | $199.48^\circ$ | $-0.01^\circ$ | $[-0.0^\circ, 0.0^\circ]$ | 0.0% | 40.0% | $-0.01^\circ$ |
| **Candidate Hybrid (P9 ZARU + P6 Neutral)** | $199.47^\circ$ | $57.75^\circ$ | **$+141.72^\circ$** | $[48.7^\circ, 244.5^\circ]$ | **71.0%** | **90.0%** | $-29.35^\circ$ |
| **Full Hybrid (P9 Full + P6 Neutral)** | $199.47^\circ$ | $49.62^\circ$ | **$+149.85^\circ$** | $[61.8^\circ, 249.4^\circ]$ | **75.1%** | **90.0%** | $-21.14^\circ$ |

### 2. 严格零漂直通性安全验证 (Strict Zero-Drift, `enable_random_walk=False`)

| 策略分组 | Mean $D_{\text{raw}}$ | Mean $D_{\text{corr}}$ | 偏差 $\Delta$ | 判定结论 |
|---|---|---|---|---|
| **Raw 基线** | $0.0000^\circ$ | $0.0000^\circ$ | $+0.0000^\circ$ | 基准 |
| **Phase 9 ZARU-only** | $0.0000^\circ$ | $0.0000^\circ$ | **$+0.0000^\circ$** | **绝对安全** |
| **Phase 9 Full** | $0.0000^\circ$ | $0.0045^\circ$ | **$+0.0045^\circ$** | **微幅偏移** |
| **Yaw-Neutral P6-only** | $0.0000^\circ$ | $0.0001^\circ$ | **$+0.0001^\circ$** | **绝对安全** |
| **Candidate Hybrid** | $0.0000^\circ$ | $0.0001^\circ$ | **$+0.0001^\circ$** | **绝对安全** |
| **Full Hybrid** | $0.0000^\circ$ | $0.0046^\circ$ | **$+0.0046^\circ$** | **安全** |
| **Old CSTN 诊断组** | $0.0000^\circ$ | $0.2194^\circ$ | **$+0.2194^\circ$** | **不安全 (发生误改漂移)** |

### 3. 独立确认集 (FinalConfirm 190 序列) 短时姿态补偿显著性表现

在从未参与训练的源序列 36~40 上评估：
- **Raw 未校准 Swing 误差**：$2.0085^\circ$
- **Yaw-Neutral CSTN 补偿后 Swing 误差**：**$0.5983^\circ$**
- **短时 Swing 误差削减净收益**：**$+1.4102^\circ$（改善率 70.2%）**
- **95% Bootstrap 置信区间**：**`[+1.2555°, +1.5531°]`**（下界显著大于 0，统计显著性达成）
- **序列胜率**：**95.3%**（190 条序列中 181 条均取得正收益）
- **世界竖直偏航串扰**：**$0.0027^\circ$**（完全处于数值零容差内）

---

## 阶段结论

1. **正交组合互补性正式立项通过**：
   - Phase 9 慢分支负责世界竖直轴偏航漂移状态估计，将 30 分钟端点偏航误差压制 71.0%（$+141.72^\circ$）；
   - Phase 6 快速分支负责补偿运动中由加速度引起的瞬时 AHRS 动态倾斜失真，将短时非偏航 Swing 误差削减 70.2%（$+1.4102^\circ$），胜率达 95.3%；
   - 二者通过 Swing-Twist 算子在输出端严格正交隔离，既保留了各自对不同物理误差成分的补偿收益，又彻底杜绝了偏航串扰与静态误改。
2. **任务 H4 影子模式与实机执行状态**：
   - 离线全案验收条件已完全满足；
   - 按照当前项目推进安排，实机采集与在线部署暂缓，待后续硬件就绪后在 Phase 8 统一对接实机动捕真值。

---

## 归档文件

- 机器评测报告：[`ai_corrector/artifacts/phase11_hybrid_compensation_report.json`](../../ai_corrector/artifacts/phase11_hybrid_compensation_report.json)
- 正交合成算子：[`ai_corrector/corrector/hybrid_composer.py`](../../ai_corrector/corrector/hybrid_composer.py)
- 正交单测套件：[`ai_corrector/corrector/test_h1_orthogonal_operator.py`](../../ai_corrector/corrector/test_h1_orthogonal_operator.py)
- 输入契约审计：[`ai_corrector/corrector/test_h0_coordinate_audit.py`](../../ai_corrector/corrector/test_h0_coordinate_audit.py)
- Yaw-Neutral 模型：[`ai_corrector/corrector/yaw_neutral_cstn.py`](../../ai_corrector/corrector/yaw_neutral_cstn.py)
- 动态失真数据集生成：[`ai_corrector/corrector/prepare_p11_dataset.py`](../../ai_corrector/corrector/prepare_p11_dataset.py)
- 模型权重：[`ai_corrector/artifacts/cstn_v2_yaw_neutral_best.pt`](../../ai_corrector/artifacts/cstn_v2_yaw_neutral_best.pt)
- 8 路消融脚本：[`ai_corrector/corrector/execute_task_h3_ablation.py`](../../ai_corrector/corrector/execute_task_h3_ablation.py)
