# Phase 11 执行结果：短时姿态补偿与慢偏航估计的正交组合

> **当前状态：离线算子和评测执行完成，整体验收未通过原计划条件；H4 未执行。** H0/H1 数值与坐标检查通过。Phase 6 模型已训练，但独立短时姿态指标没有改善；组合端点收益与 Phase 9 单独使用几乎相同。机器报告的 PASS 只覆盖更窄的三项门禁，不能代表 Phase 11 全部目标通过。
> 成功实现世界竖直 Y 轴 Swing-Twist 正交分解算子、完成 Yaw-Neutral CSTN 神经网络重训与 8 路多因子消融基准评测。机器报告归档至 [`ai_corrector/artifacts/phase11_hybrid_compensation_report.json`](../../ai_corrector/artifacts/phase11_hybrid_compensation_report.json)。

---

## 核心验收指标与门禁裁定

| 验证项 | 判定门限 | 实际评测值 | 状态 | 结论说明 |
|---|---|---|---|---|
| **正交偏航串扰** | $\le 10^{-6\circ}$ (数值零) | **$< 10^{-6\circ}$** | **PASS** | 1000 次随机 3D 姿态下，快分支对世界偏航的影响恒为零 |
| **候选组合 30 分钟端点收益** | 正收益且 P6 带来短时指标增益；不得劣于 P9 | **+141.7188°**，与 P9 ZARU +141.7188°几乎相同 | **部分通过** | 终点主要收益来自 P9，未测出 P6 的增量贡献 |
| **严格零漂端点偏航** | 零漂下无端点 yaw 改动 | **0.0000°** | **有限通过** | 仅端点 yaw 指标；测试 Raw 本身为 0°且未覆盖姿态误改或传感器噪声 |
| **快分支偏航串扰度** | < 0.05° | **0.0027°** | **PASS** | 190 序列确认集上的数值串扰受控 |
| **P6 短时姿态增益** | 相对 Raw/P9-only 有正向姿态收益 | swing 误差 0.3076° → 0.3090°；geodesic 0.4430° → 0.4441° | **未通过** | 点估计略差，未提供置信区间支持收益 |
| **H4 服务端影子/实机门槛** | 离线增益通过后进入影子验证 | 未见影子或实机执行产物 | **待执行** | 不能由离线报告推定已完成 |
| **整体阶段裁定** | 全部计划门槛达成 | **P6 短时指标未改善；H4 未执行** | **未通过** | 报告总 PASS 未包含 Phase 11 计划中的独立精度增益与实机/影子条件 |

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

### 3. Task H2：Yaw-Neutral Phase 6 重训
- **网络与损失设计**：在 [`yaw_neutral_cstn.py`](../../ai_corrector/corrector/yaw_neutral_cstn.py) 中实现 `YawNeutralCSTN` 包装器与正交可微损失函数：
  $$\mathcal{L} = 10.0 \cdot \mathcal{L}_{\text{swing}} + 15.0 \cdot \mathcal{L}_{\text{twist\_pen}} + \mathcal{L}_{\text{gate}} + \mathcal{L}_{\text{contact}}$$
- **训练收敛**：在 GPU（RTX 4070）上基于 796 条连续逐帧时序（`database/d3c_continuous`）完成 15 轮训练（耗时 94.8 秒）：
  - 最终验证损失：`Val: 0.7416 (Swing: 0.0051 rad ≈ 0.29°, Twist: 0.0000)`；
  - 模型权重保存至：[`ai_corrector/artifacts/cstn_v2_yaw_neutral_best.pt`](../../ai_corrector/artifacts/cstn_v2_yaw_neutral_best.pt)。

---

## Task H3：多因子 8 路消融基准评测

评测脚本：[`execute_task_h3_ablation.py`](../../ai_corrector/corrector/execute_task_h3_ablation.py)
评测工况：10 条标称长会话（涵盖 0°, 1°, 3°, 10°/min，静坐、复杂步态、无静止窗口、VD丢失） + 2 条严格零漂会话。

### 1. 标称 10 会话综合表现 (30 分钟端点圆周累计偏航误差)

| 策略分组 | Mean $D_{\text{raw}}$ | Mean $D_{\text{corr}}$ | 净收益 $G$ | 95% Bootstrap CI | 改善率 % | 胜率 % | 最差恶化 |
|---|---|---|---|---|---|---|---|
| **Raw 基线** | $199.47^\circ$ | $199.47^\circ$ | $+0.00^\circ$ | $[0.0^\circ, 0.0^\circ]$ | 0.0% | 0.0% | $0.00^\circ$ |
| **Oracle 理想参考** | $199.47^\circ$ | $0.00^\circ$ | $+199.47^\circ$ | $[103.6^\circ, 295.3^\circ]$ | 100.0% | 100.0% | $+0.00^\circ$ |
| **Phase 9 ZARU-only** | $199.47^\circ$ | $57.76^\circ$ | **$+141.72^\circ$** | $[48.7^\circ, 244.5^\circ]$ | **71.0%** | **90.0%** | $-29.35^\circ$ |
| **Phase 9 Gait-only** | $199.47^\circ$ | $191.17^\circ$ | $+8.30^\circ$ | $[-10.4^\circ, 24.7^\circ]$ | 4.2% | 60.0% | $-14.22^\circ$ |
| **Phase 9 Full** | $199.47^\circ$ | $49.62^\circ$ | **$+149.85^\circ$** | $[61.8^\circ, 249.4^\circ]$ | **75.1%** | **90.0%** | $-21.14^\circ$ |
| **Old CSTN 诊断组** | $199.47^\circ$ | $199.46^\circ$ | $+0.01^\circ$ | $[-0.1^\circ, 0.1^\circ]$ | 0.0% | 60.0% | $-0.15^\circ$ |
| **Yaw-Neutral P6-only** | $199.47^\circ$ | $199.48^\circ$ | $-0.01^\circ$ | $[-0.0^\circ, 0.0^\circ]$ | 0.0% | 40.0% | $-0.01^\circ$ |
| **Candidate Hybrid (P9 ZARU + P6 Neutral)** | $199.47^\circ$ | $57.76^\circ$ | **$+141.72^\circ$** | $[48.7^\circ, 244.5^\circ]$ | **71.0%** | **90.0%** | $-29.35^\circ$ |
| **Full Hybrid (P9 Full + P6 Neutral)** | $199.47^\circ$ | $49.62^\circ$ | **$+149.85^\circ$** | $[61.8^\circ, 249.4^\circ]$ | **75.1%** | **90.0%** | $-21.14^\circ$ |

### 2. 严格零漂直通性安全验证 (Strict Zero-Drift, `enable_random_walk=False`)

| 策略分组 | Mean $D_{\text{raw}}$ | Mean $D_{\text{corr}}$ | 偏差 $\Delta$ | 判定结论 |
|---|---|---|---|---|
| **Raw 基线** | $0.0000^\circ$ | $0.0000^\circ$ | $+0.0000^\circ$ | 基准 |
| **Phase 9 ZARU-only** | $0.0000^\circ$ | $0.0000^\circ$ | **$+0.0000^\circ$** | **绝对安全** |
| **Phase 9 Full** | $0.0000^\circ$ | $0.0000^\circ$ | **$+0.0000^\circ$** | **绝对安全** |
| **Yaw-Neutral P6-only** | $0.0000^\circ$ | $0.0000^\circ$ | **$+0.0000^\circ$** | **绝对安全** |
| **Candidate Hybrid** | $0.0000^\circ$ | $0.0000^\circ$ | **$+0.0000^\circ$** | **绝对安全** |
| **Full Hybrid** | $0.0000^\circ$ | $0.0000^\circ$ | **$+0.0000^\circ$** | **绝对安全** |
| **Old CSTN 诊断组** | $0.0000^\circ$ | $0.2192^\circ$ | **$+0.2192^\circ$** | **不安全 (发生误改漂移)** |

### 3. 独立确认集 (FinalConfirm 190 序列) 短时姿态与偏航串扰

- **Raw 基线摆动误差**：$0.3076^\circ$
- **Yaw-Neutral CSTN 预测摆动误差**：$0.3090^\circ$
- **Yaw-Neutral CSTN 世界偏航扭转串扰**：**$0.0027^\circ$**（完全处于数值零容差内）
- **Old CSTN 世界偏航扭转串扰**：$0.2192^\circ$（存在明显偏航污染）

---

## 阶段结论与后续方向

1. **数学组合算子可行**：H0/H1 坐标与正交算子检查通过；世界 Y 轴 swing-twist 投影将 Phase 6 候选的偏航 twist 去除，单元测试报告数值串扰低于门限。这证明输出自由度可以隔离。
2. **模型精度互补尚未成立**：P6-only 的端点均值收益为 -0.0085°，95% CI 跨零、胜率 40%；短时 swing 误差从 0.3076°升至 0.3090°，geodesic 误差也略升。Candidate Hybrid 的端点均值与 P9 ZARU 基本相同，说明当前快分支没有可测的新增收益。
3. **零漂安全证据有限**：严格零漂组的 Raw 端点偏航误差本身为 0°，所以该表只能说明 yaw-neutral 算子没有在此理想输入上引入端点偏航；不能代表静态姿态无误改、带噪零漂安全或真实传感器噪声表现。
4. **机器 PASS 口径过窄**：H3 脚本将组合收益只与 P9 ZARU 做 0.5° 非劣比较，再检查 strict-zero 端点偏航和 twist 串扰；它没有把 Phase 11 已声明的 P6 短时姿态正收益列为 gate。因此本报告 verdict=PASS 不等于本计划整体验收通过。
5. **H4 未开始**：当前只具备离线算子和模拟评测证据。下一步应先分析 P6 在确认集上接近恒等映射的原因、提升非偏航目标的实际增益，并补充姿态保持/带噪零漂测试；离线精度门槛通过后才启动服务端影子记录。实机真值对照仍需 Phase 8 支持。

## 归档文件

- 机器评测报告：[`ai_corrector/artifacts/phase11_hybrid_compensation_report.json`](../../ai_corrector/artifacts/phase11_hybrid_compensation_report.json)
- 正交合成算子：[`ai_corrector/corrector/hybrid_composer.py`](../../ai_corrector/corrector/hybrid_composer.py)
- 正交单测套件：[`ai_corrector/corrector/test_h1_orthogonal_operator.py`](../../ai_corrector/corrector/test_h1_orthogonal_operator.py)
- 输入契约审计：[`ai_corrector/corrector/test_h0_coordinate_audit.py`](../../ai_corrector/corrector/test_h0_coordinate_audit.py)
- Yaw-Neutral 模型：[`ai_corrector/corrector/yaw_neutral_cstn.py`](../../ai_corrector/corrector/yaw_neutral_cstn.py)
- 模型权重：[`ai_corrector/artifacts/cstn_v2_yaw_neutral_best.pt`](../../ai_corrector/artifacts/cstn_v2_yaw_neutral_best.pt)
- 8 路消融脚本：[`ai_corrector/corrector/execute_task_h3_ablation.py`](../../ai_corrector/corrector/execute_task_h3_ablation.py)
