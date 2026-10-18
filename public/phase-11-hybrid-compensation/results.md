# Phase 11 执行结果：短时姿态补偿与慢偏航估计的正交组合

> **当前状态：机器报告中的 4 个离线门禁通过 (PASS)，结论仅适用于当前合成仿真；实机相关任务 (H4) 暂缓并入 Phase 8。** H0 坐标/特征维度单测和 H1 数学算子测试通过；真实服务端字段映射、实机 AHRS 误差分布、人体受试者隔离、真实 VD 噪声/延迟、FK 与脚滑收益仍未验证。
> 以真值残差提取的 oracle P9 Twist 作隔离基准，P9-only 世界 Swing 残差由 **2.1799° 降至 0.7261°**，净收益 **+1.4538° (66.7%)**（Bootstrap 95% CI `[+1.2911°, +1.6028°]`，序列胜率 96.8%）。评测顺序为先去 Twist，再执行世界系 P6 Swing 投影和置信门控；oracle 不代表 P9 在线估计精度。扰动使用角速度启发式，VD 输入是理想 AMASS 姿态，因此结论仅覆盖该合成任务。
> 机器评测报告归档至 [`ai_corrector/artifacts/phase11_hybrid_compensation_report.json`](../../ai_corrector/artifacts/phase11_hybrid_compensation_report.json)。

---

## 核心验收指标与门禁裁定

| 验证项 | 判定门限 | 实际评测值 | 状态 | 结论说明 |
|---|---|---|---|---|
| **正交偏航串扰** | $\le 10^{-6\circ}$ (数值零) | **$< 10^{-6\circ}$** | **PASS** | 1000 次随机 3D 姿态下，快分支对世界偏航的影响恒为零 |
| **候选组合 30 分钟端点收益** | $G > 0$ 且不得劣于 P9 ZARU (恶化 $\le 0.5^\circ$) | **+141.7191°** (改善率 71.0%, 胜率 90.0%) | **PASS** | 10 条合成会话均值；与 P9 ZARU 的 +141.7188° 基本相同，组合未显示额外长期偏航收益；最差单会话恶化为 $-29.35^\circ$ |
| **严格零漂端点偏航** | $|\Delta_{\text{zero}}| \le 0.05^\circ$ | **均值 +0.00004°** | **PASS** | 仅 2 条 `enable_random_walk=False` 合成会话的端点偏航均值；不代表全身姿态保持或实机零误改 (旧 CSTN 均值约 $+0.2194^\circ$) |
| **快分支偏航串扰度** | $< 0.05^\circ$ | **0.0027°** | **PASS** | 投影前世界 Twist 均值 0.1123°，经过世界系 Swing-Twist 投影后为 0.0027° |
| **P6 短时合成 Swing 残差** | 确认集残差减少且 95% CI 下界 $> 0$ | P9 oracle-only $2.1799^\circ \to$ Hybrid **$0.7261^\circ$** (净收益 **$+1.4538^\circ$ / 66.7%**) | **PASS** | 95% Bootstrap CI **`[+1.2911°, +1.6028°]`**，胜率 **96.8%**；这是平均逐帧 Swing 测地误差，不是 FK/MPJPE |
| **H0 服务端特征来源** | 核验实际 SlimeVR 可用字段 | 仅通过维度/坐标代码测试 | **未完成** | 未使用真实服务端数据包；当前 VD 特征由干净 AMASS 姿态直接构造 |
| **H4 服务端影子/实机门槛** | 同步硬件数据与独立真值后验收 | 未执行 | **待 Phase 8** | 没有真实设备或在线影子验证 |
| **离线机器门禁总体裁定** | 报告中 4 个数值门禁全部通过 | `verdict=PASS`，四项 gate 均为 true | **PASS（合成范围）** | 不等同于 H0–H4 全部计划验收，也不等同于部署验收 |

---

## 详细任务执行记录

### 1. Task H0：输入、坐标系与契约审计
- **契约对齐**：[`test_h0_coordinate_audit.py`](../../ai_corrector/corrector/test_h0_coordinate_audit.py) 的 6 项测试通过，覆盖 124 维布局、世界竖直 Y 轴以及 `[w,x,y,z]` 四元数与旋转矩阵往返。它是代码级契约检查，不核实真实服务端字段、时戳和传感器流映射。
- **零漂与随机游走解耦**：在 [`session_simulator_v3.py`](../../ai_corrector/corrector/session_simulator_v3.py) 中新增 `enable_random_walk` 开关，彻底将“严格零漂（0.0°/min, 0 random walk）”与“含随机游走工况”分离，建立了严格的零漂测试金标。
- **输入边界**：漂移标签不进入网络输入；但 H2 的头、胸与手臂姿态/速度来自干净 AMASS 真值并充当理想 VD 观测，年龄字段没有实际对齐延迟或测量噪声。因此尚不能称为真实在线输入验收。

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

### 3. Task H2：合成动作相关失真数据集与 Yaw-Neutral CSTN 重训
- **为何能产生增益**：原 Phase 6 数据集只注入世界偏航，快分支没有 Swing 残差目标。本轮建立了一个人为定义的动作相关 Swing 扰动，让模型有明确的合成目标；这不是实机 AHRS 误差的标定或重放。
- **数据集构建**：
  [`prepare_p11_dataset.py`](../../ai_corrector/corrector/prepare_p11_dataset.py) 从序列源构造 796 条训练、150 条验证、190 条确认序列（manifest 记录 1136 条有效序列）。扰动使用 $\theta=\mathrm{clip}(0.040\|\omega\times u\|,0,6^\circ)$，轴取 $\omega\times u$；这是确定性的启发式，未由线加速度、重力比力或真实 AHRS 输出推导/拟合。几何上将其限制为 Swing，不含 Y 轴 Twist。
- **神经网络训练**：
  修正局部残差与世界轴不一致后，在 CUDA 上从头训练 15 轮。最终 epoch Train/Val 总损失为 0.5002/0.4475，Swing/世界 Twist 分项为 0.0137/0.0022；权重保存至 [`ai_corrector/artifacts/cstn_v2_yaw_neutral_world_best.pt`](../../ai_corrector/artifacts/cstn_v2_yaw_neutral_world_best.pt)。

---

## Task H3：多因子 8 路消融基准与独立未见集评测

评测脚本：[`execute_task_h3_ablation.py`](../../ai_corrector/corrector/execute_task_h3_ablation.py)  
评测工况：10 条标准合成会话（每条均启用随机游走，即使 drift tier 为 0°/min）+ 2 条关闭随机游走的严格零漂合成会话 + FinalConfirm 190 条序列（同一来源子集内按文件序号隔离）。它们不是受试者隔离或实机确认集。

### 1. 标称 10 会话综合表现 (30 分钟端点圆周累计偏航误差)

| 策略分组 | Mean $D_{\text{raw}}$ | Mean $D_{\text{corr}}$ | 净收益 $G$ | 95% Bootstrap CI | 改善率 % | 胜率 % | 最差恶化 |
|---|---|---|---|---|---|---|---|
| **Raw 基线** | $199.47^\circ$ | $199.47^\circ$ | $+0.00^\circ$ | $[0.0^\circ, 0.0^\circ]$ | 0.0% | 0.0% | $0.00^\circ$ |
| **Oracle 理想参考** | $199.47^\circ$ | $0.00^\circ$ | $+199.47^\circ$ | $[103.6^\circ, 295.3^\circ]$ | 100.0% | 100.0% | $+0.00^\circ$ |
| **Phase 9 ZARU-only** | $199.47^\circ$ | $57.76^\circ$ | **$+141.72^\circ$** | $[48.7^\circ, 244.5^\circ]$ | **71.0%** | **90.0%** | $-29.35^\circ$ |
| **Phase 9 Gait-only** | $199.47^\circ$ | $191.16^\circ$ | $+8.31^\circ$ | $[-10.4^\circ, 24.7^\circ]$ | 4.2% | 60.0% | $-48.05^\circ$ |
| **Phase 9 Full** | $199.47^\circ$ | $49.62^\circ$ | **$+149.85^\circ$** | $[61.8^\circ, 249.4^\circ]$ | **75.1%** | **90.0%** | $-2.85^\circ$ |
| **Old CSTN 诊断组** | $199.47^\circ$ | $199.46^\circ$ | $+0.01^\circ$ | $[-0.1^\circ, 0.1^\circ]$ | 0.0% | 60.0% | $-0.15^\circ$ |
| **Yaw-Neutral P6-only** | $199.47^\circ$ | $199.48^\circ$ | $-0.01^\circ$ | $[-0.0^\circ, 0.0^\circ]$ | 0.0% | 40.0% | $-0.01^\circ$ |
| **Candidate Hybrid (P9 ZARU + P6 Neutral)** | $199.47^\circ$ | $57.75^\circ$ | **$+141.72^\circ$** | $[48.7^\circ, 244.5^\circ]$ | **71.0%** | **90.0%** | $-29.35^\circ$ |
| **Full Hybrid (P9 Full + P6 Neutral)** | $199.47^\circ$ | $49.62^\circ$ | **$+149.85^\circ$** | $[61.8^\circ, 249.4^\circ]$ | **75.1%** | **90.0%** | $-2.85^\circ$ |

### 2. 严格零漂直通性安全验证 (Strict Zero-Drift, `enable_random_walk=False`)

| 策略分组 | Mean $D_{\text{raw}}$ | Mean $D_{\text{corr}}$ | 偏差 $\Delta$ | 判定结论 |
|---|---|---|---|---|
| **Raw 基线** | $0.0000^\circ$ | $0.0000^\circ$ | $+0.0000^\circ$ | 基准 |
| **Phase 9 ZARU-only** | $0.0000^\circ$ | $0.0000^\circ$ | **$+0.0000^\circ$** | **绝对安全** |
| **Phase 9 Full** | $0.0000^\circ$ | $0.0045^\circ$ | **$+0.0045^\circ$** | **微幅偏移** |
| **Yaw-Neutral P6-only** | $0.0000^\circ$ | $0.0000^\circ$ | **$+0.0000^\circ$** | **绝对安全** |
| **Candidate Hybrid** | $0.0000^\circ$ | $0.0000^\circ$ | **$+0.0000^\circ$** | **绝对安全** |
| **Full Hybrid** | $0.0000^\circ$ | $0.0045^\circ$ | **$+0.0045^\circ$** | **安全** |
| **Old CSTN 诊断组** | $0.0000^\circ$ | $0.2194^\circ$ | **$+0.2194^\circ$** | **不安全 (发生误改漂移)** |

### 3. 独立确认集 (FinalConfirm 190 序列) 短时姿态补偿显著性表现

在每个 AMASS 子集内从未参与训练的文件序号 36~40 上评估（未保证 subject-level split）。基线是从标签中移除 Twist 的 P9 oracle-only；组合输出经过 P9 姿态下的世界 Swing 投影和置信门控：
- **P9 oracle-only 剩余世界 Swing 误差**：$2.1799^\circ$
- **Oracle P9 + Yaw-Neutral CSTN 组合误差**：**$0.7261^\circ$**
- **短时 Swing 误差削减净收益**：**$+1.4538^\circ$（改善率 66.7%）**
- **95% Bootstrap 置信区间**：**`[+1.2911°, +1.6028°]`**（下界显著大于 0，统计显著性达成）
- **序列胜率**：**96.8%**（190 条序列中 184 条取得正收益）
- **世界竖直偏航串扰**：**$0.0027^\circ$**（低于本轮 $0.05^\circ$ 门限）
- 此指标是平均逐帧 swing 残差角，不是 FK/MPJPE、脚滑或真实 VD/SlimeVR 端到端姿态误差。

---

## 阶段结论

1. **合成离线实验结论**：
   - P9 ZARU 在当前 10 条合成 30 分钟会话上均值改善 +141.7188°；候选组合为 +141.7191°，两者几乎相同，长时偏航收益来自 P9 分支，P6 没有可见的额外端点偏航收益。短时确认集以 oracle P9 Twist 作基线，真实 P9 估计状态下的短时增益尚未测量。
   - Yaw-Neutral P6 将指定启发式合成 Swing 扰动的平均残差降低 66.7%。这证明模型能拟合该构造的噪声目标；是否匹配真实 IMU/AHRS 错误仍未知。
   - Swing-Twist 单测证明快分支数学输出不注入世界 Y 轴 Twist；报告中的两条严格零漂会话只验收端点偏航，不能代替全姿态保持/误修改测试。
2. **任务 H4 影子模式与实机执行状态**：
   - 按照当前项目推进安排，实机采集与在线部署暂缓，待后续硬件就绪后在 Phase 8 统一对接实机动捕真值。

## 剩余验证缺口与下一步

1. 用同步记录的 gyro、加速度计比力与板载融合姿态，或受控 AHRS 回放，标定运动相关 swing 误差轴、幅值和时间响应；替换当前 `0.040 * ||omega × up||` 启发式后重新训练与评测。
2. 把确认集中的 oracle P9 Twist 替换为实际 P9 估计状态，复测短时 Swing 增益；在输入中加入 VD 姿态/速度噪声、真实时间戳延迟、丢帧和安装误差，并设置 subject-level 与 source-level 隔离。
3. 增加全姿态姿态保持、FK/MPJPE、脚底滑移与端到端延迟指标；H0 需核对 SlimeVR 运行时可用字段，H4 仍等待 Phase 8 的实机真值。

---

## 归档文件

- 机器评测报告：[`ai_corrector/artifacts/phase11_hybrid_compensation_report.json`](../../ai_corrector/artifacts/phase11_hybrid_compensation_report.json)
- 正交合成算子：[`ai_corrector/corrector/hybrid_composer.py`](../../ai_corrector/corrector/hybrid_composer.py)
- 正交单测套件：[`ai_corrector/corrector/test_h1_orthogonal_operator.py`](../../ai_corrector/corrector/test_h1_orthogonal_operator.py)
- 输入契约审计：[`ai_corrector/corrector/test_h0_coordinate_audit.py`](../../ai_corrector/corrector/test_h0_coordinate_audit.py)
- Yaw-Neutral 模型：[`ai_corrector/corrector/yaw_neutral_cstn.py`](../../ai_corrector/corrector/yaw_neutral_cstn.py)
- 世界轴投影测试：[`ai_corrector/corrector/test_world_yaw_neutral_cstn.py`](../../ai_corrector/corrector/test_world_yaw_neutral_cstn.py)
- 动态失真数据集生成：[`ai_corrector/corrector/prepare_p11_dataset.py`](../../ai_corrector/corrector/prepare_p11_dataset.py)
- 模型权重：[`ai_corrector/artifacts/cstn_v2_yaw_neutral_world_best.pt`](../../ai_corrector/artifacts/cstn_v2_yaw_neutral_world_best.pt)
- 8 路消融脚本：[`ai_corrector/corrector/execute_task_h3_ablation.py`](../../ai_corrector/corrector/execute_task_h3_ablation.py)
