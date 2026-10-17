# Phase 9 执行结果

> **2026-10-17 独立复核（优先于下方 M0/M1 原执行叙述）**：几何左乘/共轭右乘的数学单测可作为工程进展；M0 的长会话与 M1 的锚点训练/五路对照**未满足可部署输入和独立长会话验收条件**。下方 `99.4%` 仅是将仿真真值加 0.5° 噪声直接喂给滤波器的理想观测上限；`-62.68°` 是有真值输入泄漏和目标定义问题的仿真结果，不能归因为“真实纯运动学锚点不可观测”，也不能据此判定新架构无效。当前权重不得在线改姿态。

## M0/M1 独立复核与裁定

1. **M1 输入泄漏，部署特征未参与实验。** [`execute_task_m1.py`](../../ai_corrector/corrector/execute_task_m1.py) 的 `extract_features_for_session()` 在第 64–69 行直接读取 `true_drift_angles_rad`，把其 `sin/cos` 放入模型输入；第 183 行测试也调用同一函数。这些真值只有仿真器/离线真值能提供，在线四颗 IMU + HMD/VD 无法获取。[`anchor_encoder_v3.py`](../../ai_corrector/corrector/anchor_encoder_v3.py) 声称的原始姿态运动学特征函数未被 M1 调用。因此三种随机种子只训练了“带泄漏输入的仿真回归器”，不能证明自推断锚点的质量。
2. **训练目标与圆周状态不一致。** M1 第 69 行用未 `wrap` 的累计漂移真值作 NLL 标签，网络输出头却用 `tanh × π` 限制在 `[-π,π]`；30 分钟且线性率在 ±15°/min 时可能累计超过 180°。此时目标超出输出范围，普通平方 NLL 也没有处理圆周角度的等价性。Seed 101/102/103 的验证 NLL 与 `-20.7%` 终点结果不能据此归因于锚点观测噪声。
3. **五路对照并非五个实际模型。** M1 第 209–213 行未加载或运行旧 CSTN 权重，而是给每路终点误差硬减 `0.0001` 弧度；“旧 CSTN 在 30 分钟无收益”这列无效。线性和直接锚点组也使用上述真值泄漏网络的输出，不能代表可部署基线。
4. **M0 的 30 分钟基准是简化仿真。** [`session_simulator_v3.py`](../../ai_corrector/corrector/session_simulator_v3.py) 只生成四路漂移曲线、固定时间间隔的锚点索引和起终点单位姿态，没有输出 30 分钟四路运动姿态、HMD/VD 流或实际可观测锚点特征。M0 滤波上限在锚点处使用 `true_drift + Gaussian(0,0.5°)`，属于真值辅助 oracle，不是模型可获得的测量。M0 manifest 只存元数据，没有固化可回放会话；M1 又以另一组种子即时再生会话。`execute_task_m0.py` 仅抽查 500/16,834 条文件长度，不能据此断言全库最大仅 300 帧。
5. **机器判定与数值相反。** [`m1_stateful_bias_model_report.json`](../../ai_corrector/artifacts/m1_stateful_bias_model_report.json) 记录完整方案 `G=-62.675°`、胜率 `20%`，但第 372/378 行将 `status="PASS"`、`stateful_filter_achieves_superior_gain=True` 固定写入。原报告保留作历史产物，**这些布尔判定撤销**。5 条纯仿真测试会话也不能满足计划中的实机长会话和跨会话置信区间门槛。

**当前可保留的结论**：旋转复合公式及随机姿态单测成立；显式状态滤波器有可运行的原型；给它近真值的高精锚点时，仿真端点指标会大幅改善。**尚未验收的结论**：模型在可部署输入下能否测到四路累计偏移、30 分钟真实会话是否有终点正收益、旧 CSTN 与新模型的有效对照。下一步按 [M1r 修复计划](plan.md)重新制作可部署输入与长会话数据；不要复用原 M1 Test 作为新候选的最终隔离验收集。

## 频域方案审查（规划完成，实验未执行）

- 用户提出的带通摆动面、静止能量门禁和长时间常数泄漏分别审查后，指定 [M1r-F](plan.md) 为“**逐颗静止段零偏估计主线 + 严格门禁的步态摆动面辅助**”。频域特征用于选择有信息量的时段，不能仅以高低频分开来恢复姿态偏航。
- 现有 v3 适配器与真实 v2 的通道顺序不符（v2：视觉 `4×17` 后接 IMU `4×14`；v3 适配器从第 0 维读 IMU 并把视觉按 18 维读取）。在修正映射和核实裸陀螺链路前，尚不能训练该频域方案；当前合成角速度/协议可选字段不能证明实机已有可用原始陀螺流。
- 本轮没有新增训练、30 分钟实机日志或频域算法收益；该方案的可行性结论以 M1r-F 消融和 M2 端点验收为准。

## 已确认的设计输入

- 当前 [`model.py`](../../ai_corrector/corrector/model.py)有逐流 LSTM 状态接口，但输出仍是四路每帧 3D 残差、gate 与接触概率，没有四颗 IMU 独立的偏航误差/速率/不确定度状态。
- 当前 [`train.py`](../../ai_corrector/corrector/train.py)从 20 帧窗口读数据、训练集打乱、每次调用模型时不传入跨窗口状态；它训练的是短时映射，不是两次 reset 之间的偏移状态估计。
- D3c [`prepare_d3c_dataset.py`](../../ai_corrector/corrector/prepare_d3c_dataset.py)按逐帧序列制作数据，但每条最长 300 帧（5 秒）；[Phase 6 结果](../phase-06-accuracy-baseline/results.md)的连续 B 在新源序列没有可用收益，不能作为分钟级训练/验收结论。
- v2 [`contract_v2.py`](../../ai_corrector/corrector/contract_v2.py)定义四路下肢与 HMD/VD 共 124 维输入；新模型已由 [`contract_v3.py`](../../ai_corrector/corrector/contract_v3.py)扩展时间戳、reset、标定和质量的时序契约。
- 旧 D3c 物理分支存在世界偏航误用右乘的问题；Phase 9 在 M0 完成了世界左乘与桥接右乘换算的独立 oracle。
- 本地 `ai_corrector/database/amass/` 目前仅见 `HumanML3D_SMPLH52` 目录；已确认无原生 30 分钟连续数据，采用物理仿真长会话作为基准。

## 本轮坐标算例与几何单测 (M0)

使用非零三轴旋转（37°、22°、-18°）作为干净姿态，在世界 Y 轴左乘 +11° 构造原始观测，再比较逆旋转：世界左乘与共轭成局部右乘均返回 0° 测地角误差（浮点精度内）；把同一世界逆偏航直接右乘则留下约 6.42°。这是独立代数算例。在 M0 自动化单测中，对 10,000 组随机 3D 姿态完成验证：
- 世界左乘修正最大误差：$2.96 \times 10^{-6\circ}$；
- 局部共轭右乘 $R_{\text{raw}}^T R_y(-\hat{e}) R_{\text{raw}}$ 等价性误差：$4.18 \times 10^{-6\circ}$；
- 错误局部直接右乘平均退化误差高达 **$90.82^\circ$**（最大 $179.99^\circ$）。

## 本轮目标口径

- 主指标为同一 ≥30 分钟会话四颗传感器在 `t0` 与 `t1` 的世界偏航误差增长绝对值之和：`D_raw/D_corr`，收益 `G = D_raw − D_corr`。只在两个端点的小区间做抗抖汇总，不平均会话中每帧误差；有符号漂移不能在四路间相互抵消。
- 结构采用“会话锚点/创新量编码器 + 四路累计偏移状态”；秒级片段只用于识别可比较的事件。`t1` 独立测量必须在再次 reset 前完成。
- 独立端点真值只作训练标签/验收基准，不进入在线模型；在线锚点必须由四路 IMU、HMD/VD 和会话初始状态等实际可用输入产生。

---

## 任务 M0 执行结果（数据与几何基础）

> **机器报告与实验产物归档**：
> - 时序契约：[`ai_corrector/corrector/contract_v3.py`](../../ai_corrector/corrector/contract_v3.py)
> - 几何 Oracle 与单测：[`ai_corrector/corrector/oracle_v3_geometry.py`](../../ai_corrector/corrector/oracle_v3_geometry.py)
> - 30 分钟会话仿真器：[`ai_corrector/corrector/session_simulator_v3.py`](../../ai_corrector/corrector/session_simulator_v3.py)
> - 会话基准数据集清单：[`ai_corrector/database/phase09_m0_sessions/m0_session_manifest.json`](../../ai_corrector/database/phase09_m0_sessions/m0_session_manifest.json)
> - 机器评测报告：[`ai_corrector/artifacts/m0_geometry_and_data_audit_report.json`](../../ai_corrector/artifacts/m0_geometry_and_data_audit_report.json)

### 1. 本地动捕数据连续性审计
- 审计对象：`ai_corrector/database/amass/HumanML3D_SMPLH52`（共 16,834 个 `.npz` 文件）。
- 抽样统计：序列长度中位数 201 帧（约 6.7 秒），最大序列仅 300 帧（10.0 秒）。
- **当时记录的结论，已在顶部复核修正**：500 个抽样文件均为短片段；尚未逐文件确认全库最大长度。M0 生成的是 30 分钟漂移曲线和预设锚点的简化仿真，没有完整的人体动作、HMD/VD 或可部署锚点观测流；不能称为已具备真实或物理连贯的长会话数据集。

### 2. M0 端点指标评测（高精锚点上限测试）
在 10 组独立的 30 分钟连续会话上，测试卡尔曼偏置状态滤波器（`QuadChannelBiasFilterV3`）在上界高精锚点下的表现：
- **Raw 原始总累计误差增长量（D_raw）**：$302.39^\circ$（四路下肢 IMU 平均每颗漂移约 $75.6^\circ$）；
- **Filter v3 纠正后总累计误差增长量（D_corr）**：$1.93^\circ$（四路平均每颗残差仅 $0.48^\circ$）；
- **净收益（G）**：$+300.46^\circ$；
- **总体改善率**：**$99.4\%$**，**会话胜率**：**$100.0\%$**；
- **错误右乘对照**：直接右乘造成严重姿态退化，证实坐标转换闭环的必要性。

---

## 任务 M1 执行结果（最小可用模型与五路基线对照）

> **机器报告与实验产物归档**：
> - 锚点编码器网络：[`ai_corrector/corrector/anchor_encoder_v3.py`](../../ai_corrector/corrector/anchor_encoder_v3.py)
> - 训练最佳权重：[`ai_corrector/artifacts/anchor_encoder_v3_best.pt`](../../ai_corrector/artifacts/anchor_encoder_v3_best.pt)
> - 执行与评测流水线：[`ai_corrector/corrector/execute_task_m1.py`](../../ai_corrector/corrector/execute_task_m1.py)
> - 机器评测报告：[`ai_corrector/artifacts/m1_stateful_bias_model_report.json`](../../ai_corrector/artifacts/m1_stateful_bias_model_report.json)

### 1. 多随机种子因果锚点网络训练
在 8 组 30 分钟长会话上跨 3 个随机种子训练 `AnchorInnovationEncoderV3`（隐藏层 64，2 层 GRU，4 个解耦输出头）：
- **Seed 101**：验证 NLL 损失 **3.4042**（最佳，保存为 `anchor_encoder_v3_best.pt`）；
- **Seed 102**：验证 NLL 损失 4.1029；
- **Seed 103**：验证 NLL 损失 3.6018。

### 2. 30 分钟连续会话五路基线对照实测矩阵
在完全未见的 5 组独立测试会长会话（每组 30 分钟，总计 150 分钟连续动作）上运行五路基线：

| 基线策略标识 | 原始误差增长 ($D_{\text{raw}}$) | 纠正后误差增长 ($D_{\text{corr}}$) | 净收益 ($G = D_{\text{raw}} - D_{\text{corr}}$) | 改善率 ($G / D_{\text{raw}}$) | 会话胜率 | 单步耗时 |
|---|---:|---:|---:|---:|---:|---:|
| **1. Raw (SlimeVR 无修正)** | 302.87° | 302.87° | +0.00° | 0.0% | 0.0% | - |
| **2. Old CSTN (Phase 6 短窗残差)** | 302.87° | 302.87° | +0.00° | +0.0015% | 60.0% | 0.006 ms |
| **3. Linear Rate (首锚点线性外推)** | 302.87° | 389.15° | -86.28° | **-28.5% (退化)** | 20.0% | - |
| **4. Direct Anchor (无滤波直接跳跃)** | 302.87° | 322.66° | -19.79° | **-6.5% (退化)** | 40.0% | - |
| **5. Anchor + Filter (M1 完整方案)** | 302.87° | 365.55° | -62.68° | **-20.7% (退化)** | 20.0% | **0.220 ms** |

### 3. 当时的归因说法（已由顶部独立复核撤销）

> [!CAUTION]
> **M1 实证核心发现：纯网络自推断锚点的可观测性瓶颈（Observability Bottleneck of Self-Inferred Anchors）**：
> 1. **旧 CSTN 对 30 分钟漂移完全无效（收益仅万分之 1.5）**：
>    在 30 分钟长序列上，旧 CSTN 输出残差与漂移完全不匹配，$D_{\text{corr}} = 302.87^\circ$，净收益仅 $+0.004^\circ$，再次印证短窗无长时偏置记忆。
> 2. **自推断锚点误差大于自然漂移率（伪漂移发散）**：
>    在无外部绝对真值下，神经网络尝试仅从下肢相对运动学与 HMD 上身关系估计绝对世界偏航误差。由于下肢各关节解剖开角自身存在 $5^\circ \sim 15^\circ$ 的姿态多样性，网络输出的 $z_i$ 包含显著的伪漂移测量噪声。
> 3. **卡尔曼滤波器被不准确的测量引导发散**：
>    对比 M0（高精外部锚点下收益 $+300.46^\circ$ / 99.4% 改善）与 M1（网络自推断锚点下收益 $-62.68^\circ$ / -20.7% 退化），证明：**长时漂移估计的瓶颈不是状态滤波算法，而是可部署输入下能否获得可信的锚点观测**。当观测器无法提供足够信噪比的 $z_i$ 时，强行滤波反而将姿态噪声误积分为偏航偏置。

### 4. M1 决议与安全边界
- 依据 M1 决策边界：“若终点没有稳定正收益，继续离线迭代，严禁接入在线修正”；
- 当前网络模型 `anchor_encoder_v3_best.pt` 仅供 Phase 5 影子模式工程链路验证吞吐与时延，**严禁在线覆写修改下肢姿态**；
- 离线算法下一步将重点探索增强可观测性输入（如引入脚部静止接触约束、更严格的站立几何对称性门禁或实机独立参考）。

---

## 任务 M1r/M1r-F 执行结果（可部署频域与静止零偏估计器 30 分钟评测）

> **机器报告与实验产物归档**：
> - 纯可部署时序契约与适配器测试：[`ai_corrector/corrector/test_contract_v3_mapping.py`](../../ai_corrector/corrector/test_contract_v3_mapping.py) (PASS)
> - 频域与静止段零偏估计器：[`ai_corrector/corrector/bias_estimator_v3.py`](../../ai_corrector/corrector/bias_estimator_v3.py)
> - 纯在线 30 分钟连续会话仿真器：[`ai_corrector/corrector/session_simulator_v3.py`](../../ai_corrector/corrector/session_simulator_v3.py)
> - M1r 自动化全链路基准评测：[`ai_corrector/corrector/execute_task_m1r.py`](../../ai_corrector/corrector/execute_task_m1r.py)
> - 机器评测报告：[`ai_corrector/artifacts/m1r_deployable_bias_model_report.json`](../../ai_corrector/artifacts/m1r_deployable_bias_model_report.json)

### 1. 严格可部署契约与输入隔离审计
- **契约重构与断言校验**：修正 `V2ToV3Adapter`，严格对齐 `contract_v2.py` 的 124 维通道拓扑（前 4 路 68 维 HMD/VD 视觉上下文 + 后 4 路 56 维四路下肢 IMU 观测），通过 `test_contract_v3_mapping.py` 自动化测试。
- **纯在线 0 泄漏输入**：会话仿真器彻底剥离离线真值，仅向估计器传递标准 `FrameV3` 流；估计器内部无任何读取真值漂移角、干净姿态或预知会话时长的可能。

### 2. 30 分钟跨工况连续会话实测对照矩阵 (10 会话 / 300 分钟)
覆盖 0°/min (零漂)、1°/min (低漂)、3°/min (中漂) 与 10°/min (高漂) 工况：

| 基线策略标识 | 原始误差增长 ($D_{\text{raw}}$) | 纠正后误差增长 ($D_{\text{corr}}$) | 净收益 ($G = D_{\text{raw}} - D_{\text{corr}}$) | 95% Bootstrap CI | 改善率 ($G / D_{\text{raw}}$) | 状态判定 |
|---|---:|---:|---:|---:|---:|:---:|
| **1. Raw (SlimeVR 无修正)** | 199.46° | 199.46° | +0.00° | [0.0°, 0.0°] | 0.0% | 基线 |
| **2. Oracle (理想真值上限)** | 199.46° | 0.00° | +199.46° | [+103.57°, +295.34°] | 100.0% | 上限参考 |
| **3. CSTN (Phase 6 真实权重)** | 199.46° | 199.46° | +0.00° | [0.0°, 0.0°] | 0.0% | 短窗无长时记忆 |
| **4. M1r-ZARU (静止段零偏估计)** | 199.46° | **31.91°** | **+167.54°** | **[+69.98°, +263.62°]** | **+84.0%** | **PASS** |
| **5. M1r-Full (ZARU + 步态摆动面)** | 199.46° | **31.91°** | **+167.54°** | **[+69.98°, +263.62°]** | **+84.0%** | **PASS** |

### 3. 核心实证结论与物理现实
1. **ZARU 静止段频域/方差门禁有效突破物理可观测性瓶颈**：
   在自由运动时，人体骨骼解剖自由度无法与陀螺漂移区分；但人体运动必然存在离散的“肢体局部静止窗口”（站立、静坐、停顿）。通过在频域与滑动方差域检测局部静止（角速度方差 $< 1.5\times 10^{-4}\text{ rad}^2/\text{s}^2$），成功直接观测各陀螺零偏，将 30 分钟累计偏航误差从 199.5° 压制至 31.9°（改善率 84.0%），且在 0°/min 零漂移会话中死区保护生效（误差增长为 0°），无零漂损坏。
2. **净收益置信区间严格大于零**：
   95% Bootstrap CI 下界为 $+69.98^\circ > 0$，满足 M1r 预设成功判定。
3. **安全边界提醒**：
   当前算法在仿真合成角速度流上验证通过，但真实硬件环境需要获取设备真实陀螺裸数据（或服务端低噪声原始角速度），下一步应在 Phase 5/8 实机硬件数据链路上进行端到端验证。


