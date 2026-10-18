# SlimeVR 神经运动学与偏航状态估计原型

> **项目定位**：面向 SlimeVR 6DoF 下肢 IMU 的偏航漂移研究、离线评测与服务端影子接入原型。
> **当前状态**：Phase 9 的偏航改善来自合成长会话，Full 零漂安全和实机有效性未证实。Phase 11 的 4 项合成离线门禁通过；在真值残差提供的 oracle P9 Twist 条件下，世界系投影版模型对合成 Swing 残差改善 +1.4538° (66.7%)。扰动为角速度启发式、上身输入为理想 AMASS 姿态；实际 P9 在线估计状态下的组合收益、真实服务端字段、FK/脚滑及实机精度仍未验证，H4 待 Phase 8。所有模型继续保持离线/影子模式。
> **协作记录**：阶段计划与结果统一归档在 [`public/`](public/README.md)。

---

## 一、 项目背景与研究范围

SlimeVR 使用低成本 6 轴 IMU 传感器（如 BMI160、BMI270 和 LSM6DSV）。为减少室内电磁环境对磁力计的影响，常见配置不使用地磁计。

然而，**缺少磁力计会使绝对航向约束变弱，下肢 IMU 的偏航误差可能随时间累积**。具体漂移速度取决于传感器、融合器、安装和动作，不能仅凭短窗或合成数据外推到真实使用。

* **研究问题**：能否利用 HMD/VD 上身观测、四路腿部 IMU 的运动上下文，以及可检测的静止窗口，估计四颗传感器的慢变偏航偏置。
* **当前路线**：保留 v1 CSTN 作为短窗残差与工程基线；新主线采用四路显式慢偏移状态、静止段陀螺零偏估计（ZARU）和带条件的步态辅助观测。
* **安全边界**：HMD/VD 只作为带质量标记的辅助观测，不被假设为绝对无漂移锚点；模型输出在实机独立真值和服务端链路验收前不得直接覆写 Avatar 姿态。

---

## 二、 已有证据与当前验收状态

### v1 CSTN 工程基线（合成数据 / 工程验收）

基于本地 NVIDIA GeForce RTX 4070 Laptop GPU，v1 记录处理了 16,785 条有效 AMASS 序列、165,229 个窗口和约 330.5 万帧动作流。以下指标证明训练与部署工程链路可运行，**不证明真实 IMU 长期偏航、脚滑或 VRChat 体验已经改善**：

| 指标 | 结果 | 证据边界 |
| :--- | :---: | :--- |
| 合成残差角误差 (`angle_err`) | **$0.1533^\circ$** | 合成标签拟合；不是实机漂移误差 |
| 合成触地分类 F1 (`contact_f1`) | **95.80%** | 合成接触代理；不是已接入的 ZUPT 状态机 |
| 单线程 CPU ONNX 延迟 | 均值约 **0.397 ms**，P95 约 **0.610 ms** | 模型推理冒烟；不含采样、无线、服务端和求解链路 |
| 模型规模 / 紧凑 ONNX | **约 0.97M / 3.99 MB** | v1 Hidden 96 工程产物 |
| PyTorch/ONNX 8 帧递归对齐 | `atol ≤ 2×10⁻⁵` | 数值一致性检查 |

### Phase 9 M1r/M1r-F 状态估计原型（合成长会话）

最新 10×30 分钟合成报告显示 Raw 总终点误差均值为 199.46°；ZARU-only 为 9.28°（平均收益 190.17°，改善 95.3%），Full 为 6.68°（平均收益 192.77°，改善 96.6%）。这是模拟条件下的积极信号，不是实机结论。

零漂安全尚未证明：报告的安全状态只检查 ZARU；Full 在两条标称 0°/min 会话中把端点误差从 4.76°、3.48°分别改至 1.64°、1.25°。压力集的 drift_tier=0 还包含随机游走，不能充当严格零漂金标。仿真先构造世界漂移再反投影为角速度，也未验证真实固定机体系偏置经板载 AHRS 的行为。详见 [Phase 9 复核](public/phase-09-stateful-bias-model/results.md)。裸陀螺数据来源、独立真值和服务端端到端延迟均未验收。

---

## 三、 工程结构

项目按功能划分目录。官方前后端、驱动和固件保持独立；算法、评测和阶段文档分别放在 `ai_corrector/` 与 `public/`：

```
h:/program/slimevr/
├── ai_corrector/                       # 神经运动学修正器相关组件
│   ├── corrector/                      # CSTN、v2/v3 契约、状态滤波、ZARU、评测与 ONNX 工具
│   ├── database/                       # 动捕数据库 (SMPL-H 体型参数、AMASS 1.68 万序列、加噪处理张量、下载器)
│   ├── checkpoints/                    # 本地分层权重存储 (best/, latest/, top_k/，非 Git 隔离)
│   ├── artifacts/                      # 本地模型与机器评测报告（非 Git）
│   ├── latex/                          # 模型数学证明与技术规范 LaTeX 单一源
│   ├── .model-deps/                    # 本地 ONNX/ONNXRuntime 运行依赖
│   └── README.md                       # 神经模型使用文档
├── server/                             # SlimeVR Server 服务端 (Kotlin/Gradle) 与 GUI 前端控制台 (Electron/React)
├── driver/                             # SlimeVR SteamVR C++ 驱动源码
├── firmware/                           # SlimeVR 追踪节点单片机固件 (nRF52840 / ESP)
├── scripts/                            # 项目工程脚本与辅助工具
├── public/                             # 【团队协作中心】
│   ├── plan.md                         # 总路线图与当前状态
│   └── phase-*/                         # 各阶段独立 plan.md / results.md
├── BUILD-REPORT.md                     # 本地驱动与服务端环境构建执行报告
└── .gitignore                          # 完善的排除规则 (隔离大张量、二进制权重与外部子仓库)
```

---

## 四、 数据契约与模型概要

仓库同时保留两个契约版本，不能混用：

### 1. v1 CSTN 输入：`X: [Batch, Time, 135]`

这是 Phase 1–3 的短窗残差基线。统一右手坐标系（重力矢量指向 $-Z$），每帧由 8 个解耦的人体观测流组合而成：
* **流 0 (头部 HMD 视觉锚)**：17 维（相对位置 3，6D 连续旋转 6，世界线速度 3，世界角速度 3，有效标志 1，时效 1）
* **流 1 (躯干骨骼：胸部 + 骨盆 IMU)**：28 维（胸部 IMU 14 维 + 骨盆 IMU 14 维）
* **流 2, 3 (双手手柄视觉定位)**：各 17 维（左/右手 6DoF 视觉观测位姿与速度）
* **流 4, 5 (双大腿 IMU)**：各 14 维（左/右大腿局部连续旋转 6，角加速度 3，角速度 3，有效位 1）
* **流 6, 7 (双小腿 IMU)**：各 14 维（左/右小腿局部连续旋转 6，角加速度 3，角速度 3，有效位 1）
* **足端加速度偏离量附加项**：$2 \times 2 = 4$ 维（双足动态模长残差）

### 2. v2/v3 部署时序契约：124 维基础流

Phase 9 使用 `contract_v2.py` / `contract_v3.py` 对齐在线输入：前四路 HMD/VD 视觉流共 `4×17=68` 维，后四路下肢 IMU 流共 `4×14=56` 维，合计 124 维；v3 另带时间戳、reset/重标定、有效性、数据年龄和追踪质量等时序状态。`V2ToV3Adapter` 与原始字段映射是当前服务端接入的关键验收项。

### 3. v1 CSTN 输出与限制

v1 网络前向传播同时驱动三个解耦的任务头：
* **姿态残差四元数 $\Delta q \in \mathbb{H}^6$**：预测 6 处下半身关节点单位四元数增量，经由 $\tanh$ 硬约束在 $\pm 8.0^\circ$ 物理限幅内。
  **四元数双重覆盖性测地损失**：
  $$\mathcal{L}_{\text{rot}} = \frac{1}{6} \sum_{j=1}^6 \left(1 - \langle \Delta q_j^{\text{pred}}, \Delta q_j^{\text{true}} \rangle^2\right)$$
  对 $q$ 与 $-q$ 不敏感，可减少欧氏 MSE 在四元数符号切换时产生的训练不稳定。
* **可靠性门控权重 $\text{gate} \in [0, 1]^6$**：Sigmoid 激活，输出当前预测置信度；低置信度时降低修正影响。
* **足底触地判别 $\text{contact} \in [0, 1]^2$**：Sigmoid 激活，输出双脚当前帧处于支撑相（触地）的置信度。该输出目前只是模型头；完整 ZUPT/脚锁状态机尚未在服务端实现或验收。

Phase 9 的主线不再把逐帧 3D 残差作为长期漂移解法，而是维护四路独立慢偏移状态 `e_i, b_i, P_i`，只在可靠锚点或逐颗传感器静止窗口出现时更新。

---

## 五、 本地权重与检查点管理 (Non-Git Weights Lifecycle)

为避免数以 GB 计的二进制模型写入 Git，本项目建立了专门的非 Git 本地检查点管理体系 [`ai_corrector/corrector/checkpoint.py`](ai_corrector/corrector/checkpoint.py)：

### 1. 结构与分工
* **断点续训态 (`cstn_best_full.pt`)**：包含 PyTorch 模型、AdamW 优化器动量、Cosine 学习率调度器与 AMP 梯度缩放器，用于恢复训练状态。
* **纯推理脱水态 (`cstn_best_infer.pt`)**：剥离优化器与训练态冗余张量，仅保留模型结构和权重，便于推理加载。
* **流式基线态 (`cstn_best.onnx`)**：单帧显式状态递归输入 $(x_t, h_{t-1}, c_{t-1}) \to (\Delta q_t, \text{gate}_t, \text{contact}_t, h_t, c_t)$，无未来帧等待；它是工程基线，不是已通过实机验收的生产修正器。

### 2. 管理器常用命令行工具
```powershell
# 1. 格式化列出本地所有管理权重、版本、文件大小与物理指标
python -m ai_corrector.corrector.checkpoint list

# 2. 深度体检任意权重文件（查看内部张量列表、参数量、是否含优化器、训练指标溯源）
python -m ai_corrector.corrector.checkpoint info ai_corrector/checkpoints/best/cstn_best_full.pt

# 3. 将训练检查点剥离为推理用权重
python -m ai_corrector.corrector.checkpoint strip ai_corrector/checkpoints/best/cstn_best_full.pt --output ai_corrector/checkpoints/best/cstn_best_infer.pt
```

---

## 六、 v1 工程基线复现指南

> **提示**：以下命令复现的是 v1 CSTN 的数据、训练、验收和 ONNX 工程基线，不会完成真实硬件校正。所有模型相关操作既可在根目录通过 `ai_corrector.` 运行，也可直接进入 `ai_corrector/` 目录运行；具体限制见 [`ai_corrector/corrector/README.md`](ai_corrector/corrector/README.md)。

### 1. 环境准备与依赖约定
推荐使用支持 CUDA 加速的 PyTorch 环境：
```powershell
# 网络受限时统一设置本地代理
$env:HTTP_PROXY="http://127.0.0.1:7892"
$env:HTTPS_PROXY="http://127.0.0.1:7892"

# 引入本地 ONNX 与 ONNXRuntime 运行依赖
$env:PYTHONPATH="h:\program\slimevr\ai_corrector\.model-deps;$env:PYTHONPATH"
```

### 2. 执行工程验收测试
```powershell
python -m ai_corrector.corrector.acceptance
```
*检查参数量预算、模型正向传播、四元数归一化、符号不变性、流式状态递归、ONNX 对齐和推理延迟；这些是工程检查，不是实机精度验收。*

### 3. 数据拉取与全量切窗流水线
```powershell
# 1. 批量多线程拉取全量 38 个 AMASS 子集与 SMPL-H 几何模型 (走 7892 代理)
python -u ai_corrector/database/download.py --threads 16

# 2. 使用 10 个进程切窗 (处理本地数据并生成 train.npz / val.npz)
python -u -m ai_corrector.corrector.prepare_dataset --per-subset-limit 0 --windows-per-sequence 10 --workers 10
```

### 4. 启动 GPU 全量多任务模型训练
```powershell
# 全量训练（标准实验日志格式，支持 AMP 混合精度与 EarlyStopping）
python -u -m ai_corrector.corrector.train `
  ai_corrector/database/processed/train.npz `
  ai_corrector/checkpoints/best/cstn_best_full.pt `
  --val-data ai_corrector/database/processed/val.npz `
  --epochs 20 --batch-size 256 --hidden 96 --patience 3 --print-iters 100

# 从检查点继续训练：
python -u -m ai_corrector.corrector.train `
  --resume ai_corrector/checkpoints/latest/cstn_latest.pt --epochs 20
```

### 5. 导出流式 ONNX 模型并进行单核 CPU 延迟实测
```powershell
python -m ai_corrector.corrector.export `
  ai_corrector/checkpoints/best/cstn_best_infer.pt `
  ai_corrector/checkpoints/best/cstn_best.onnx
```

### 6. Phase 9 合成状态估计评测

M1r/M1r-F 的 30 分钟连续会话评测由 [`execute_task_m1r.py`](ai_corrector/corrector/execute_task_m1r.py) 生成机器报告。它使用合成 `FrameV3` 流验证输入隔离和 ZARU 原型，不能替代真实设备日志：

```powershell
python -u ai_corrector/corrector/execute_task_m1r.py
```

报告默认写入被 `.gitignore` 隔离的 `ai_corrector/artifacts/m1r_deployable_bias_model_report.json`。当前服务端桥接仍未创建 ONNX session，也未把修正结果提交给 Avatar 求解器。

---

## 七、 团队协作与当前工程阶段 (Roadmap)

团队持续更新的完整计划书参见 [`public/plan.md`](public/plan.md)。

* [x] **Phase 1–3：v1 原型、训练与 ONNX 工程基线**（AMASS/SMPL-H 流水线、CSTN、参数/数学/流式验收；指标来自合成数据和工程测试）
* [x] **Phase 4：v2 数据契约与方案调研**（124 维部署契约、时序状态与观测边界）
* [ ] **Phase 5：服务端影子桥接**（已有 Kotlin 钩子和配置骨架；ONNX 推理、契约映射、影子输出和状态清理待完成）
* [ ] **Phase 6：序列隔离精度评测**（D3c 连续模型 B 当前未显示可用收益；旧 C 分支坐标/观测条件有误，不能据此否定正确物理方案）
* [ ] **Phase 7：合成回放与离线端到端验收**（已有部分合成结果，等待服务端链路打通）
* [ ] **Phase 8：真实设备采集与在线迁移**（尚无同步动捕、独立真值和 ≥30 分钟实机会话结果）
* [ ] **Phase 9：四路偏航状态估计原型**（合成 10×30 分钟报告 Full 均值改善 96.6%；Full 零漂安全未证实，真实硬件和服务端验收待完成）
* [x] **Phase 10：README 与阶段成果同步**（本页与 `public/` 阶段索引已对齐）
* [x] **Phase 11：Phase 6/9 正交组合验证**（4 项合成离线门禁 PASS；P6 世界 Swing 残差改善 +1.4538°/66.7%；服务端字段、FK/脚滑与实机仍待验证，H4 待 Phase 8）

### 当前结论边界

- 当前结果不能证明真实 SlimeVR 偏航漂移、脚滑或 VRChat 全身体验已经改善。
- M0 的 99.4% 理想锚点上限、M1 的 -20.7% 历史对照和 v1 的 0.1533° 合成残差不应作为实机 KPI。
- 在 Phase 5/8 的契约、裸陀螺链路、独立真值和安全回放完成前，不在线覆写下肢姿态。

---

## 八、 开源协议与鸣谢
* 动捕数据集版权归属于 [AMASS (Max Planck Institute for Intelligent Systems)](https://amass.is.tue.mpg.de/)。
* 人体骨骼模型参数基于 SMPL-H 拓扑。
* 本项目核心校正架构与推理桥接专为开源全身体态追踪项目 [SlimeVR](https://github.com/SlimeVR/SlimeVR-Server) 定制优化。
