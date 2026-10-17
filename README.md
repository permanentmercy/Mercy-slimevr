# SlimeVR 神经网络时空残差校正器 (AI Kinematics Residual Corrector)

> **基于因果时空网络（CSTN）的 SlimeVR 6DoF 偏航累积漂移抑制与足部零速锁足（ZUPT）系统**  
> **适用硬件场景**：Meta Quest 3（上半身 6DoF 视觉零漂移锚点） + SlimeVR 下半身 6 轴 IMU（无磁力计方案）  
> **状态**：Phase 1 ~ Phase 3 全部完成交付（全量 16,785 条 AMASS 动捕序列训练收敛，流式 ONNX 单核 CPU 推理耗时 0.40 ms）

---

## 一、 项目背景与解决的行业痛点

SlimeVR 在 VR 社区被广泛使用，其核心优势是采用低成本 6 轴 IMU 传感器（如 BMI160 / BMI270 / LSM6DSV）。为了避免室内复杂电磁环境对磁力计的严重干扰，SlimeVR 普遍禁用地磁计。

然而，**禁用磁力计意味着缺乏绝对物理航向真北，下半身 IMU 的偏航角（Yaw）必然会随时间产生积分发散漂移**。
* **传统体验痛点**：玩家在 VRChat 中游玩 10~15 分钟，双腿和膝盖朝向便会明显扭曲变形（甚至出现扭麻花），脚底在站立和行走时严重滑步（Foot Skating），必须频繁长按手柄进行“快速重置校准（Quick Reset）”。
* **本方案解决方案**：
  利用 Quest 3 头显和双手手柄**绝对不会漂移的 6DoF 空间位姿作为几何锚点**，构建极轻量因果时空神经网络（CSTN）。模型以 60Hz/100Hz 逐帧捕获下半身 IMU 读数与上半身视觉锚点的动力学差异，在后台全自动推算微小四元数残差与足底触地状态，**实现无需人工干预的后台隐式自适应纠偏与物理锁足**。

---

## 二、 核心性能指标与交付成绩单 (KPI Benchmarks)

基于本地 NVIDIA GeForce RTX 4070 Laptop GPU 并在 100% AMASS 全量数据集（16,785 条序列，165,229 个时空窗口，330.5 万帧动作流）上完成训练收敛，实测交付指标如下：

| 评估维度 | 核心物理指标 | 标称验收门槛 | **本项目实测交付值** | 体验提升分析 |
| :--- | :--- | :---: | :---: | :--- |
| **偏航漂移消除** | 关节测地角误差 (`angle_err`) | $\le 0.50^\circ$ | **$0.1533^\circ$** (约 9 弧分) | 处于肉眼完全不可感知的亚度级精度，彻底告别下半身偏转扭曲。 |
| **足底抗滑步** | 触地判别 F1 (`contact_f1`) | $\ge 93.0\%$ | **$95.80\%$** | 越过 95% 标称线，为后续零速更新（ZUPT）状态机提供极高可靠性。 |
| **单帧推理耗时** | 单线程 CPU ONNX 延迟 | $\le 4.50$ ms | **$0.40$ ms** (P95: 0.54 ms) | 仅占 60Hz 帧周期的 2.4%，完全不挤占 SteamVR 和大型游戏 CPU 资源。 |
| **网络参数规模** | 模型可学习参数量 | $\le 2.00$ M | **$0.97$ M** (Hidden 96) | 架构高度精简，内存占用极低。 |
| **部署模型体积** | 流式 ONNX 文件体积 | $\le 8.00$ MB | **$3.99$ MB** | 轻量化独立部署，可直接被 C++/Rust/Java 服务端调用。 |
| **数值一致性** | ONNX vs PyTorch 8帧递归误差 | $\le 10^{-4}$ | $\text{atol} \le 2\times 10^{-5}$ | 严格保证长时递归滚动下计算图数学等价性。 |

---

## 三、 全局工程架构与目录规范

本项目采用高内聚、模块化架构设计。官方前后端与驱动、固件维持独立子系统隔离，**所有与神经网络模型相关的算法代码、数据集、检查点与技术文档统一定义并封装在 `ai_corrector/` 专属目录中**：

```
h:/program/slimevr/
├── ai_corrector/                       # 【神经运动学修正器全套组件 (AI Subproject)】
│   ├── corrector/                      # 核心 AI 算法 (CSTN 模型、测地损失、GPU 训练、流式 ONNX 导出、权重生命周期)
│   ├── database/                       # 动捕数据库 (SMPL-H 体型参数、AMASS 1.68 万序列、加噪处理张量、下载器)
│   ├── checkpoints/                    # 本地分层权重存储 (best/, latest/, top_k/，非 Git 隔离)
│   ├── artifacts/                      # 导出部署模型与基准评测报告 (cstn_best.onnx, report.json)
│   ├── latex/                          # 模型数学证明与技术规范 LaTeX 单一源
│   ├── .model-deps/                    # 本地 ONNX/ONNXRuntime 运行依赖
│   └── README.md                       # 神经模型专属全套使用文档
├── server/                             # SlimeVR Server 服务端 (Kotlin/Gradle) 与 GUI 前端控制台 (Electron/React)
├── driver/                             # SlimeVR SteamVR C++ 驱动源码
├── firmware/                           # SlimeVR 追踪节点单片机固件 (nRF52840 / ESP)
├── scripts/                            # 项目工程脚本与辅助工具
├── public/                             # 【团队协作中心】
│   └── plan.md                         # 核心路线图、数据契约接口定义与团队协作执行手册
├── BUILD-REPORT.md                     # 本地驱动与服务端环境构建执行报告
└── .gitignore                          # 完善的排除规则 (隔离大张量、二进制权重与外部子仓库)
```

---

## 四、 核心数学原理与输入输出数据契约

### 1. 输入特征张量 `X: [Batch, Time, 135]`
统一右手坐标系（重力矢量指向 $-Z$），每帧由 8 个解耦的人体观测流组合而成：
* **流 0 (头部 HMD 视觉锚)**：17 维（相对位置 3，6D 连续旋转 6，世界线速度 3，世界角速度 3，有效标志 1，时效 1）
* **流 1 (躯干骨骼：胸部 + 骨盆 IMU)**：28 维（胸部 IMU 14 维 + 骨盆 IMU 14 维）
* **流 2, 3 (双手手柄视觉定位)**：各 17 维（左/右手 6DoF 视觉观测位姿与速度）
* **流 4, 5 (双大腿 IMU)**：各 14 维（左/右大腿局部连续旋转 6，角加速度 3，角速度 3，有效位 1）
* **流 6, 7 (双小腿 IMU)**：各 14 维（左/右小腿局部连续旋转 6，角加速度 3，角速度 3，有效位 1）
* **足端加速度偏离量附加项**：$2 \times 2 = 4$ 维（双足动态模长残差）

### 2. 输出三解耦头与损失函数设计
网络前向传播同时驱动三个解耦的任务头：
* **姿态残差四元数 $\Delta q \in \mathbb{H}^6$**：预测 6 处下半身关节点单位四元数增量，经由 $\tanh$ 硬约束在 $\pm 8.0^\circ$ 物理限幅内。  
  **四元数双重覆盖性测地损失**：
  $$\mathcal{L}_{\text{rot}} = \frac{1}{6} \sum_{j=1}^6 \left(1 - \langle \Delta q_j^{\text{pred}}, \Delta q_j^{\text{true}} \rangle^2\right)$$
  对 $q$ 与 $-q$ 具备完全严格的符号不变性，彻底规避欧氏 MSE 引起的梯度剧烈震荡。
* **可靠性门控权重 $\text{gate} \in [0, 1]^6$**：Sigmoid 激活，输出当前预测置信度；当遇到剧烈非稳态动作时自动归零旁路，避免网络误干预。
* **足底触地判别 $\text{contact} \in [0, 1]^2$**：Sigmoid 激活，输出双脚当前帧处于支撑相（触地）的置信度。当 $\ge 0.5$ 时在下游触发零速修正（ZUPT），瞬间锁死脚踝世界坐标。

---

## 五、 本地模型权重与检查点管理系统 (Non-Git Weights Lifecycle)

为避免数以 GB 计的二进制模型写入 Git，本项目建立了专门的非 Git 本地检查点管理体系 [`ai_corrector/corrector/checkpoint.py`](ai_corrector/corrector/checkpoint.py)：

### 1. 结构与分工
* **断点续训态 (`cstn_best_full.pt`)**：包含 PyTorch 模型、AdamW 优化器动量、Cosine 学习率调度器与 AMP 梯度缩放器，用于断点完全恢复。
* **纯推理脱水态 (`cstn_best_infer.pt`)**：自动剥离优化器与训练态冗余张量，仅保留纯净结构和权重，体积紧凑、加载速度极快。
* **流式部署态 (`cstn_best.onnx`)**：单帧显式状态递归输入 $(x_t, h_{t-1}, c_{t-1}) \to (\Delta q_t, \text{gate}_t, \text{contact}_t, h_t, c_t)$，无未来帧等待。

### 2. 管理器常用命令行工具
```powershell
# 1. 格式化列出本地所有管理权重、版本、文件大小与物理指标
python -m ai_corrector.corrector.checkpoint list

# 2. 深度体检任意权重文件（查看内部张量列表、参数量、是否含优化器、训练指标溯源）
python -m ai_corrector.corrector.checkpoint info ai_corrector/checkpoints/best/cstn_best_full.pt

# 3. 将任意全量训练检查点手动脱水剥离，生成生产用轻量纯推理权重
python -m ai_corrector.corrector.checkpoint strip ai_corrector/checkpoints/best/cstn_best_full.pt --output ai_corrector/checkpoints/best/cstn_best_infer.pt
```

---

## 六、 快速上手与全链路复现指南

> **提示**：所有模型相关操作既可在根目录通过 `ai_corrector.` 运行，也可直接进入 `ai_corrector/` 目录运行。更多详细文档请参考 [`ai_corrector/README.md`](ai_corrector/README.md)。

### 1. 环境准备与依赖约定
推荐使用支持 CUDA 加速的 PyTorch 环境：
```powershell
# 网络受限时统一设置本地代理
$env:HTTP_PROXY="http://127.0.0.1:7892"
$env:HTTPS_PROXY="http://127.0.0.1:7892"

# 引入本地 ONNX 与 ONNXRuntime 运行依赖
$env:PYTHONPATH="h:\program\slimevr\ai_corrector\.model-deps;$env:PYTHONPATH"
```

### 2. 执行工程合规性全套验收测试
```powershell
python -m ai_corrector.corrector.acceptance
```
*自动验证：参数量预算、模型正向传播、四元数超球面单位归一化、符号不变性、流式状态递归、ONNX 对齐与延迟合规性。*

### 3. 数据拉取与全量切窗流水线
```powershell
# 1. 批量多线程拉取全量 38 个 AMASS 子集与 SMPL-H 几何模型 (走 7892 代理)
python -u ai_corrector/database/download.py --threads 16

# 2. 启动 10 进程并发并行切窗引擎 (自动处理 16,834 条序列并生成 train.npz / val.npz)
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

# 若训练意外中断，一键秒级断点无缝续训：
python -u -m ai_corrector.corrector.train `
  --resume ai_corrector/checkpoints/latest/cstn_latest.pt --epochs 20
```

### 5. 导出流式 ONNX 模型并进行单核 CPU 延迟实测
```powershell
python -m ai_corrector.corrector.export `
  ai_corrector/checkpoints/best/cstn_best_infer.pt `
  ai_corrector/checkpoints/best/cstn_best.onnx
```

---

## 七、 团队协作与当前工程阶段 (Roadmap)

团队持续更新的完整计划书参见 [`public/plan.md`](public/plan.md)。

* [x] **Phase 1: 数据准备与骨骼几何** (SMPL-H 模型对齐、1.68 万条 AMASS 序列下载、多进程切片加噪)
* [x] **Phase 2: 离线多任务因果训练** (符号不变性损失、RTX 4070 AMP 加速、EarlyStopping 达成 0.153° 偏角残差与 95.8% 触地 F1)
* [x] **Phase 3: 模型轻量化与流式 ONNX 交付** (单帧状态递归 ONNX 导出、0.40 ms 极低单核 CPU 延迟验收)
* [x] **Phase 3.5: 模型权重全生命周期管理系统** (`checkpoints/` 规范化分层流转、权重脱水瘦身、断点秒级恢复)
* [ ] **Phase 4: 跨域与真实物理传感器评测** (接入 DIP-IMU Xsens 真实噪声集测试、ZUPT 锁足状态机联合调试)
* [ ] **Phase 5: SlimeVR 服务端 OSC 桥接中继实机联调** (编写独立 OSC/VMC 桥接守护进程，打通 SlimeVR Server $\to$ VRChat 联机全身体感验证)

---

## 八、 开源协议与鸣谢
* 动捕数据集版权归属于 [AMASS (Max Planck Institute for Intelligent Systems)](https://amass.is.tue.mpg.de/)。
* 人体骨骼模型参数基于 SMPL-H 拓扑。
* 本项目核心校正架构与推理桥接专为开源全身体态追踪项目 [SlimeVR](https://github.com/SlimeVR/SlimeVR-Server) 定制优化。
