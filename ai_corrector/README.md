# SlimeVR 神经运动学修正器 (Neural Kinematics Corrector Subproject)

> **基于因果时空网络（CSTN）的 SlimeVR 6DoF 偏航累积漂移抑制与足部零速锁足（ZUPT）系统**  
> **适用硬件场景**：Meta Quest 3（上半身 6DoF 视觉零漂移锚点） + SlimeVR 下半身 6 轴 IMU（无磁力计方案）  
> **工程定位**：SlimeVR 服务端伴生高内聚 AI 子系统，独立承载算法、动捕数据库、模型分层生命周期、导出产物与数学技术文档。

---

## 一、 模块架构与组织规范 (`ai_corrector/`)

为了保持 SlimeVR 项目根目录的整洁高内聚，所有神经网络模型相关的算法源码、动捕数据库、权重资产、导出模型和规范文档均收敛于本目录：

```
ai_corrector/
├── corrector/                          # 【核心 AI 算法与运行时模块】
│   ├── model.py                        # CSTN 因果时空网络模型 (8 通道因果 LSTM + 空间 Transformer)
│   ├── loss.py                         # 符号不变性四元数测地损失与解剖学限幅目标函数
│   ├── train.py                        # GPU 全量训练执行器 (标准实验日志、AMP 混合精度、EarlyStopping)
│   ├── export.py                       # 单帧递归流式 ONNX 模型无损导出器 (含 8 帧对齐检验)
│   ├── checkpoint.py                   # 本地模型权重全生命周期管理系统 (best/latest/top_k 维护)
│   ├── prepare_dataset.py              # 全量 16,834 动捕序列多进程切片与偏航加噪引擎
│   ├── prepare_amass_subset.py         # 快速冒烟小样本生成工具
│   ├── rotations.py                    # 四元数与连续 6D 旋转数学法则算子
│   └── acceptance.py                   # 工程合规性与数学法则全套验收单元测试
├── database/                           # 【动捕基础数据库】
│   ├── download.py                     # 16 线程高并发动捕拉取器 (支持 7892 本地代理)
│   ├── smplh/                          # SMPL-H 基础人体骨骼网格参数 (男、女、中性 3 模型)
│   ├── amass/                          # AMASS 52 关节动捕原始库 (16,834 条 .npz 序列，1.81 GB)
│   └── processed/                      # 仿真加噪后的全量标准时空切片张量 (train.npz, val.npz)
├── checkpoints/                        # 【本地权重存储架构 (非 Git 管理)】
│   ├── best/                           # 生产最优权重 (cstn_best_full.pt, cstn_best_infer.pt, cstn_best.onnx)
│   ├── latest/                         # 最新一轮训练断点快照 (cstn_latest.pt，供无缝续训)
│   └── top_k/                          # 历史表现最好的 Top-3 候选池 (自动轮转淘汰)
├── artifacts/                          # 历史导出模型检查点与延迟报告存档 (.onnx, .pt, .json)
├── latex/                              # 【模型数学证明与训练全套技术规范】
│   ├── model_training_specification.tex# CSTN 训练技术规范 LaTeX 单一源
│   └── slimevr.tex                     # 系统总体设计 LaTeX 技术文档
├── .model-deps/                        # 本地隔离的 ONNXRuntime / ONNX 依赖库
├── __init__.py                         # Python 包入口
└── README.md                           # 本说明文档
```

---

## 二、 核心性能指标与交付成绩单 (KPI Benchmarks)

基于本地 NVIDIA GeForce RTX 4070 Laptop GPU，在 100% AMASS 全量数据集（16,785 条有效序列，165,229 个时空窗口，330.5 万帧动作流）上完成全量多任务训练收敛：

| 评估维度 | 核心物理指标 | 标称验收门槛 | **实测交付值** | 体验提升分析 |
| :--- | :--- | :---: | :---: | :--- |
| **偏航漂移消除** | 关节测地角误差 (`angle_err`) | $\le 0.50^\circ$ | **$0.1533^\circ$** (约 9 弧分) | 亚度级精度，消除大腿与小腿“扭麻花”变形。 |
| **足底抗滑步** | 触地判别 F1 (`contact_f1`) | $\ge 93.0\%$ | **$95.80\%$** | 越过 95% 标称线，为后续零速锁足（ZUPT）提供高置信触发。 |
| **单帧推理耗时** | 单线程 CPU ONNX 延迟 | $\le 4.50$ ms | **$0.40$ ms** (P95: 0.54 ms) | 仅占 60Hz 帧周期的 2.4%，完全不挤占游戏与 SteamVR 资源。 |
| **网络参数规模** | 模型可学习参数量 | $\le 2.00$ M | **$0.97$ M** (Hidden 96) | 结构紧凑精简，显存与内存开销几乎为零。 |
| **部署模型体积** | 流式 ONNX 文件体积 | $\le 8.00$ MB | **$3.99$ MB** | 轻量化独立部署，可直接被 C++/Rust/Java 服务端调用。 |
| **数值一致性** | ONNX vs PyTorch 8 帧对齐误差 | $\le 10^{-4}$ | $\text{atol} \le 2\times 10^{-5}$ | 严格保证长时递归滚动下计算图数学等价性。 |

---

## 三、 快速上手与操作指令

本模块支持在 `ai_corrector` 目录内运行，也支持在 SlimeVR 项目根目录下通过 `-m ai_corrector.corrector.xxx` 执行。

### 1. 动捕数据集拉取与切片

```powershell
# 拉取 AMASS 38 个动捕子集 (遇网络受限走 7892 本地代理)
python -u ai_corrector/database/download.py --threads 16

# 全量切片流水线 (10 进程多核并行，约 60 秒完成 16,785 条序列切片)
python -m ai_corrector.corrector.prepare_dataset --workers 10
```

### 2. 自动化工程验收测试 (Unit / Math Acceptance)

```powershell
python -m ai_corrector.corrector.acceptance
```
> 自动测试未来帧隔离、流式单帧与全序列等价性、四元数单位范数与 $\pm 8^\circ$ 物理限幅、符号不变性、无损张量加载等 12 项严苛法则。

### 3. 模型训练 (全量多任务时序训练)

```powershell
python -u -m ai_corrector.corrector.train `
    ai_corrector/database/processed/train.npz `
    ai_corrector/artifacts/cstn_compact_expanded.pt `
    --val-data ai_corrector/database/processed/val.npz `
    --epochs 20 `
    --batch-size 256 `
    --hidden 96 `
    --patience 3 `
    --print-iters 100
```

### 4. 本地权重分层生命周期管理 (CLI)

```powershell
# 1. 查看所有分层权重 (best, latest, top_k) 及其物理指标
python -m ai_corrector.corrector.checkpoint list

# 2. 深入解析权重内部张量、优化器状态与训练轮次
python -m ai_corrector.corrector.checkpoint info ai_corrector/checkpoints/best/cstn_best_full.pt

# 3. 剥离优化器动量生成纯推理轻量权重
python -m ai_corrector.corrector.checkpoint strip ai_corrector/checkpoints/best/cstn_best_full.pt --output ai_corrector/checkpoints/best/cstn_best_infer.pt

# 4. 意外中断秒级续训
python -u -m ai_corrector.corrector.train --resume ai_corrector/checkpoints/latest/cstn_latest.pt --epochs 20
```

### 5. 流式 ONNX 模型导出与一致性对齐检验

```powershell
python -m ai_corrector.corrector.export `
    ai_corrector/checkpoints/best/cstn_best_infer.pt `
    ai_corrector/checkpoints/best/cstn_best.onnx
```

### 6. 技术规范文档编译

```powershell
cd ai_corrector/latex
xelatex model_training_specification.tex
```
