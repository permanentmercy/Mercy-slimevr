# SlimeVR 神经网络残差校正器 —— 团队协作与实施规划 (Plan)

> **文档定位**：团队协作计划书与工程路线图，持续更新各阶段进展、数据规范与交付基准。  
> **更新时间**：2026-10-17  
> **网络环境约定**：外部资产与开源库拉取遇网络受限时，统一使用本地代理端口 `http://127.0.0.1:7892`。

---

## 一、 项目背景与核心技术指标

针对 SlimeVR 6 轴 IMU 方案在下半身动捕中的**累积偏航漂移（Yaw Drift）**与**足部滑步（Foot Skating）**问题，本项目构建一套在服务端实时运行的紧凑型因果时空神经网络（CSTN）。该网络通过 Quest 3 上半身视觉锚点与下半身 IMU 数据时空融合，在每帧输出姿态四元数增量残差、置信度门控与足部触地概率。

### 核心交付指标 (KPI)

| 指标项 | 验收门槛 | 标称目标 | 验证方式 |
| :--- | :--- | :--- | :--- |
| **模型网络参数量** | $\le 2.0 \times 10^6$ (2.0 M) | $\sim 1.54$ M (Hidden 256) / $\sim 0.97$ M (Hidden 96) | `sum(p.numel())` 统计 |
| **单帧推理耗时** | $\le 2.0$ ms (GPU FP16) / $\le 4.5$ ms (CPU ONNX) | $< 1.8$ ms (GPU) / $< 2.4$ ms (CPU) | 1000 帧流式推理统计均值 |
| **全链路端到端延迟** | $\le 20.0$ ms | $< 18.0$ ms | IMU 采样到 VRChat 接收骨骼 |
| **关节位置误差 (MPJPE)** | $\le 45.0$ mm | $< 38.0$ mm | DIP-IMU / AMASS 测试集评测 |
| **偏航漂移消除率** | $\ge 88.0\%$ | $> 92.0\%$ | 注入 $10^\circ$/min 漂移测试残差压制 |
| **足底触地判别 F1** | $\ge 93.0\%$ | $> 95.0\%$ | 动捕接触真值二分类混淆矩阵 |
| **足部滑步抑制率** | $\ge 80.0\%$ | $> 85.0\%$ | 静态站立时脚踝世界坐标漂移削减 |

---

## 二、 阶段路线图与当前进展 (Roadmap)

```
[Phase 1: 数据准备与骨骼几何] (数据拉取完成，待切窗加噪 ⏳)
  ├─ [√] 接入 SMPL-H 几何模型 (male / female / neutral，共 250 MB)
  ├─ [√] 建立 HuggingFace 高并发多线程拉取器 (16 线程，走 7892 代理)
  ├─ [√] 批量同步 AMASS 38 个高质量子集至 ./database (16,834 动作序列，1.81 GB)
  └─ [√] 构建仿真加噪与因果切窗打包流水线 (corrector/prepare_amass_subset.py)

[Phase 2: 离线多任务训练]
  ├─ [√] 多任务 CSTN 架构与解耦输出头实现 (corrector/model.py)
  ├─ [√] 符号不变性四元数损失与解剖学约束 (corrector/loss.py)
  ├─ [√] 单元与工程合规验收脚本 (corrector/acceptance.py)
  └─ [~] 已完成 12 序列小样本训练；完整 AMASS 80 Epochs 正式训练待数据与 FK 管线完善

[Phase 3: 模型轻量化与 ONNX 部署]
  ├─ [√] 单帧无损流式 ONNX 递归导出器 (corrector/export.py)
  ├─ [ ] FP16 半精度转换与算子融合
  └─ [ ] 精度与推理延迟实测验收

[Phase 4: 跨域与物理接触评测]
  ├─ [ ] 接入真实传感器 DIP-IMU 测试集交叉检验
  └─ [ ] 物理运动学仲裁器与 ZUPT 足锁状态机集成测试

[Phase 5: 服务端中继联调]
  ├─ [ ] 编写独立 OSC/VMC 桥接守护进程 (UDP 39539 -> 9000)
  └─ [ ] SlimeVR Server + VRChat 联机全身追踪实机验证
```

---

## 三、 数据集组织规范与目录结构 (`./database`)

项目所有原始动捕数据、几何模型与预处理张量统一定义在 `./database` 目录下：

```
database/
├── download.py                         # 高并发多线程数据拉取器
├── smplh/                              # SMPL-H 基础骨骼几何体型参数 (已就绪)
│   ├── male/model.npz                  # 男性骨骼模型 (87.2 MB)
│   ├── female/model.npz                # 女性骨骼模型 (87.2 MB)
│   └── neutral/model.npz               # 中性骨骼模型 (87.2 MB)
├── amass/
│   └── HumanML3D_SMPLH52/              # AMASS 52 关节动捕序列 (同步中)
│       ├── 0000/ ~ 0022/               # CMU 日常动作子集 (行走、转身、起坐、跳跃等 23 组)
│       └── M000/ ~ M014/               # HDM05 技巧动作子集 (跑跳、体操、非稳态姿态等 15 组)
└── processed/                          # 仿真加噪后的标准训练/验证张量 (待生成)
    ├── subset_train.npz                # 小样本 [72, 20, 135]（本地生成）
    ├── subset_val.npz                  # 小样本 [24, 20, 135]（本地生成）
    └── test_dip.npz                    # 跨域真实传感器测试集
```

---

## 四、 核心数据契约与接口定义 (Input / Output Contract)

为确保模型训练、ONNX 部署与后续 SlimeVR 服务端各模块无缝衔接，严格遵循以下契约：

### 1. 输入特征张量 `X_in: [B, T, 135]`
采用 8 身体流通道规范，统一右手世界坐标系（重力指向 $-Z$）：
* **流 0 (头部参考锚)**：17 维（相对位置 3，6D 连续旋转 6，世界线速度 3，世界角速度 3，有效标志 1，时效 1）
* **流 1 (躯干：胸部 + 髋部)**：28 维（胸部 IMU 14 维 + 髋部 IMU 14 维）
* **流 2, 3 (双肘/双手)**：各 17 维（左/右手 6DoF 视觉观测）
* **流 4, 5 (双大腿)**：各 14 维（左/右大腿 IMU 读数与局部姿态）
* **流 6, 7 (双小腿/双足)**：各 14 维（左/右小腿 IMU 读数与局部姿态）
* **足端加速度模长附加项**：$2 \times 2 = 4$ 维（双足线加速度动态模长偏离度）

### 2. 输出解耦头
* **姿态残差 `dq: [B, T, 6, 4]`**：6 处下半身关节点单位四元数增量，物理限幅在 $\pm 8^\circ$ 内。
* **门控置信度 `gate: [B, T, 6]`**：Sigmoid 激活，各节点当前预测可信度权重；若 $< 0.3$ 则自动旁路跳过。
* **触地判定 `contact: [B, T, 2]`**：Sigmoid 激活，双足当前帧支撑相/触地概率。

---

## 五、 团队协作指令手册

1. **增量拉取/继续同步数据**：
   ```powershell
   python -u database/download.py --threads 16
   ```
2. **执行模型工程合规验收测试**：
   ```powershell
   python -m corrector.acceptance
   ```
3. **冒烟训练验证流水线**：
   ```powershell
   python -m corrector.train artifacts/smoke_train.npz artifacts/smoke_corrector.pt --epochs 2 --hidden 96 --batch-size 4
   ```

6. **AMASS 小样本训练与验证**：
   ```powershell
   python -m corrector.prepare_amass_subset --sequences 12 --windows-per-sequence 8 --window 20 --stride 5
   python -u -m corrector.train database/processed/subset_train.npz artifacts/amass_subset96.pt --epochs 10 --hidden 96 --batch-size 12
   ```
   当前结果见 `corrector/SUBSET_TRAINING_REPORT.md`；该结果只验证训练管线，不代表真实传感器精度。
4. **编译 LaTeX 全套技术规范文档**：
   ```powershell
   cd latex
   xelatex model_training_specification.tex
   ```
