# AMASS 小样本训练记录

日期：2026-10-17

## 数据和切分

使用本地已下载的 `database/amass/HumanML3D_SMPLH52`，选取排序后的前 12 条 HumanML3D/AMASS 序列。每条序列按 20 帧窗口、步长 5 帧提取，最多保留 8 个窗口。先按原始序列划分，再生成窗口，避免同一动作的相邻窗口同时进入训练和验证：9 条序列训练、3 条序列验证，共 72/24 个窗口。

预处理脚本是 `corrector/prepare_amass_subset.py`。它从 SMPL-H 前 22 个身体关节提取旋转和近似关节位置，按输入契约生成 `[N,20,135]`，并给六个下半身观测注入独立 yaw 随机游走。当前近似位置和接触标签用于训练管线验证，尚未替代完整 SMPL-H FK、传感器安装标定或真实足底真值。

## 训练

```powershell
D:\software\conda\envs\tqnet\python.exe -m corrector.prepare_amass_subset --sequences 12 --windows-per-sequence 8 --window 20 --stride 5
D:\software\conda\envs\tqnet\python.exe -u -m corrector.train database\processed\subset_train.npz artifacts\amass_subset96.pt --epochs 10 --hidden 96 --batch-size 12
```

轻量模型 hidden=96，参数量 974,426。AdamW 初始学习率 `1e-3`，Cosine 调度，梯度裁剪 1.0。训练 loss 从 0.368541（第 1 个 epoch）降至 0.038627（第 10 个 epoch）。

| 划分 | 总 loss | 平均残差角误差 | 接触 F1 |
|---|---:|---:|---:|
| 训练集 | 0.038313 | 0.1123° | 0.9876 |
| 验证集 | 0.048855 | 0.1138° | 0.9820 |

不修正基线的验证集平均目标残差角为 0.1168°，因此当前小样本训练把该合成残差降低约 2.6%。接触头的全正类基线 F1 为 0.9171，模型在合成标签上有明显提升。

## 结论和限制

训练管线、数据契约、序列级切分和验证计算均正常；模型确实在这批动捕窗口上学习到了合成残差和接触标签。残差本身很小，改善幅度也很小，不能据此证明能消除真实 IMU 漂移。

当前结果没有覆盖真实 LSM6DSV 噪声、绑带滑动、VD 观测误差、DIP-IMU、独立光学真值、长期漂移、GPU FP16 或 SlimeVR/VRChat 端到端链路。下一步应使用完整 FK 和物理传感器噪声模型扩大数据，并保留按受试者/原始序列的独立测试集。
