# SlimeVR 因果残差模型

实现文档中的八路独立嵌入、八路单向 LSTM、三层空间 Transformer 与残差/门控/触地三头。默认 hidden=256、8 attention heads、FFN=1024；可选 hidden=96 用于轻量基线。每步仅对当前八个 token 做 attention，通过均值池化得到三头输入。共享主干上的多任务梯度会相互影响。

这是可训练的模型实现，尚未训练出可用于实际追踪的权重。没有声明通过 MPJPE、漂移、触地 F1 或全链路延迟验收。完整配置不能满足文档原先的 1.2M/2M 参数与 8MB 预算，实际值见验收报告。

## 输入契约 v1

输入 `[B,T,135]` float32，禁止 NaN/Inf。T 可变，训练推荐 20；在线每次 T=1，持续传递 h/c。人物切换、reset、参考空间重定位和长时间断流时清空状态。

八流按以下顺序拼接，所有速度单位为 m/s、rad/s，加速度 m/s²，位置 m，时间 s。使用统一右手世界坐标，重力为 -Z；传感器到肢段安装变换须先校准。位置减去 HMD 位置，加速度保留重力或移除重力必须在生成数据和部署间一致；v1 规定使用移除重力后的世界线加速度。valid 无效时其他字段置零，age 限在 [0,1]，归一化统计只能从训练划分拟合。

| 顺序 | 内容 | 维度 |
|---|---|---:|
| 0 | HMD anchor | 17 |
| 1 | 胸 IMU、髋 IMU | 28 |
| 2、3 | 左手、右手 anchor | 各17 |
| 4、5 | 左大腿、右大腿 IMU | 各14 |
| 6、7 | 左小腿、右小腿 IMU | 各14 |

anchor 字段依次是相对位置3、旋转6、世界线速度3、世界角速度3、valid1、age1。IMU 字段依次是旋转6、世界线加速度3、世界角速度3、valid1、age1。旋转6为旋转矩阵前两列按列拼接。HMD anchor 相对位置为零，仍保留该字段以统一格式。

这是本项目新定义的 v1 契约，不兼容 HMD-Poser 的原生135维。当前 VD 胸/肘辅助观测尚未接入此契约，不可直接替换 IMU 流；后续增加输入版本及适配器。当前固件/OSC 也未必提供原始角速度与硬件时间戳，真实采集适配仍待实现。

输出 dq `[B,T,6,4]`，顺序为胸、髋、左/右大腿、左/右小腿；gate `[B,T,6]` 表示**修正可靠性**；contact `[B,T,2]` 表示左/右足触地。四元数 wxyz，采用 `clean=raw*dq` 右乘，目标为 `inverse(raw)*clean`。三维旋转向量有界映射到单位四元数，整体角度不超过8°。这限制绝对残差，不能单独恢复任意大的累计漂移。

## 运行

在独立 Python 环境安装 requirements.txt（不必修改现有服务端环境）：

```powershell
python -m pip install -r corrector/requirements.txt
python -m corrector.acceptance
python -m corrector.train prepared_train.npz artifacts/corrector.pt --epochs 80
python -m corrector.export artifacts/corrector.pt artifacts/corrector.onnx
```

NPZ 必含 `x [N,T,135]`、单位残差四元数 `dq [N,T,6,4]`、`gate [N,T,6]`、`contact [N,T,2]`。标签概率在 [0,1]。按受试者/原始序列划分训练验证测试后再切窗口；不同窗口训练时隐状态重置。数据加载器只接受已预处理张量，不能直接载入 AMASS 原始姿态文件。数据模拟器、SMPL/FK 映射、真人数据采集与独立测试集还需实现。

训练损失采用四元数符号不变旋转项 + 0.5触地BCE + 0.1门控BCE。门控需要独立监督，不能直接用损坏传感器可靠性代替修正可靠性。未加入未经肢段标定的关节极限损失。

ONNX 导出显式暴露八路 h/c 状态，固定 batch=1、T=1、opset14、FP32，执行模型检查及八帧递归数值一致性校验。FP16/INT8 需另作精度验证。

`rotations.blend` 实现右乘残差与基于门控的 SLERP，低置信退回原姿态。不包含完整运动学、脚锁状态机、零偏滤波或 OSC/VMC 协议转换；模型不能保证绝对航向可观测。未训练模型禁止驱动实际 avatar。

## 验收边界

`acceptance.py` 检查未来帧隔离、整段与逐帧一致、角度上限、四元数代数、门控退避、三头梯度、微型样本优化和权重往返。CPU 时延只测单线程 PyTorch 每帧模型，不包含采样、无线、同步、脚锁、渲染或 VRChat。合成微型样本优化只证明训练管线工作，不能视为真实精度结果。

精度验收还必须具备独立 AMASS/DIP 测试集、与真实输入一致的仿真、原生 SlimeVR 基线、独立下肢真值及真实设备长时测试。当前报告会明确保留这些待验项目。
