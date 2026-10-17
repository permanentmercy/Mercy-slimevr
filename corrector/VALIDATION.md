# 工程验收记录 — 2026-10-17

结论：模型代码工程验收通过；真实追踪精度与整机验收未完成。不能据此声称文档的全部目标已达标。

执行环境：Windows，Intel Family 6 Model 151 CPU，Python 3.8.20，PyTorch 2.4.1+cu121，CPU 单线程。ONNX 1.17.0、ONNX Runtime 1.19.2。CUDA 可用，但本次没有测量 GPU FP16。依赖安装在项目 `.model-deps/`，没有修改 SlimeVR 环境。测试版本清单见 requirements-tested.txt。

## 通过项目

- `python -m corrector.acceptance`：两种配置的因果隔离、逐帧/整段一致、四元数归一化与8°限幅、右乘目标、门控退避、有限梯度、微型样本优化、检查点往返、NPZ 数据加载与错误标签拒绝。
- `python -m corrector.train artifacts/smoke_train.npz artifacts/smoke_corrector.pt --epochs 2 --hidden 96 --batch-size 4`：4条20帧随机输入、单位残差标签的训练命令冒烟；loss 从0.398491下降至0.269507。不是 AMASS 训练，不具有泛化或精度含义。
- `python -m corrector.export ...`：完整配置和轻量配置均通过 ONNX checker 与8帧隐状态递归数值对照，容差 atol=2e-5、rtol=2e-4。轻量使用上述冒烟检查点，完整配置使用未训练初始化权重。
- `python -m compileall -q corrector`、`git diff --check`。

| 配置 | 参数数 | 测试 ONNX 文件 | CPU ONNX 均值 / P95 |
|---|---:|---:|---:|
| 完整256维 | 6,689,306 | 20,521,774 bytes | 1.956 / 2.370 ms |
| 轻量96维 | 974,426 | 3,997,816 bytes | 0.397 / 0.610 ms |

各延迟测量1000帧，30帧预热，batch=1、T=1、FP32、CPU 单线程，只含 ONNX 模型和状态传递。完整模型初始化参数可能被导出器去重，真实训练后的文件大小需重新测量。数值记录分别见 acceptance-report.json、onnx-full-report.json、onnx-compact-report.json。

## 未验收项目

没有找到完整 AMASS/DIP 测试集和对应预处理张量，也没有独立光学真值或设备会话。MPJPE、偏航误差削减率、触地F1、误修正率、重置间隔、GPU FP16、完整脚锁、实时传感器数据适配与 SlimeVR/VD/VRChat 链路均未验收。

完整256维配置不符合原文参数/体积预算；轻量配置符合规模目标，但不能推断其精度达标。ONNX CPU 时延符合4.5ms模型预算，不能代替端到端20ms验收。

随机输入、冒烟检查点与 ONNX 二进制位于忽略的 artifacts/ 中，不提交为正式训练成果。只提交模型源码、验收脚本、报告和设计修正。
