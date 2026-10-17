# SlimeVR CSTN 项目总计划

> **更新时间**：2026-10-17
> **目标（2026-10-17 修订）**：一次 reset 后连续运行至少 30 分钟，使四颗下肢 IMU 从开始到结束的累计偏航误差增长总量低于 Raw；不以短窗或全程逐帧平均误差作为主验收。HMD / VD 上身参考只作辅助观测。
> **网络约定**：下载遇到网络阻断时使用 `http://127.0.0.1:7892`。

本文件是阶段索引。每阶段的子计划与执行结果保存在各自目录；见 [public 说明](README.md)。

## 当前状态与依赖

- **Phase 1–4**：v1 原型、训练、ONNX 工程基线及 v2 数据契约有完成记录。
- **Phase 5**：服务端已有桥接骨架，正进入正确特征映射（124 维）、实际 ONNX 探针推理和影子吞吐链路开发。**（并行工程重心）**
- **Phase 6**：D3c 在完整逐帧数据和新源序列上验证了连续模型 B；当前最长 5 秒片段未见可用收益。物理慢偏置 C 的世界偏航补偿误置于局部右乘，且脚接触未证明腿段静止，故不能据其退化裁定正确物理方案无效。精度目标未验收结项，后续 [D3d 坐标与观测核验](phase-06-accuracy-baseline/plan.md)；现有权重仅作影子探针，不启用在线姿态修改。
- **Phase 7**：合成回放已有记录；依赖 Phase 5 真实服务端工程打通后进行全链路 ≤20 ms 延迟验收。**（紧跟 Phase 5）**
- **Phase 8**：实机采集、独立真值和在线有效性验收待完成。
- **Phase 9**：[四路显式偏航状态模型](phase-09-stateful-bias-model/plan.md)：几何单测和状态滤波原型已完成。M1r 合成 10×30 分钟平均收益约 84%，但两条零漂会话明显恶化、`zero_drift_safe=false`，步态辅助触发 0 次；机器 `PASS` 漏掉安全否决，**独立复核撤销阶段通过判定**。下一步修正物理仿真与验收条件，核实裸陀螺链路，再进入实机 ≥30 分钟端点验收。

## 阶段目录

| 阶段 | 子计划 | 执行结果 | 状态 |
|---|---|---|---|
| Phase 1–3：v1 原型、训练与 ONNX 工程基线 | [plan.md](phase-01-03-v1-prototype/plan.md) | [results.md](phase-01-03-v1-prototype/results.md) | 已归档 |
| Phase 4：v2 数据契约与调研 | [plan.md](phase-04-v2-contract-research/plan.md) | [results.md](phase-04-v2-contract-research/results.md) | 已完成契约 |
| Phase 5：服务端影子桥接 | [plan.md](phase-05-server-shadow-mode/plan.md) | [results.md](phase-05-server-shadow-mode/results.md) | 并行工程重心 |
| Phase 6：序列隔离评测与补偿精度改进 | [plan.md](phase-06-accuracy-baseline/plan.md) | [results.md](phase-06-accuracy-baseline/results.md) | **D3c 复核：B 无收益，C 待修正** |
| Phase 7：合成回放与离线验收 | [plan.md](phase-07-replay-validation/plan.md) | [results.md](phase-07-replay-validation/results.md) | 紧跟 Phase 5 |
| Phase 8：实机采集与在线迁移 | [plan.md](phase-08-device-validation/plan.md) | [results.md](phase-08-device-validation/results.md) | 待实机 |
| Phase 9：四路下肢偏航状态模型重构 | [plan.md](phase-09-stateful-bias-model/plan.md) | [results.md](phase-09-stateful-bias-model/results.md) | **合成原型有平均收益；安全验收失败，实机待验收** |
| Phase 10：README 与阶段成果同步 | [plan.md](phase-10-readme-sync/plan.md) | [results.md](phase-10-readme-sync/results.md) | **已完成** |

## 当前精度结论

- **Phase 9 M0/M1 复核**：世界左乘与共轭右乘的随机姿态数学单测已完成。`99.4%` 是向滤波器输入 `真实漂移 + 0.5° 噪声` 的理想仿真上限；M1 的 `-20.7%` 来自含真值泄漏的输入，且旧 CSTN 对照是假设常数修正，不能解释为可部署模型的精度结论。现有报告 `PASS` 固定写入，已在[复核结果](phase-09-stateful-bias-model/results.md)中撤销判定。真实 30 分钟终点收益未知。
- **D3c 连续模型 B**：在 796 条完整逐帧序列（约 16.3 万帧）上训练；反事实方向一致率为 **48.97%**（开发集）与 **51.16%**（新源序列），中位数预测/真值差值比约 **0.006%**。该模型在当前短片段未展示可用补偿，不能外推为分钟级或所有长期估计器均无效。
- **C 评测无效的关键原因**：世界坐标注入的偏航漂移应由世界逆旋转左乘修正；现代码把世界慢偏航右乘到局部残差，并用脚接触/小腿阈值更新大腿与小腿。10°/min 退化约 0.19°、零漂移退化约 0.08° 是当前实现的结果，不能归因于正确坐标的偏置滤波器。
- **工程安全边界**：所有权重仍仅作 Phase 5 影子探针，严禁在线覆写下肢姿态；详见 [Phase 6 复核](phase-06-accuracy-baseline/results.md)。
