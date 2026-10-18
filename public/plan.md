# SlimeVR CSTN 项目总计划

> **更新时间**：2026-10-18
> **目标（2026-10-17 修订）**：一次 reset 后连续运行至少 30 分钟，使四颗下肢 IMU 从开始到结束的累计偏航误差增长总量低于 Raw；不以短窗或全程逐帧平均误差作为主验收。HMD / VD 上身参考只作辅助观测。
> **网络约定**：下载遇到网络阻断时使用 `http://127.0.0.1:7892`。

本文件是阶段规划索引；全阶段核心量化指标与成果汇总请参阅 [总成果看板 (results.md)](results.md)。每阶段的子计划与执行结果保存在各自目录，见 [public 说明](README.md)。

## 当前状态与依赖

- **Phase 1–4**：v1 原型、训练、ONNX 工程基线及 v2 数据契约有完成记录。
- **Phase 5**：服务端有本机 ONNX shadow bridge 原型和 Kotlin 测试；Session 关闭路径、完整服务循环验收及独立仓库提交仍未完成，继续保持影子模式。**（并行工程重心）**
- **Phase 6**：D3c 在完整逐帧数据和新源序列上验证了连续模型 B；当前最长 5 秒片段未见可用收益。物理慢偏置 C 的世界偏航补偿误置于局部右乘，且脚接触未证明腿段静止，故不能据其退化裁定正确物理方案无效。精度目标未验收结项，后续 [D3d 坐标与观测核验](phase-06-accuracy-baseline/plan.md)；现有权重仅作影子探针，不启用在线姿态修改。
- **Phase 7**：合成回放已有记录；依赖 Phase 5 真实服务端工程打通后进行全链路 ≤20 ms 延迟验收。**（紧跟 Phase 5）**
- **Phase 8**：实机采集、独立真值和在线有效性验收待完成。
- **Phase 9**：[四路显式偏航状态模型](phase-09-stateful-bias-model/plan.md)：M1r/M1r-F 的合成漂移收益仍是研究信号。Phase 12 严格零漂用例测得 endpoint 改变量 0.008977°，但静态姿势门禁没有执行，且仿真用直接构造的陀螺偏置角速度、服务端当前使用姿态差分；Full 零漂安全与硬件有效性仍未验收。
- **Phase 11**：[阶段六/九正交组合](phase-11-hybrid-compensation/plan.md)：4 项合成离线机器门禁 PASS；Oracle P9 Twist 条件下，P6 将世界系 Swing 残差在 190 条序列上从 2.1799° 降至 0.7261°，改善 +1.4538° (66.7%)。30 分钟合成端点收益与 P9 ZARU 单独几乎相同。Phase 12 复用已使用过的 FinalConfirm，且 P9-only 与 Raw 完全相同；其 +1.352° Swing 结果只说明 P6 在该合成扰动上的探索收益，不能证明 P9+P6 组合、FK 或世界足滑改善，H4 仍待新隔离集与 Phase 8。
- **Phase 12**：[无实机影子推理、回放与算法离线复验](phase-12-prehardware-validation/plan.md)：**整体验收未通过**。契约 6 项、推理 5 项、回放 2 项 Kotlin 测试通过；最新回放逐帧推理 108,000 次，特征组装加推理 P95 0.657 ms，但未经过完整 `VRServer.tick()`，GC 后内存 delta -0.517 MB 也没有门限。S1 缺服务关闭时释放 Session 的调用；A1 静态安全门禁硬编码通过且仿真输入不匹配服务端角速度来源；A2 复用旧 FinalConfirm、无新隔离 manifest，P9-only 与 Raw 相同，FK/足滑为简化代理。待办见 [Phase 12 R1](phase-12-prehardware-validation/plan.md)。

## 阶段目录

| 阶段 | 子计划 | 执行结果 | 状态 |
|---|---|---|---|
| Phase 1–3：v1 原型、训练与 ONNX 工程基线 | [plan.md](phase-01-03-v1-prototype/plan.md) | [results.md](phase-01-03-v1-prototype/results.md) | 已归档 |
| Phase 4：v2 数据契约与调研 | [plan.md](phase-04-v2-contract-research/plan.md) | [results.md](phase-04-v2-contract-research/results.md) | 已完成契约 |
| Phase 5：服务端影子桥接 | [plan.md](phase-05-server-shadow-mode/plan.md) | [results.md](phase-05-server-shadow-mode/results.md) | 并行工程重心 |
| Phase 6：序列隔离评测与补偿精度改进 | [plan.md](phase-06-accuracy-baseline/plan.md) | [results.md](phase-06-accuracy-baseline/results.md) | **D3c 复核：B 无收益，C 待修正** |
| Phase 7：合成回放与离线验收 | [plan.md](phase-07-replay-validation/plan.md) | [results.md](phase-07-replay-validation/results.md) | 紧跟 Phase 5 |
| Phase 8：实机采集与在线迁移 | [plan.md](phase-08-device-validation/plan.md) | [results.md](phase-08-device-validation/results.md) | 待实机 |
| Phase 9：四路下肢偏航状态模型重构 | [plan.md](phase-09-stateful-bias-model/plan.md) | [results.md](phase-09-stateful-bias-model/results.md) | **合成收益积极；Phase 12 静态安全门禁无效** |
| Phase 10：README 与阶段成果同步 | [plan.md](phase-10-readme-sync/plan.md) | [results.md](phase-10-readme-sync/results.md) | **已完成** |
| Phase 11：Phase 6/9 正交组合验证 | [plan.md](phase-11-hybrid-compensation/plan.md) | [results.md](phase-11-hybrid-compensation/results.md) | **4 项合成离线机器门禁 PASS；Phase 12 后续结果仅为探索，组合收益未证实** |
| Phase 12：无实机影子推理、回放与算法离线复验 | [plan.md](phase-12-prehardware-validation/plan.md) | [results.md](phase-12-prehardware-validation/results.md) | **未通过整体验收；S0–S2 有本机通过项，A1/A2 需重做** |

## 当前精度结论

- **Phase 11 合成离线结果**：先修正局部右乘残差与世界 Y 轴投影不一致，再训练世界系投影版模型。Oracle P9-only 的合成 Swing 残差由 2.1799° 降至 0.7261° (Hybrid)，改善 **+1.4538° (66.7%)**（95% CI `[+1.2911°, +1.6028°]`，胜率 96.8%）。30 分钟合成会话端点收益 **+141.7191°**，与 P9 ZARU 的 +141.7188° 几乎相同。Phase 11 的 oracle 结论仍是历史结果；Phase 12 用实际因果 P9 重算时复用了已评过的 FinalConfirm，且 P9-only 与 Raw 完全相同，因此只提供 P6 单模型探索信号，不能确认 P9+P6 组合、FK 或足滑增益。详见[结果](phase-11-hybrid-compensation/results.md)与[Phase 12 复核](phase-12-prehardware-validation/results.md)。
- **Phase 9 最新 M1r/M1r-F 复核**：合成报告的 96.6% 是 10 条 30 分钟模拟会话的 Full 均值结果，能作为后续研究信号，不是硬件精度结论。Phase 12 严格零漂报告给出的 0.008977° endpoint 数值低于 0.05°，但静态姿势门禁没有执行且仿真角速度来源与线上不一致，不能据此认定 Full 零漂安全。真实 SlimeVR 30 分钟终点收益仍未知，详见[复核结果](phase-09-stateful-bias-model/results.md)与[Phase 12 复核](phase-12-prehardware-validation/results.md)。
- **D3c 连续模型 B**：在 796 条完整逐帧序列（约 16.3 万帧）上训练；反事实方向一致率为 **48.97%**（开发集）与 **51.16%**（新源序列），中位数预测/真值差值比约 **0.006%**。该模型在当前短片段未展示可用补偿，不能外推为分钟级或所有长期估计器均无效。
- **C 评测无效的关键原因**：世界坐标注入的偏航漂移应由世界逆旋转左乘修正；现代码把世界慢偏航右乘到局部残差，并用脚接触/小腿阈值更新大腿与小腿。10°/min 退化约 0.19°、零漂移退化约 0.08° 是当前实现的结果，不能归因于正确坐标的偏置滤波器。
- **工程安全边界**：所有权重仍仅作 Phase 5 影子探针，严禁在线覆写下肢姿态；详见 [Phase 6 复核](phase-06-accuracy-baseline/results.md)。
- **Phase 6/9 组合判断**：可将 Phase 9 用作慢世界偏航分支，将经 yaw-neutral 约束并重训的 Phase 6 用作短时 swing/tilt 分支；现有完整四元数残差不能直接叠加。执行路线见 [Phase 11](phase-11-hybrid-compensation/plan.md)。
