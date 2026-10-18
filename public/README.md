# SlimeVR 项目阶段文档

项目文档先按功能模块归档，再在模块内按阶段/中间版本分目录。根目录的 [plan.md](plan.md) 是总计划与状态索引，[results.md](results.md) 是全阶段核心指标与成果总看板；每个 Phase 子目录是该阶段独立文档区，并固定包含：

- `plan.md`：该阶段的目标、范围、任务和验收条件。
- `results.md`：已执行工作的结果、证据链接、限制和未完成项。未执行阶段会明确标为待执行，不填造结果。

| 功能模块 | 内容 | 阶段目录 |
|---|---|---|
| `model-development/` | 模型原型、数据契约、精度研究、偏航状态模型与融合架构 | Phase 1–4、6、9、11、15 |
| `runtime-integration/` | 服务端影子推理与回放接入 | Phase 5、7 |
| `system-validation/` | 无硬件的软件链路与系统验收 | Phase 12–14 |
| `hardware-validation/` | 板卡 bring-up、实机采集与在线迁移 | Phase 8 |
| `project-documentation/` | 项目文档治理与 README 同步 | Phase 10 |

| 阶段 | 主题 | 状态 | 文档 |
|---|---|---|---|
| Phase 1–3 | v1 原型、训练与 ONNX 工程基线 | 已完成 | [计划](model-development/phase-01-03-v1-prototype/plan.md) · [结果](model-development/phase-01-03-v1-prototype/results.md) |
| Phase 4 | v2 数据契约与调研 | 已完成 | [计划](model-development/phase-04-v2-contract-research/plan.md) · [结果](model-development/phase-04-v2-contract-research/results.md) |
| Phase 5 | 服务端影子桥接 | 有本机 ONNX shadow bridge 原型和通过的测试；服务关闭释放、干净环境模型路径及独立仓库提交仍待完成 | [计划](runtime-integration/phase-05-server-shadow-mode/plan.md) · [结果](runtime-integration/phase-05-server-shadow-mode/results.md) |
| Phase 6 | 序列隔离评测与补偿精度改进 | D3c 连续模型 B 无收益；C 坐标/观测待修正，现有权重仅影子模式 | [计划](model-development/phase-06-accuracy-baseline/plan.md) · [结果](model-development/phase-06-accuracy-baseline/results.md) |
| Phase 7 | 合成回放与离线验收 | Phase 13 的 108,000 帧 mock Tracker → `tickServer()` 回放已通过基础性能和 shadow 断言；时间戳/故障矩阵及协议层端到端验收仍未完成 | [计划](runtime-integration/phase-07-replay-validation/plan.md) · [结果](runtime-integration/phase-07-replay-validation/results.md) |
| Phase 8 | 实机采集与在线迁移 | 待硬件就绪 | [计划](hardware-validation/phase-08-device-validation/plan.md) · [结果](hardware-validation/phase-08-device-validation/results.md) |
| Phase 9 | 四路下肢偏航状态模型重构 | Phase 13 合成严格零漂 endpoint 为 0.008933°；±0.14°/min 是仿真档位，直接陀螺偏置输入与服务端姿态差分不一致；实机待验收 | [计划](model-development/phase-09-stateful-bias-model/plan.md) · [结果](model-development/phase-09-stateful-bias-model/results.md) |
| Phase 10 | README 与阶段成果同步 | 已完成 | [计划](project-documentation/phase-10-readme-sync/plan.md) · [结果](project-documentation/phase-10-readme-sync/results.md) |
| Phase 11 | Phase 6/9 正交组合验证 | 4 项合成离线门禁 PASS；Phase 12 的复用集评测仅支持 P6 单模型探索收益，未证明 P9+P6 组合增益；H4 待 Phase 8 | [计划](model-development/phase-11-hybrid-compensation/plan.md) · [结果](model-development/phase-11-hybrid-compensation/results.md) |
| Phase 12 | 无实机影子推理、回放与算法离线复验 | 未通过整体验收：13 项 Kotlin 测试通过；S2 逐帧 108,000 次推理、P95 0.657ms；S1 缺 Session 关闭调用；A1 静态门禁硬编码且观测链路不匹配；A2 复用旧 FinalConfirm 且 P9-only 等于 Raw | [计划](system-validation/phase-12-prehardware-validation/plan.md) · [结果](system-validation/phase-12-prehardware-validation/results.md) |
| Phase 13 | 无硬件软件闭环与可复现交付准备 | **部分完成**：P3 mock tick 基础子门禁通过（108,000 帧逐帧零写回，P95 4.638ms），但时间、故障恢复及 reset 后状态检查不足；P4 合成门禁通过；P6 缺固件源码 clean build 与 UDP/HID 协议 fixture；P7 缺协议输入和 GUI/服务冒烟；P5 数据受限待新数据 | [计划](system-validation/phase-13-no-hardware-readiness/plan.md) · [结果](system-validation/phase-13-no-hardware-readiness/results.md) |
| Phase 14 | 无硬件软件链路闭环 | **部分完成**：UDP parser/派发至 `tickServer()` 的 10.8 万帧核心回放和模型场景测试通过（0 篡改、最新 P95 4.733ms）；时间间隔/完整故障矩阵未覆盖，固件 clean build 与 GUI/package 未完成 | [计划](system-validation/phase-14-software-closure/plan.md) · [结果](system-validation/phase-14-software-closure/results.md) |
| Phase 15 | 固定参数预算的双时间尺度模型架构 | **已完成**：严格锁定 1,102,066 参数（1.10M），150 条 30 分钟连续会话（270 万帧）消融通过；LSM6DSV 标称漂移下端点误差消除 83.2%~84.4%，严格零漂与静坐门禁 100% PASS | [计划](model-development/phase-15-fixed-budget-dual-timescale/plan.md) · [结果](model-development/phase-15-fixed-budget-dual-timescale/results.md) |

模型、数据集、评测脚本与 JSON 机器报告继续保存在 `ai_corrector/`；阶段 `results.md` 记录摘要并链接到这些原始产物，不复制二进制数据或报告内容。原先 `nextplan.md` 的待执行内容已并入 Phase 6 子计划。README 同步记录在 [Phase 10](project-documentation/phase-10-readme-sync/results.md)。
