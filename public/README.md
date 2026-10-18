# SlimeVR 项目阶段文档

项目文档按进展阶段归档。根目录的 [plan.md](plan.md) 是总计划与状态索引，[results.md](results.md) 是全阶段核心指标与成果总看板；每个阶段目录是该阶段独立的 `public` 文档区，并固定包含：

- `plan.md`：该阶段的目标、范围、任务和验收条件。
- `results.md`：已执行工作的结果、证据链接、限制和未完成项。未执行阶段会明确标为待执行，不填造结果。

| 阶段 | 主题 | 状态 | 文档 |
|---|---|---|---|
| Phase 1–3 | v1 原型、训练与 ONNX 工程基线 | 已完成 | [计划](phase-01-03-v1-prototype/plan.md) · [结果](phase-01-03-v1-prototype/results.md) |
| Phase 4 | v2 数据契约与调研 | 已完成 | [计划](phase-04-v2-contract-research/plan.md) · [结果](phase-04-v2-contract-research/results.md) |
| Phase 5 | 服务端影子桥接 | 有本机 ONNX shadow bridge 原型和通过的测试；服务关闭释放、干净环境模型路径及独立仓库提交仍待完成 | [计划](phase-05-server-shadow-mode/plan.md) · [结果](phase-05-server-shadow-mode/results.md) |
| Phase 6 | 序列隔离评测与补偿精度改进 | D3c 连续模型 B 无收益；C 坐标/观测待修正，现有权重仅影子模式 | [计划](phase-06-accuracy-baseline/plan.md) · [结果](phase-06-accuracy-baseline/results.md) |
| Phase 7 | 合成回放与离线验收 | 合成测试有结果；服务端端到端回放未完成 | [计划](phase-07-replay-validation/plan.md) · [结果](phase-07-replay-validation/results.md) |
| Phase 8 | 实机采集与在线迁移 | 待硬件就绪 | [计划](phase-08-device-validation/plan.md) · [结果](phase-08-device-validation/results.md) |
| Phase 9 | 四路下肢偏航状态模型重构 | 合成漂移收益积极；Phase 12 严格零漂 endpoint 数字为 0.008977°，但静态保持门禁无效且输入链路不一致；实机待验收 | [计划](phase-09-stateful-bias-model/plan.md) · [结果](phase-09-stateful-bias-model/results.md) |
| Phase 10 | README 与阶段成果同步 | 已完成 | [计划](phase-10-readme-sync/plan.md) · [结果](phase-10-readme-sync/results.md) |
| Phase 11 | Phase 6/9 正交组合验证 | 4 项合成离线门禁 PASS；Phase 12 的复用集评测仅支持 P6 单模型探索收益，未证明 P9+P6 组合增益；H4 待 Phase 8 | [计划](phase-11-hybrid-compensation/plan.md) · [结果](phase-11-hybrid-compensation/results.md) |
| Phase 12 | 无实机影子推理、回放与算法离线复验 | 未通过整体验收：13 项 Kotlin 测试通过；S2 逐帧 108,000 次推理、P95 0.657ms；S1 缺 Session 关闭调用；A1 静态门禁硬编码且观测链路不匹配；A2 复用旧 FinalConfirm 且 P9-only 等于 Raw | [计划](phase-12-prehardware-validation/plan.md) · [结果](phase-12-prehardware-validation/results.md) |
| Phase 13 | 无硬件软件闭环与可复现交付准备 | 已规划、待执行：服务端生命周期与完整 tick 回放、有效离线门禁、双固件可重复构建、PC 模拟集成和到货检查包；不扩容模型 | [计划](phase-13-no-hardware-readiness/plan.md) · [结果](phase-13-no-hardware-readiness/results.md) |

模型、数据集、评测脚本与 JSON 机器报告继续保存在 `ai_corrector/`；阶段 `results.md` 记录摘要并链接到这些原始产物，不复制二进制数据或报告内容。原先 `nextplan.md` 的待执行内容已并入 Phase 6 子计划。README 同步记录在 [Phase 10](phase-10-readme-sync/results.md)。
