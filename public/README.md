# SlimeVR 项目阶段文档

项目文档按进展阶段归档。根目录的 [plan.md](plan.md) 是状态索引；每个阶段目录是该阶段独立的 `public` 文档区，并固定包含：

- `plan.md`：该阶段的目标、范围、任务和验收条件。
- `results.md`：已执行工作的结果、证据链接、限制和未完成项。未执行阶段会明确标为待执行，不填造结果。

| 阶段 | 主题 | 状态 | 文档 |
|---|---|---|---|
| Phase 1–3 | v1 原型、训练与 ONNX 工程基线 | 已完成 | [计划](phase-01-03-v1-prototype/plan.md) · [结果](phase-01-03-v1-prototype/results.md) |
| Phase 4 | v2 数据契约与调研 | 已完成 | [计划](phase-04-v2-contract-research/plan.md) · [结果](phase-04-v2-contract-research/results.md) |
| Phase 5 | 服务端影子桥接 | 未完成：已有接入骨架，推理与修正链路待实现 | [计划](phase-05-server-shadow-mode/plan.md) · [结果](phase-05-server-shadow-mode/results.md) |
| Phase 6 | 序列隔离评测与补偿精度改进 | D3c 连续模型 B 无收益；C 坐标/观测待修正，现有权重仅影子模式 | [计划](phase-06-accuracy-baseline/plan.md) · [结果](phase-06-accuracy-baseline/results.md) |
| Phase 7 | 合成回放与离线验收 | 合成测试有结果；服务端端到端回放未完成 | [计划](phase-07-replay-validation/plan.md) · [结果](phase-07-replay-validation/results.md) |
| Phase 8 | 实机采集与在线迁移 | 待硬件就绪 | [计划](phase-08-device-validation/plan.md) · [结果](phase-08-device-validation/results.md) |
| Phase 9 | 四路下肢偏航状态模型重构 | 合成结果积极；Full 零漂安全未证实，实机待验收 | [计划](phase-09-stateful-bias-model/plan.md) · [结果](phase-09-stateful-bias-model/results.md) |
| Phase 10 | README 与阶段成果同步 | 已完成 | [计划](phase-10-readme-sync/plan.md) · [结果](phase-10-readme-sync/results.md) |
| Phase 11 | Phase 6/9 正交组合验证 | 4 项合成离线门禁 PASS；服务端字段、FK/脚滑与实机尚未验收，H4 待 Phase 8 | [计划](phase-11-hybrid-compensation/plan.md) · [结果](phase-11-hybrid-compensation/results.md) |

模型、数据集、评测脚本与 JSON 机器报告继续保存在 `ai_corrector/`；阶段 `results.md` 记录摘要并链接到这些原始产物，不复制二进制数据或报告内容。原先 `nextplan.md` 的待执行内容已并入 Phase 6 子计划。README 同步记录在 [Phase 10](phase-10-readme-sync/results.md)。
