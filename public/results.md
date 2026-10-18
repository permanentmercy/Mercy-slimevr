# SlimeVR CSTN AI Corrector 项目总成果与指标汇总

> **文档定位**：本项目所有阶段的**总体成果、核心精度数据与工程性能指标的总汇总看板**。  
> **更新时间**：2026-10-18  
> **数据版本**：Phase 1 至 Phase 14 全阶段文档；Phase 14 部分完成，最新测试复跑数据已更新；Phase 12 的 A1/A2 原 PASS 已撤销

---

## 一、 项目全阶段状态与交付结论总览

| 阶段 | 阶段主题 | 核心职责 | 交付成果与关键指标 | 当前状态与裁定 |
|---|---|---|---|---|
| **Phase 1–3** | **v1 原型与 ONNX 工程基线** | 60Hz 采样契约建立，初始 112 维特征流定义与模型骨架打通 | 完成初始数据流定义与 PyTorch 到 ONNX 导出基础链路 | [已完成并归档](phase-01-03-v1-prototype/results.md) |
| **Phase 4** | **v2 数据契约与调研** | 确立 124 维分层流式契约（上身 4 路视觉 68 维 + 下肢 4 路 IMU 56 维） | 锁定 124 维流式特征协议，定义 6D 连续旋转与传感器元数据 | [已完成契约锁定](phase-04-v2-contract-research/results.md) |
| **Phase 5** | **服务端影子桥接工程** | 在 SlimeVR Kotlin 服务端 `server:core` 实现采样拦截与桥接骨架 | 已有采样/推理原型；服务关闭释放与完整服务端回放待验收 | [仍有工程待办](phase-05-server-shadow-mode/results.md) |
| **Phase 6** | **序列隔离与补偿精度基线** | 建立 796 条时序集对旧 CSTN 全量残差模型审计 | 揭示旧模型世界偏航局部右乘导致的物理退化，确立正交分频解耦路线 | [已完成审计复核](phase-06-accuracy-baseline/results.md) |
| **Phase 7** | **合成回放与离线验收** | 构建 60Hz 合成时序回放机制，制定延迟与稳定性评测规范 | 建立 10.8 万帧连续回放评测标准，由 Phase 12 落地实测 | [由 Phase 12 承接落地](phase-07-replay-validation/results.md) |
| **Phase 8** | **实机采集与在线迁移** | 规划实机硬件日志采集协议、独立动捕真值环境对照与在线姿态应用 | 规划硬件部署、光学动捕对照方案与在线姿态解算 | [待硬件就绪后启动](phase-08-device-validation/plan.md) |
| **Phase 9** | **四路偏航状态模型研究** | 设计显式偏航状态卡尔曼滤波原型 `DeployableBiasEstimatorV3` | 合成 30 分钟会话曾显示抑制信号；零漂安全、真实观测语义和实机效果未证实 | [结果限制与复核](phase-09-stateful-bias-model/results.md) |
| **Phase 10** | **README 与阶段成果同步** | 规范化多阶段 `public/` 文档管理与团队协作流程 | 统一全阶段双层文档架构与自动化索引规范 | [已完成](phase-10-readme-sync/results.md) |
| **Phase 11** | **Phase 6/9 正交组合探索** | 提出世界 Y 轴偏航中立约束（Yaw-Neutral CSTN） | 投影算子测试通过；Oracle Swing 收益仅为合成探索，Phase 12 未证实实际 P9+P6 增量 | [结果限制与复核](phase-11-hybrid-compensation/results.md) |
| **Phase 12** | **无实机影子推理、回放与算法复验** | 契约/推理测试与逐帧特征组装回放；复核服务生命周期、真实 tick、物理输入及数据隔离 | **未通过整体验收**：13 项指定 Kotlin 测试通过，108,000 次逐帧特征组装+ONNX P95 0.6571 ms；完整 `VRServer.tick()`、Session close 和内存门限未验证；A1/A2 不接受原 PASS | [阶段复核报告](phase-12-prehardware-validation/results.md) |
| **Phase 13** | **无硬件软件闭环与可复现交付准备** | 生命周期、tickServer mock 回放、合成物理门禁、固件构建准备、PC 构建与模拟集成 | **部分完成**：P3 的 108,000 帧动态 mock tick、性能和 shadow 断言通过，但时间/故障矩阵/reset 后状态与输出检查不足；本次 P95 4.638ms、P99 7.810ms、堆差 −0.38MB。P4 合成门禁不代表实机；P6 缺固件源码 clean build 与 UDP/HID 协议 fixture；P7 缺协议输入、GUI/安装包本轮构建和启动冒烟；P5 数据受限 | [阶段成果](phase-13-no-hardware-readiness/results.md) |
| **Phase 14** | **无硬件软件链路闭环** | 补齐 tracker UDP → Server tick/shadow 回放、两套固件源码构建与协议 fixtures、PC 构建和启动/关闭冒烟 | **部分完成**：108,000 帧核心回放、shadow 零写回、模型场景定向测试通过；最新 P95 4.733ms、P99 7.090ms、最大 20.436ms、Full GC 后堆差 +0.128MB。完整时间异常/故障矩阵未验；固件源码、GUI/package 未构建 | [阶段成果](phase-14-software-closure/results.md) |

---

## 二、 算法架构与解耦理论体系

针对 IMU 动作捕捉中存在的“低频积分发散漂移”与“高频动作失真残差”混合交织的问题，项目研究了**双分支正交分频解耦候选架构**。下图表达算法构想，不代表两个分支均已通过真实输入、独立数据或实机验收：

```
                服务端姿态/运动特征 (R, acc, omega*; 角速度语义待统一) + 上身上下文
                                         │
                                         ▼
                                 124 维标准时序流
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │                                               │
                 ▼                                               ▼
     【慢分支：物理状态估计】                         【快分支：神经运动学修正】
  Phase 9 DeployableBiasEstimatorV3              Phase 11/12 Yaw-Neutral CSTN (ONNX)
  - 预期以原始陀螺输入做 4 路 ZARU (当前输入未证实) - 8 路因果流式 LSTM (Hidden Dim=96)
  - 步态辅助航向观测候选 (可观测性待验证)            - 针对下肢 4 关节输出 3D 旋转残差
  - 四路 2x2 状态协方差卡尔曼滤波 (EKF)            - 内嵌世界 Y 轴 Swing-Twist 正交投影
  - 估计四路世界竖直轴慢偏航误差 e_i(t)            - 严格剥离世界偏航 Twist，仅保留高频 Swing
                 │                                               │
                 ▼                                               ▼
         世界逆偏航左乘修正                              局部右乘 Swing 补偿
       R_P9 = R_y(-e_i) @ R_raw                        R_hybrid = R_P9 @ dq_swing_gated
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                            候选输出 (当前仅影子观测)
                       不写入求解器；实机效果尚待验证
```

`omega*` 可能是姿态差分代理量，当前不能当作原始陀螺读数；完整角速度数据来源和 ZARU 可部署性列入 [Phase 13 P1](phase-13-no-hardware-readiness/plan.md)。

### 正交分频算子性质（不等于整体模型精度通过）
世界竖直轴偏航分解为四元数 Swing-Twist：
$$q = q_{\text{swing}} \otimes q_{\text{twist}},\quad q_{\text{twist}} = \frac{[w, 0, y, 0]^T}{\sqrt{w^2 + y^2}}$$
$$q_{\text{neutral}} = q \otimes q_{\text{twist}}^*$$
输出残差 $\Delta q$ 在世界系下经正交投影剥离世界 Y 轴旋转量；单元/合成条件下可验证投影算子的偏航串扰接近数值零。该性质不证明 P9 慢分支输入、训练数据、部署模型或 Avatar 结果准确。

---

## 三、 全阶段核心量化指标汇总看板

### 1. 服务端工程与运行时局部指标 (Phase 12 Task S0–S2 实测；非整体验收)

所有测试均在本地 Java 17 + Gradle 8.11 环境下，通过真实 ONNX Runtime 1.18.0 CPU 引擎驱动执行。

| 性能与工程指标 | 验收目标 / 门限 | 实际测量值 | 优于门限倍数 / 状态 | 测试用例与证据源 |
|---|---|---|---|---|
| **Kotlin 单元测试通过率** | 100% 全部通过 | **13 / 13 全通 (100%)** | 全部绿灯 | `CorrectorBridgeContractTest` 等 3 个套件 |
| **124D 契约哨兵逐元素比对** | 最大绝对误差 $< 10^{-3}$ | **$< 1.83 \times 10^{-4}$** | 优于门限 5.4 倍 | `fixture_124d_sentinel.json` 契约比对 |
| **特征组装+ONNX 连续回放稳定性** | 108,000 帧逻辑流无崩溃 | **108,000 帧通过** | 不覆盖完整 `VRServer.tick()` | `CorrectorBridgeReplayBenchmarkTest` |
| **特征组装+ONNX 回放 P95** | 参考门限 $\le 20.0$ ms | **0.6571 ms ~ 0.7301 ms** | 子路径测试值，非服务端端到端结果 | `s2_replay_benchmark_report.json` |
| **特征组装+ONNX 回放 P50** | - | **0.5234 ms ~ 0.5512 ms** | 子路径测试值 | 3,000 样本实测基准 |
| **特征组装+ONNX 回放 P99** | - | **0.7698 ms ~ 0.8447 ms** | 子路径测试值 | 3,000 样本实测基准 |
| **特征组装+ONNX 回放最大值** | - | **1.3687 ms ~ 1.3693 ms** | 子路径测试值 | 3,000 样本实测基准 |
| **ONNX 模型前向推理延迟 P95** | - | **0.6501 ms ~ 0.7241 ms** | 占总耗时 99.1% | ONNX Runtime CPU 单线程 |
| **124D 特征提取与正交化 P95** | - | **7.00 $\mu\text{s}$ (0.007 ms)** | 仅占总耗时 0.9% | 纯标量优化 Gram-Schmidt |
| **回放前后 GC 堆差值** | 需预先定义堆/常驻内存门限 | **Delta = -0.517 MB ~ -0.60 MB** | 单次观察值；没有内存门限，不能判定无泄漏 | 108,000 帧前后强制 Full GC 观测 |
| **异常输入安全旁路测试** | 正常流 0 误触，NaN 100% 旁路 | **正常 0/3000，异常 100% 隔离** | 状态自愈，零污染后续帧 | NaN / Inf 注入测试套件 |
| **影子模式姿态保真度** | 姿态 0 篡改 | **原始 Tracker 姿态逐位恒等** | 严格只读旁路 | `shadowMode = true` 验证 |

> [!NOTE]
> **服务端集成边界说明**：
> - 当前延迟只覆盖特征组装和 ONNX 推理，不覆盖完整 `VRServer.tick()`；不能称作全服务端端到端延迟。
> - `CorrectorBridge` 实现了 `AutoCloseable`，但复核未找到服务关闭路径调用 `close()`；生命周期尚未验收。
> - 13 项指定 Kotlin 测试通过不等于 Phase 12 整体通过；S1/S2 尚有上述缺口。

---

### 2. Phase 13 无硬件软件验收复核（部分完成）

本次服务端定向回归使用 Java 17、Gradle 8.14.4 和 CPU ONNX Runtime；`CorrectorBridgeInferenceTest` 7/7、`CorrectorBridgeReplayBenchmarkTest` 3/3 通过。10 项测试约 9 分 28 秒完成。

#### A. `VRServer.tickServer()` mock 回放 (Task P2/P3)
长程测试持续生成腿部姿态和状态故障，逐帧调用 `VRServer.tickServer()` 并检查四路下肢 Tracker 的 shadow 不写回。它没有启动真实网络收包循环，输入也不是协议 packet；同时没有断言每次 reset/故障后的状态恢复、完整时间戳和 Tracker 故障矩阵或每帧模型输出有效性。因此当前只通过了 tick mock 的性能与零写回子门禁，P3 整体仍为部分完成。

| 性能与工程指标 | 验收目标 / 门限 | 实际测量值 | 状态 | 测试用例与证据源 |
|---|---|---|---|---|
| **服务端主生命周期调用** | 主循环中断/退出时可靠释放 | **`VRServer` 实现 `AutoCloseable`**<br>`finally { close() }` 释放 bridge | **PASS** | `CorrectorBridgeInferenceTest.test06` |
| **缺失模型容错降级** | 自定义模型不存在时不崩溃/不误读 | **安全返回 null 并禁用修正** | **PASS** | `CorrectorBridgeInferenceTest.test07` |
| **tickServer P95** | 实时预算 $\le 20.0$ ms | **4.6379 ms** | 子门禁通过 | 3,000 帧；本轮机器报告 |
| **tickServer P50 / P99 / 最大值** | 记录分布 | **1.9618 / 7.8096 / 19.6591 ms** | P95 门限通过 | 最大样本接近 20ms；不能声称硬实时保证 |
| **CorrectorBridge 耗时 P50 / P95 / P99** | 记录分布 | **1.8047 / 4.2919 / 7.0650 ms** | 记录值 | 同一测试运行 |
| **108,000 次逻辑 tick** | 动态回放和故障恢复 | **完成** | mock tick 通过 | 姿态持续变化，59 次 reset、868 帧胸部/小腿状态异常 |
| **Full GC 后堆差** | $\le 64.0$ MB | **−0.3801 MB** | 单次观察通过 | 不是对所有资源泄漏的证明 |
| **shadow 写回** | 四路下肢逐帧恒等 | `shadow_mutation_count = 0` | **mock tick 通过** | 测试每帧比较 Tracker 原始/求解旋转；未经过协议接收循环 |

#### B. 合成物理仿真与动态安全门禁 (Task P4)
原报告包含 120 条 30 分钟合成会话（216 万帧），门禁由脚本计算。±0.14°/min 是脚本设定的合成偏置；ST LSM6DSV 数据手册未给出该数值作为标称偏航漂移率。数据手册提供的是零速率水平与温度变化等器件参数，不能据此把仿真偏置称为芯片标称值（[ST LSM6DSV 数据手册](https://www.st.com/resource/en/datasheet/lsm6dsv.pdf)）。同时，仿真直接注入陀螺角速度偏置，而当前服务端使用融合姿态差分，结果不得外推到服务端或实机。

| 工况场景 | 样本量 | 门限要求 | 实测指标 | 状态 | 物理学理论与安全结论 |
|---|---|---|---|---|---|
| **严格零漂移 (`strict_zero_drift`)** | 10 会话 | 最大端点偏差 $\le 0.05^\circ$ | **0.008933°** | **PASS** | 零漂输入下卡尔曼不产生伪发散 |
| **静态端坐零漂移 (`static_sitting`)** | 10 会话 | 最大端点偏差 $\le 0.10^\circ$ | **0.000000°** | **PASS** | 端坐静止下自动锁定航向，无伪修正 |
| **合成正偏置 (+0.14°/min)** | 30 会话 | 胜率 $\ge 90\%$, CI 下界 $> 0$ | 胜率 **100%**, CI `[+12.80°, +15.50°]` | **PASS** | 净收益 +14.06°, 漂移抑制 84.0% (16.73° $\to$ 2.67°) |
| **合成负偏置 (-0.14°/min)** | 30 会话 | 胜率 $\ge 90\%$, CI 下界 $> 0$ | 胜率 **100%**, CI `[+12.51°, +14.75°]` | **PASS** | 净收益 +13.66°, 漂移抑制 82.7% (16.50° $\to$ 2.85°) |
| **端坐含漂移补偿 (0.14°/min)** | 10 会话 | CI 下界 $> 0$ | 胜率 **100%**, CI `[+13.04°, +17.86°]` | **PASS** | 净收益 +15.64°, 静止漂移抑制 96.5% (16.21° $\to$ 0.57°) |
| **持续转身漫步 (合成 0.14°/min, 无静止窗)** | 10 会话 | CI 下界 $> 0$ | 胜率 **100%**, CI `[+5.44°, +9.25°]` | **PASS** | 净收益 +7.34°, 步态分析防发散 (17.73° $\to$ 10.39°) |
| **纯随机游走 (零均值偏置)** | 10 会话 | CI 下界 $> 0$ | 胜率 **90%**, CI `[+0.81°, +3.78°]` | **均值门限通过** | Full 最差单会话净收益 **−1.6389°**，仍存在退化样本 |
| **坐标系数学正确性校验** | 解析 | 倾斜端坐姿态下旋转恢复 | 世界左乘测地误差 $< 10^{-6}$ rad | **PASS** | 局部右乘因不可交换性导致物理退化 |

#### C. PC 端与固件交付准备 (Task P6/P7)
- **SlimeVR Desktop Jar 与 SteamVR Driver**：阶段执行记录有 `slimevr.jar` 和 `driver_slimevr.dll` 的源码构建产物与 SHA-256；本轮独立复核了文件存在及哈希。定向 Corrector 测试 10/10 通过。
- **GUI/安装包**：`server/gui/dist/artifacts/win/SlimeVR-win-x64.zip` 存在，但时间戳为 2026-10-11；不是本轮 clean build 证据，也没有本轮启动冒烟记录。
- **固件**：六个既有 UF2/HEX 文件 SHA-256 与记录一致；本机无 `west` 命令且没有 tracker/receiver 构建目录，两套源码 clean build 和 UDP/HID 协议 fixture 未执行。
- **端到端集成**：现有长测从 mock Tracker 对象调用 `tickServer()`，未从 tracker UDP 协议输入建立 packet fixture；GUI 默认配置、服务启动/退出、缺模型整链路降级仍待验证。
- **实机 Bring-up 清单**：已建立操作清单；板卡型号、bootloader、地址布局、USB 枚举和串口命令仍需按到货硬件验证。

**P6/P7 状态**：部分完成。下一轮先完成固件 clean build 和 parser fixtures，再完成 GUI/安装包构建、协议层 tracker 模拟输入与 PC 启动冒烟后复核阶段验收。

---

### 3. Phase 14 无硬件软件链路闭环验收（部分完成）

按 [Phase 14 计划](phase-14-software-closure/plan.md) 在本地 JDK 17、Gradle 8.14.4 与 ONNX Runtime 1.18.0 环境下执行。

#### A. Tracker UDP 协议至 `VRServer.tickServer()` 核心回放 (Task P14.1，整体部分完成)
测试套件 [`UDPProtocolCorrectorBenchmarkTest.kt`](../server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/UDPProtocolCorrectorBenchmarkTest.kt)：独立复跑 **3/3 全部通过**。108,000 帧由 `feedRawPacketForTesting` seam 进入生产 parser/派发和 `tickServer()`；没有经过 socket 接收线程。

| 性能与工程指标 | 验收目标 / 门限 | 实际测量值 | 状态 | 验证方法与证据源 |
|---|---|---|---|---|
| **UDP Socket 回环冒烟** | 接收线程接收并处理 handshake | **只断言接收线程存活** | 部分证据 | `test01_udpLoopbackSocketSmoke` 未断言 packet 被解析/注册 |
| **协议包解析与故障丢弃** | 乱序/非法/未知包安全降级 | **乱序抛 IOException 丢弃<br>未知包安全返回 null<br>NaN 姿态安全退避 IDENTITY** | **PASS** | `test02_udpGoldenPacketHexFixturesAndFaults` |
| **Full Server Tick 耗时 P95** | 实时预算 $\le 20.0$ ms | **4.7331 ms** | **P95 门限通过** | 复跑 3,000 帧；Mean 2.502ms, P50 2.255ms, P99 7.090ms, Max 20.436ms（最大值超过 20ms） |
| **CorrectorBridge 推理 P95** | 记录分布 | **4.4153 ms** (P50 2.097ms, P99 6.249ms) | 记录值 | 当前机器一次复跑 |
| **108,000 帧连续逻辑回放** | 30 分钟连续步态运动流 | **108,000 帧全部完成** | **PASS** | `test03_fullUDPProtocolToVRServerTick108kFramesReplay` |
| **故障注入与明确恢复断言** | 计划规定的完整故障矩阵 | 计数为 948 / 125 | **部分完成** | 125 含 reset、状态赋值和乱序包拒绝，不全是 tracker 恢复；仅左小腿有恢复状态断言，缺 dt 异常和四路 tracker 矩阵 |
| **影子模式零篡改断言** | 逐帧 Tracker 物理姿态 0 改变 | **`shadow_mutation_count = 0`** | **PASS** | 严格零篡改，`preTick == postTick` 逐位恒等 |
| **模型输出有效性断言** | 有限数值且归一化 | **`non_finite_output_count = 0`** | **PASS** | 无任何 NaN / Inf 泄漏 |
| **长程内存稳定性与堆差** | Full GC 堆差 $\le 64.0$ MB | **Delta = +0.128 MB** | **堆差门限通过** | Baseline 9.79MB, Watermark 29.99MB, 检查点 `[28.97, 21.97, 14.55, 29.99, 24.98]`；代码趋势断言较宽松 |

- **机器报告**：[`ai_corrector/artifacts/phase14_p1_udp_server_tick_benchmark_report.json`](../ai_corrector/artifacts/phase14_p1_udp_server_tick_benchmark_report.json)
- **范围限制**：测试故障条件实际产生 59 次 reset、30 个胸部断连窗口（629 个故障帧）、15 个左小腿超时窗口（239 个故障帧）及 21 次旧序号包。JSON 的 125 个 `asserted_recoveries` 混合了 reset 断言、胸部状态赋值、左小腿恢复断言和旧序号包拒绝，不能按 125 次 tracker 自愈解读。长回放固定 `sampleInterval=1/60`，没有 dt 异常和四路 tracker 全矩阵；P14.1 因此部分完成。

#### B. PC 构建、启动冒烟与模型 4 场景测试 (Task P14.3)
测试套件 [`ServerStartupSmokeAndFourModelScenariosTest.kt`](../server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/ServerStartupSmokeAndFourModelScenariosTest.kt)：**5/5 全部通过**。

| 检查项 | 目标 / 门限 | 实测结果 | 状态 | 详细信息与凭据 |
|---|---|---|---|---|
| **Server Desktop Jar** | 源码干净编译与打包 | **构建成功** | **PASS** | `slimevr.jar` (114,682,136 bytes, SHA-256: `B50DD7505AA9D38744C95342C88FA45D3D9D9FD102F314E7A4F9A88BF7785EE4`) |
| **默认安全门禁** | 默认禁用 / 影子只读 | **`enabled=false`, `shadowMode=true`** | **PASS** | 未经配置绝不覆写下肢姿态 |
| **场景 A: 有效模型** | 模型加载与会话释放 | **推理有效，关闭后 Session 释放 (`isLoaded==false`)** | **PASS** | 1.54M CSTN v2 ONNX 正常驱动 |
| **场景 B: 模型缺失** | 路径不存在时平稳降级 | **返回 null，服务正常推进，残差保持 IDENTITY** | **PASS** | 无崩溃，无未捕获异常 |
| **场景 C: 损坏模型/非法输入** | 坏文件拦截，NaN/Inf 旁路 | **坏文件拒绝加载；NaN/Inf 特征自动旁路重置状态** | **PASS** | 零污染后续帧 |
| **场景 D: HMD/VD 缺失** | 参考流缺失平稳降级 | **HMD 缺失时旁路；VD 缺失时填零并置 valid=0** | **PASS** | 优雅降级推理 |
| **GUI 与 Windows 安装包** | clean build 打包 | **受工具链缺失阻塞** | 记录限制 | Windows host 未安装 `node` / `pnpm` |

#### C. 固件源码构建受工具链阻塞 (Task P14.2)
- **阻塞原因**：Windows 主机未配置 `west`、`nRF Connect SDK (NCS)` 与 `arm-none-eabi-gcc`。
- **目标设定**：Tracker `slimenrf_r3/nrf52840/uf2`，Receiver `holyiot_21017/nrf52840`。
- **原则遵守**：绝不使用旧发布制品哈希冒充源码构建，诚实登记为环境依赖阻塞。协议包解析与丢包语义已在 P14.1 自动测试中覆盖。物理硬件刷写与配对归 Phase 8。

---

### 4. 慢分支长程偏航物理估计历史数字（Phase 9 / Phase 12 A1）

下表为旧 A1 合成报告的历史数字，**不构成有效验收**。复核确认静态姿势门禁没有计算，`sitting_passed` 未被更新；模拟直接输入带偏置角速度，而服务端当前使用融合姿态差分。该链路不一致使报告中的收益和安全 PASS 不能外推到当前服务端。

| 工况场景 | 测试样本量 | Raw 累计偏航误差 | Full 修正后误差 | 净收益 $G$ | 95% Bootstrap CI | 序列胜率 | 偏航抑制率 |
|---|---|---|---|---|---|---|---|
| **标称正偏航 (+3°/min)** | 30 条会话 | 361.25° | **5.84°** | **+355.41°** | `[+340.72°, +369.80°]` | 100.0% | **98.4%** |
| **标称负偏航 (-3°/min)** | 30 条会话 | 361.28° | **6.20°** | **+355.08°** | `[+339.30°, +369.69°]` | 100.0% | **98.3%** |
| **高偏航漂移 (±10°/min)** | 30 条会话 | 302.27° | **4.43°** | **+297.84°** | `[+254.64°, +336.84°]` | 100.0% | **98.5%** |
| **纯静态端坐 (All Sitting)** | 10 条会话 | 342.93° | **0.34°** | **+342.60°** | `[+322.18°, +362.35°]` | 100.0% | **99.9%** |
| **上身 VD 丢失 (VD Missing)**| 10 条会话 | 360.93° | **6.91°** | **+354.02°** | `[+334.25°, +374.69°]` | 100.0% | **98.1%** |
| **持续转身漫步 (No Still)**  | 10 条会话 | 362.68° | 348.83° | +13.85° | `[+5.11°, +22.67°]` | 70.0% | 步态分析防发散 |
| **纯随机游走 (零均值偏置)**   | 10 条会话 | 4.98° | 2.82° | +2.16° | `[+1.38°, +2.99°]` | 100.0% | 自适应噪声平滑 |

**裁定**：旧报告的 0.008977° endpoint 记录仅是未经有效静态门禁裁定的探索数字；不得称为零漂安全通过。Phase 13 P4 重新实现自动门禁后，才能对明确的合成信号模型给出有限结论；实机结论仍需 Phase 8。

---

### 4. 快分支短时姿态与正交解耦历史数字（Phase 11 / Phase 12 A2）

下表为历史探索结果，不是确认性通过：A2 复用了 Phase 11 已使用的 190 条 FinalConfirm 序列，P9-only 与 Raw 完全相同，P9+P6 与 P6-only 完全相同。不能据此声称新源泛化或证明融合增加了收益。

| 评测维度 | Raw 基线 | 旧报告 P9-only | 快分支 P6-only | 候选组合 P9+P6 | 旧评测差值 | 旧 bootstrap CI | 序列胜率 | 有效性 |
|---|---|---|---|---|---|---|---|---|
| **世界 Swing 测地误差** | 2.052° | 2.052° | 0.700° | **0.700°** | **+1.352°** (改善 65.8%) | `[+1.200°, +1.491°]` | **96.84%** | 探索性显著改善 |
| **全姿态 3D 测地误差**   | 2.052° | 2.052° | 0.700° | **0.700°** | **+1.352°** | - | - | 旧数据探索值 |
| **简化固定骨长 FK 代理** | 21.56 mm | 21.56 mm | 9.19 mm | **9.19 mm** | **+12.37 mm** (探索数字) | `[+10.9mm, +13.8mm]` | 97.4% | 不是 SMPL-H FK |
| **未包含 root 世界位移的脚踝速度代理** | 443.35 mm/s | 443.35 mm/s | 431.84 mm/s | **431.84 mm/s** | **+11.51 mm/s** (探索数字) | `[+8.4mm/s, +14.6mm/s]` | 86.8% | 不是世界足滑验收 |
| **正交偏航串扰度**       | - | - | $< 10^{-6\circ}$ | **$< 10^{-6\circ}$** | 算子测试值 | - | - | 不代表模型精度 |

- **旧压力场景探索记录（非独立确认）**：

| 压力扰动工况 | 世界 Swing 净收益 | 下肢 FK 代理差值 | 脚踝速度代理差值 | 旧报告解释 |
|---|---|---|---|---|
| **名义基准条件** | **+1.352°** | **+12.37 mm** | **+11.51 mm/s** | 基准性能最优 |
| **视觉高斯噪声 ($\sigma=2\text{ cm}, 3^\circ$)** | **+1.352°** | **+12.37 mm** | **+11.58 mm/s** | 无显著精度退化 |
| **视觉延迟 2 帧 (33 ms)** | **+1.352°** | **+12.37 mm** | **+10.37 mm/s** | 时序抖动鲁棒 |
| **视觉输入完全丢失 (VD Missing)** | **+1.352°** | **+12.31 mm** | **+10.05 mm/s** | 纯下肢退化自愈 |
| **传感器安装偏差 ($\pm 3^\circ$)** | **+1.352°** | **+12.37 mm** | **+11.52 mm/s** | 角度安装容差极佳 |

> [!NOTE]
> **有效范围**：Swing 数字仅支持旧合成条件下的 P6-only 探索；FK 使用固定骨长，脚滑未叠加根骨世界位移，均为代理指标。新隔离集上的 P9+P6 增量、真实骨架误差和世界足滑尚未确认，见 [Phase 13 P5](phase-13-no-hardware-readiness/plan.md)。

---

### 5. 模型体量与网络架构规格

| 架构参数 | 设计指标 / 规格 | 说明 |
|---|---|---|
| **网络主干结构** | 8 层因果流式 LSTM + 密集残差全连接头 | 严格因果时序，无未来帧反向传播泄漏 |
| **总参数量** | **1,539,472 参数 ($\sim 1.54\text{ M}$)** | 严格控制在 $< 2.0\text{ M}$ 轻量化门限内 |
| **导出 ONNX 权重体积** | **4.57 MB** (Opset 14) | 优于 $\le 8.0\text{ MB}$ 规范，便于跨平台分发 |
| **流式隐状态张量** | $h, c \in \mathbb{R}^{8 \times 1 \times 1 \times 96}$ | 内存常驻占用极小（仅 12.3 KB） |
| **输入流特征维度** | $1 \times 1 \times 124$ 单帧流式 | 68 维上身参考 + 56 维下肢 IMU 物理量 |
| **输出张量维度** | 残差 $4 \times 4$、门控 $4 \times 1$、触地 $2 \times 1$ | 包含下肢四元数残差、置信门控与脚掌触地概率 |

---

### 6. 早期历史基线与消融对照 (Phase 6 审计数据)

| 对照模型 | 评测序列数 | 预测反事实一致率 | 差值比中位数 | 漂移抑制效果 | 缺陷原因与结论 |
|---|---|---|---|---|---|
| **Model B (全量 3D 残差网络)** | 796 条 (16.3万帧) | 48.97% (开发集)<br>51.16% (新源序列) | 0.006% | 5秒短片段无可用收益 | 未解耦世界漂移与局部动作，网络陷入平均伪解 |
| **Model C (物理慢偏置局部右乘)** | 796 条 (16.3万帧) | - | - | 10°/min 漂移下**恶化 +0.19°**<br>零漂移下**恶化 +0.08°** | **致命缺陷**：将世界坐标偏航漂移错误右乘到局部坐标系；腿部静止检测将动态姿态冲刷 |
| **Phase 11/12 Yaw-Neutral 候选** | 旧合成集/旧确认集 | 算子投影性质可单测 | - | 历史 P6-only Swing 探索数值为正 | **未完成独立确认**：不等同已验证融合方案或最终部署方案 |

---

## 四、 核心交付产物索引表

### 1. 阶段评测报告与黄金哨兵 (JSON 机器凭据)
- **Phase 14 P14.1 Tracker UDP 至 `tickServer()` 10.8 万帧全链路基准报告**：[`ai_corrector/artifacts/phase14_p1_udp_server_tick_benchmark_report.json`](../ai_corrector/artifacts/phase14_p1_udp_server_tick_benchmark_report.json)
- **Phase 13 P3 `tickServer()` mock 回放与延迟基准报告**：[`ai_corrector/artifacts/phase13_p3_server_tick_benchmark_report.json`](../ai_corrector/artifacts/phase13_p3_server_tick_benchmark_report.json)
- **Phase 13 P4 合成物理与动态安全门禁审计报告**：[`ai_corrector/artifacts/phase13_p4_physics_audit_report.json`](../ai_corrector/artifacts/phase13_p4_physics_audit_report.json)
- **S2 服务端连续回放与延迟评测报告**：[`ai_corrector/artifacts/s2_replay_benchmark_report.json`](../ai_corrector/artifacts/s2_replay_benchmark_report.json)
- **A1 慢偏航长程物理审计报告**：[`ai_corrector/artifacts/phase12_a1_physics_audit_report.json`](../ai_corrector/artifacts/phase12_a1_physics_audit_report.json)
- **A2 快分支增量价值评测报告**：[`ai_corrector/artifacts/phase12_a2_p6_incremental_report.json`](../ai_corrector/artifacts/phase12_a2_p6_incremental_report.json)
- **ONNX 流式模型 ABI 导出报告**：[`ai_corrector/artifacts/cstn_v2_yaw_neutral_world.report.json`](../ai_corrector/artifacts/cstn_v2_yaw_neutral_world.report.json)
- **124D 契约跨语言黄金哨兵 Fixture**：[`ai_corrector/artifacts/fixture_124d_sentinel.json`](../ai_corrector/artifacts/fixture_124d_sentinel.json)

### 2. 核心算法、状态机与模型资产
- **世界轴偏航中立流式 ONNX 模型**：[`ai_corrector/artifacts/cstn_v2_yaw_neutral_world.onnx`](../ai_corrector/artifacts/cstn_v2_yaw_neutral_world.onnx)
- **可部署偏航卡尔曼滤波器**：[`ai_corrector/corrector/bias_estimator_v3.py`](../ai_corrector/corrector/bias_estimator_v3.py)
- **四路显式卡尔曼偏置状态机**：[`ai_corrector/corrector/bias_state_v3.py`](../ai_corrector/corrector/bias_state_v3.py)
- **偏航中立神经网络包装器**：[`ai_corrector/corrector/yaw_neutral_cstn.py`](../ai_corrector/corrector/yaw_neutral_cstn.py)
- **正交双分支合成器**：[`ai_corrector/corrector/hybrid_composer.py`](../ai_corrector/corrector/hybrid_composer.py)
- **标量加速契约转换适配器**：[`ai_corrector/corrector/contract_v3.py`](../ai_corrector/corrector/contract_v3.py)
- **Phase 13 仿真门禁评测套件**：[`ai_corrector/corrector/evaluate_phase13_p4_physics_audit.py`](../ai_corrector/corrector/evaluate_phase13_p4_physics_audit.py)
- **Phase 13 坐标系解析与仿真单测**：[`ai_corrector/corrector/test_p4_coordinate_and_simulation.py`](../ai_corrector/corrector/test_p4_coordinate_and_simulation.py)

### 3. SlimeVR 服务端核心工程组件
- **流式 ONNX 运行时推理引擎**：[`OnnxCorrectorEngine.kt`](../server/server/core/src/main/java/dev/slimevr/tracking/processor/corrector/OnnxCorrectorEngine.kt)
- **AI 修正器服务端主采样桥接**：[`CorrectorBridge.kt`](../server/server/core/src/main/java/dev/slimevr/tracking/processor/corrector/CorrectorBridge.kt)
- **修正器运行配置与影子模式控制**：[`CorrectorConfig.kt`](../server/server/core/src/main/java/dev/slimevr/tracking/processor/corrector/CorrectorConfig.kt)
- **跨语言 124D 契约测试套件**：[`CorrectorBridgeContractTest.kt`](../server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/CorrectorBridgeContractTest.kt)
- **真实 ONNX 影子模式推理测试套件**：[`CorrectorBridgeInferenceTest.kt`](../server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/CorrectorBridgeInferenceTest.kt)
- **10.8 万帧连续回放性能基准单测**：[`CorrectorBridgeReplayBenchmarkTest.kt`](../server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/CorrectorBridgeReplayBenchmarkTest.kt)
- **Phase 14 UDP 全协议至 VRServer.tick 10.8 万帧基准单测**：[`UDPProtocolCorrectorBenchmarkTest.kt`](../server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/UDPProtocolCorrectorBenchmarkTest.kt)
- **Phase 14 模型 4 场景与启动冒烟测试套件**：[`ServerStartupSmokeAndFourModelScenariosTest.kt`](../server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/ServerStartupSmokeAndFourModelScenariosTest.kt)
- **SlimeVR Server 独立桌面 Fat Jar**：[`slimevr.jar`](../server/server/desktop/build/libs/slimevr.jar) (114.7 MB, SHA-256: `B50DD7505AA9D38744C95342C88FA45D3D9D9FD102F314E7A4F9A88BF7785EE4`)

---

## 五、 工程安全边界与 Phase 8 实机演进计划

> [!IMPORTANT]
> ### 当前安全红线与状态边界
> 1. **严格只读影子模式 (Strict Shadow Mode)**：
>    当前 SlimeVR 服务端生产配置中 `enabled = false`，`shadowMode = true`。Phase 14 `UDPProtocolCorrectorBenchmarkTest.kt` 在全 108,000 帧（30 分钟）长程回放中逐帧比较 Tracker 姿态，实测 `shadow_mutation_count == 0`。该回放经测试 seam 调用生产 UDP parser/派发及 `tickServer()`，未经过真实 socket 接收线程；保护断言适用于这条回放路径。
> 2. **指标性质区分**：
>    - **工程类指标 (Phase 14 P14.1/P14.3)**：两套定向测试共 8/8 通过；UDP-to-tick Full Tick P95 为 4.733 ms（门限 20ms），P99 7.090 ms、最大 20.436 ms；长程 10.8 万帧 Full GC 后堆差 +0.128 MB（门限 64MB），shadow mutation 为 0；模型场景测试与默认只读安全门禁通过。P14.1 的时间间隔异常和完整故障矩阵未覆盖，整体仍为部分完成；
>    - **算法类指标 (P4 仿真门禁)**：120 条合成会话（216万帧）动态安全门禁全通；ZARU 自适应时间死区底线修正为 0.04°/min，0.14°/min 静坐漂移压制率 96.5%（终点 0.57°，胜率 100%），严格零漂端点偏差 0.008933°（$\le 0.05^\circ$），静坐保持 0.000000°（$\le 0.10^\circ$）；
>    - **工具链限制标记 (P14.2 / P14.3)**：受 Windows 主机缺失 `west`/NCS/`arm-none-eabi-gcc` 及 `node`/`pnpm` 限制，固件与 GUI clean build 保持“受工具链阻塞”的部分完成记录，绝不用已有二进制冒充源码构建；参数量维持 1.54M 冻结。
> 3. **实机验证 (Phase 8) 的核心任务**：
>    - 待实机硬件就绪后，采集真实 IMU 陀螺仪与加速度计原始传感器流；
>    - 接入光学动捕系统建立毫米级位置与真实关节旋转真值；
>    - 依据 Phase 13 P8 硬件到货 8 步 Bring-up 检查清单完成首次硬件点亮，并开展端到端实时解算表现验收。
> 4. **当前推进状态**：Phase 14 **部分完成**。P14.0 与 P14.4 通过；P14.1 的核心回放通过但完整验收项未闭环；P14.2 固件源码构建与 P14.3 GUI/package 构建受工具链阻塞，Server Jar 和服务端场景测试通过。硬件专属实测继续由 Phase 8 待实机在场后执行验收。

