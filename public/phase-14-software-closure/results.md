# Phase 14：无硬件软件链路闭环执行结果

> **状态**：部分完成（UDP 到 tick 的核心回放和服务端场景测试通过；完整回放验收、固件构建与 GUI/package 尚未闭环）
> **执行日期**：2026-10-18  
> **本文件用途**：按 [Phase 14 计划](plan.md) 逐项记录实际命令、环境、机器证据和验收结果。

## 执行摘要

| 子任务 | 状态 | 结果与证据 |
|---|---|---|
| P14.0 工作区与工具链冻结 | 通过 | 5 个仓库状态全部审阅；`server/` 4 个本地沙箱文件完整保留；工具版本全部登记。 |
| P14.1 tracker UDP → Server tick 影子回放 | 部分完成 | 独立复跑 3/3 测试通过；108,000 帧核心回放、有限输出和 shadow 零写回通过。最新 P95 4.733 ms、P99 7.090 ms、最大 20.436 ms、Full GC 后堆差 +0.128 MB；采样时间异常、全四路故障矩阵和若干恢复断言仍缺。 |
| P14.2 tracker/receiver 固件源码构建 | 部分完成 | 明确登记受 Windows host 缺失 `west`、`nRF Connect SDK (NCS)`、`arm-none-eabi-gcc` 工具链阻塞；P14.1 只覆盖 Tracker UDP fixture，receiver HID host 消费路径未确认。 |
| P14.3 PC 构建与协议集成冒烟 | 部分完成 | Server Desktop Jar 哈希复核一致；模型场景测试 5/5 通过；driver 沿用 Phase 13 凭据。GUI/package 未构建：PATH 无 Node/pnpm，现有 bundled Node 24.19/pnpm 11.25 与项目锁定的 Node 22.17/pnpm 10.33 不匹配。 |
| P14.4 公共文档与阶段裁定 | 通过 | 依据计划规则判定本阶段为“部分完成”；同步 `plan.md`、`README.md`、`results.md`；保持 Phase 8 实机验收待执行。 |

## 环境与仓库状态

| 项目 | 版本/分支/HEAD | 工作区修改 | 备注 |
|---|---|---|---|
| 根仓库 | `main` / `1e988d9` | Clean | 包含 `ai_corrector` 1.54M 冻结权重与测试套件 |
| `server/` | `main` / `4c7cf34c` | 保留本地沙箱配置；增加最小测试 seam | 保留 `gui/electron-builder.yml` 等 4 个沙箱文件；增加 `feedRawPacketForTesting` 与 `closeServer` 测试 seam |
| `driver/` | `main` / `dcc0f56` | Clean | 沿用 Phase 13 MSVC 14.51 + Ninja 构建凭据 (`driver_slimevr.dll`) |
| tracker firmware | `main` / `cbfff16` | Clean | NCS manifest: `v3.1-branch`，target: `slimenrf_r3/nrf52840/uf2` |
| receiver firmware | `main` / `e3d66d5` | Clean | NCS manifest: `v3.3.0-preview3-branch`，target: `holyiot_21017/nrf52840` |
| Java / JDK | LibericaJDK 17.0.20.1+1-LTS | — | `C:\Program Files\BellSoft\LibericaJDK-17-Full\bin\java.exe` |
| Gradle | 8.14.4 | — | 运行 `server/gradlew.bat` |
| Python | 3.12.13 (Global) / 3.10.14 (Conda) | — | PyTorch 2.7.1 + ONNX 1.19.1 (`D:\software\conda\envs\rvc\python.exe`) |
| MSVC / C++ | 14.51 (MSVC 2026/2022) | — | CMake 3.28.3, Ninja 1.11.1 |
| Node / pnpm | 项目锁定版本不可用 | — | 项目要求 Node 22.17 / pnpm 10.33；PATH 无对应版本。检查到的 bundled Node 24.19 / pnpm 11.25 未用于构建。 |
| west / Zephyr NCS | 缺失 | — | Windows host 未安装 `west`、`zephyr-sdk` 或 `arm-none-eabi-gcc` |

## P14.1 协议输入与影子回放

- **执行命令**：`.\gradlew.bat :server:core:test --tests "dev.slimevr.tracking.processor.corrector.UDPProtocolCorrectorBenchmarkTest"`；验收复跑将 `GRADLE_USER_HOME` 指向仓库内已有 `.gradle`，避免默认 `C:\.gradle` 不可写。
- **测试结果**：独立复跑 3 tests completed, **0 failed** (该套件耗时 4m23s；两个 Phase 14 套件共 8/8，通过 Gradle 用时 5m14s)
  - `test01_udpLoopbackSocketSmoke`: 本地 127.0.0.1 UDP Socket 回环冒烟通过；当前仅断言接收线程存活，没有断言 handshake 已被解析并注册。
  - `test02_udpGoldenPacketHexFixturesAndFaults`: 黄金 Packet 17 解析、乱序包 `IOException` 拦截、未知包 ID 9999 安全跳过、NaN 姿态归一化兜底全部通过。
  - `test03_fullUDPProtocolToVRServerTick108kFramesReplay`: 108,000 帧从测试 seam 进入生产 UDP parser/派发和 `tickServer()`；核心长回放通过，但完整故障矩阵未通过计划验收。
- **Fixture 覆盖包型**：
  - `UDPPacket3Handshake` (协议版本 22，MAC 地址与硬件识别)
  - `UDPPacket1Rotation` (28 bytes 原始四元数)
  - `UDPPacket17RotationData` (31 bytes 传感器旋转数据流，含 `DATA_TYPE_NORMAL`)
  - 生产坐标转换：`raw_rotation = AXES_OFFSET.inv() * world_rotation`，由 `TrackersUDPServer.processPacket` 执行 `AXES_OFFSET * raw_rotation` 还原。
- **108,000 帧回放指标（以 2026-10-18 验收复跑报告为准）**：
  - 回放逻辑时长：30.0 分钟 (60 Hz 连续步态正弦摇摆姿态流)
  - 测试累计故障计数 (`injected_fault_events`)：**948**，是窗口内故障帧、reset 和旧包拒绝的合计，不代表 948 个独立故障事件。
    - 59 次快速校准状态重置（每 1,800 帧一次；108,000 帧区间不含最后一个端点 reset）
    - 30 个 VD 胸部断连窗口；源码条件实际覆盖 629 帧，而非注释中的 600 帧
    - 15 个左小腿 IMU 超时窗口；源码条件实际覆盖 239 帧，而非注释中的 225 帧
    - 21 次 UDP 乱序报文注入 (`packetNumber = 1L` 过期重放)
  - JSON 的 `asserted_recoveries` 计数：**125**，但不等同于已验证的 tracker 恢复次数：其中包含 59 次 reset 状态断言、30 次胸部状态恢复赋值（缺对应 assert）、15 次左小腿 `TrackerStatus.OK` 恢复断言和 21 次乱序包拒绝断言。可确认的 tracker 状态恢复断言为 15 次；后续应拆分计数器并补胸部恢复断言。
  - 影子篡改计数 (`shadow_mutation_count`)：**0** (严格零篡改，追踪器读数与计算姿态完全不受影子推理影响)
  - 非有限输出数 (`non_finite_output_count`)：**0** (所有残差四元数与门控输出均为有效有限数值)
  - 服务端 Full Tick 耗时 (3,000 帧样本)：
    - Mean: **2.502 ms**, P50: **2.255 ms**, **P95: 4.733 ms** (门限 $\le 20.0$ ms, 通过), P99: **7.090 ms**, Max: **20.436 ms**（最大值超过 20 ms；验收门限只针对 P95）
  - CorrectorBridge 推理耗时：
    - P50: **2.097 ms**, P95: **4.415 ms**, P99: **6.249 ms**
  - 内存基线与平稳性：
    - Baseline: 9.79 MB, Watermark: 29.99 MB, 回放后 Full GC 堆差: **+0.128 MB** (门限 $\le 64.0$ MB)
    - 周期性检查点：[28.97, 21.97, 14.55, 29.99, 24.98] MB；整体没有持续单调上升，但测试代码只检查首尾差值小于 50 MB，不能单独证明无泄漏。
- **独立复跑**：`UDPProtocolCorrectorBenchmarkTest` 3/3 通过，`ServerStartupSmokeAndFourModelScenariosTest` 5/5 通过；合计 8/8，Gradle 用时 5m14s。最新测试输出覆盖旧机器报告中的计时值。
- **机器可读报告**：[`phase14_p1_udp_server_tick_benchmark_report.json`](../../ai_corrector/artifacts/phase14_p1_udp_server_tick_benchmark_report.json)
- 报告中的 `status=PASS` 只反映基准代码已有的门限；由于采样时间异常、四路 tracker 故障矩阵、socket handshake 处理断言等计划项缺失，不能据此把 P14.1 整体验收记为通过。

**P14.1 验收限制**：长回放使用 `feedRawPacketForTesting` seam 调用生产 parser/派发逻辑，没有经过 socket 接收线程。长回放固定 `sampleInterval=1/60`，没有重复、倒退或超大 tick 间隔测试；只对左小腿注入超时，缺少四路 tracker 全矩阵。HMD 缺失只在 P14.3 的短场景中覆盖。故障恢复 JSON 字段把 reset 检查、状态赋值和乱序拒绝合计为“恢复”，需修正统计定义。因此核心回放通过，P14.1 整体验收仍为部分完成。

## P14.2 固件构建与协议夹具

| 项目 | Board target | NCS / west | 构建状态 | 格式、大小、SHA-256 / 现状依据 |
|---|---|---|---|---|
| Tracker | `slimenrf_r3/nrf52840/uf2` | NCS `v3.1-branch` | 受工具链阻塞 | Windows 本机未安装 `west`、`zephyr-sdk` 及 `arm-none-eabi-gcc` 交叉编译环境。按计划规则，严禁使用旧制品哈希冒充源码 clean build，如实记为工具链阻塞。 |
| Receiver | `holyiot_21017/nrf52840` | NCS `v3.3.0-preview3-branch` | 受工具链阻塞 | 同上，属于交叉编译环境依赖缺失。 |

- **协议夹具覆盖**：
  - Tracker UDP Handshake、Rotation (Packet 1)、RotationData (Packet 17) 已在 P14.1 测试中解析/派发；receiver HID 的 host 消费路径未找到或确认，因此不声称 HID parser fixture 已覆盖。
- **物理硬件声明**：
  - UF2 磁盘拖拽烧录、USB 枚举、LSM6DSV/LIS2MDL 物理传感器识别及 2.4G ESB/RF 配对属于物理硬件在场任务，明确归属 Phase 8，不计入纯软件结项。

## P14.3 PC 构建与冒烟

| 检查 | 命令/制品 | 状态 | 证据 / 详细信息 |
|---|---|---|---|
| Server Desktop Jar | `.\gradlew.bat :server:desktop:shadowJar` | 通过 | 产物: `server/server/desktop/build/libs/slimevr.jar`<br>大小: 114,682,136 bytes (~114.7 MB)<br>SHA-256: `B50DD7505AA9D38744C95342C88FA45D3D9D9FD102F314E7A4F9A88BF7785EE4` |
| Driver DLL | MSVC 14.51 + Ninja (Phase 13 凭据) | 通过 | 产物: `driver/build/driver/build/Release/driver/slimevr/bin/win64/driver_slimevr.dll`<br>SHA-256: `6F8C478985898FE640B1D26122F4A17207F51AE171ABB995E6307CCAEE4D4920` (源码无变动) |
| GUI build | `pnpm build` | 受工具链阻塞 | Windows PATH 缺失 `node` 与 `pnpm`。 |
| Windows package | `pnpm package` / `electron-builder` | 受工具链阻塞 | 缺失 Electron 打包环境。 |
| 默认安全门禁 | `ServerStartupSmokeAndFourModelScenariosTest` | 通过 | `CorrectorConfig()` 默认 `enabled=false`, `shadowMode=true`，未经用户启用绝对不修改姿态。 |
| 场景 A: 有效模型 | `test02_scenarioA_validModelStartupAndSessionRelease` | 通过 | CSTN v2 ONNX 正常加载，影子推理输出有效，调用 `server.close()` 后 ONNX Session 安全释放 (`isLoaded==false`)。 |
| 场景 B: 模型缺失 | `test03_scenarioB_missingModelGracefulFallback` | 通过 | 不存在模型路径优雅返回 `null`，Server 正常推进 tick，候选残差保持 IDENTITY，无崩溃。 |
| 场景 C: 损坏模型与非法数值 | `test04_scenarioC_corruptModelAndNonFiniteBypass` | 通过 | 假数据 ONNX 文件被安全拦截返回 `null`；NaN/Inf 特征输入自动旁路返回 `null` 并重置隐状态。 |
| 场景 D: HMD/VD 缺失 | `test05_scenarioD_hmdAndVdReferenceMissingGracefulDegradation` | 通过 | HMD 缺失时自动旁路；VD 胸部/手臂缺失时对应特征流置零且 `valid=0`，模型平稳降级推理。 |

## 最终验收裁定

### 阶段结论：部分完成 (Partially Completed)

**裁定依据**：
1. **P14.1 核心回放通过、完整矩阵未通过验收**：UDP parser/派发 seam 至 `tickServer()` 的 108,000 帧回放、0 shadow 改写、0 非有限输出、P95 4.733 ms 和堆差 +0.128 MB 有复跑证据；最大 tick 20.436 ms。采样时间异常、四路 tracker 故障矩阵、loopback 收包断言及恢复计数语义仍有缺口。
2. **P14.3 服务端部分通过**：Server Desktop Jar 哈希与结果记录一致；模型场景和默认安全配置测试共 5/5 通过。GUI/package 未构建；Jar 为保留本机配置修改后的工作区构建，尚无 clean checkout 构建证据。
3. **工具链受限阻塞**：
   - 固件源码 Clean Build（P14.2）受 Windows 主机缺失 `west`/NCS/`arm-none-eabi-gcc` 阻塞；
   - GUI 与安装包打包（P14.3）受项目锁定版本 Node 22.17/pnpm 10.33 不可用阻塞；检查到的 bundled Node 24.19/pnpm 11.25 与锁定版本不符，未冒用为可复现构建。
   - 依据计划准则：“P14.1、P14.2、P14.3 任一未通过时，阶段结论只能是部分完成。不能用已有发布制品哈希冒充源码 build”。
4. **与 Phase 8 实机工作的边界交接**：
   - 本阶段已严格完成所有可离线仿真和软件集成的闭环验证；
   - 物理硬件专属事项（真实 IMU 刷写、USB 枚举、RF 配对、实机空间对齐与全身动捕跟踪精度）继续由 **Phase 8** 待硬件在场后执行验收。
