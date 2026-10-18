# Phase 14：无硬件软件链路闭环执行结果

> **状态**：部分完成（软件全链路回放与测试通过，固件与GUI构建受工具链阻塞）  
> **执行日期**：2026-10-18  
> **本文件用途**：按 [Phase 14 计划](plan.md) 逐项记录实际命令、环境、机器证据和验收结果。

## 执行摘要

| 子任务 | 状态 | 结果与证据 |
|---|---|---|
| P14.0 工作区与工具链冻结 | 通过 | 5 个仓库状态全部审阅；`server/` 4 个本地沙箱文件完整保留；工具版本全部登记。 |
| P14.1 tracker UDP → Server tick 影子回放 | 通过 | 108,000 帧全动态回放通过；948 次故障注入/125 次恢复；0 次 shadow 改写；P95 耗时 1.106 ms；堆差 +0.078 MB。 |
| P14.2 tracker/receiver 固件源码构建 | 部分完成 | 明确登记受 Windows host 缺失 `west`、`nRF Connect SDK (NCS)`、`arm-none-eabi-gcc` 工具链阻塞；协议 fixture 在 P14.1 完整覆盖。 |
| P14.3 PC 构建与协议集成冒烟 | 部分完成 | Server Desktop Jar 构建成功 (`slimevr.jar`, 114.7 MB)；driver 保持 Phase 13 MSVC 验证凭据；模型 4 场景与默认安全门禁 100% 通过；GUI/package 受 `node`/`pnpm` 缺失阻塞。 |
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
| Node / pnpm | 缺失 | — | Windows PATH 未安装 global `node`/`pnpm` |
| west / Zephyr NCS | 缺失 | — | Windows host 未安装 `west`、`zephyr-sdk` 或 `arm-none-eabi-gcc` |

## P14.1 协议输入与影子回放

- **执行命令**：`.\gradlew.bat :server:core:test --tests "dev.slimevr.tracking.processor.corrector.UDPProtocolCorrectorBenchmarkTest"`
- **测试结果**：3 tests completed, **0 failed** (耗时 1m 40s)
  - `test01_udpLoopbackSocketSmoke`: 本地 127.0.0.1 UDP Socket 回环冒烟通过，验证后台网络线程稳定接收。
  - `test02_udpGoldenPacketHexFixturesAndFaults`: 黄金 Packet 17 解析、乱序包 `IOException` 拦截、未知包 ID 9999 安全跳过、NaN 姿态归一化兜底全部通过。
  - `test03_fullUDPProtocolToVRServerTick108kFramesReplay`: 108,000 帧长程全链路回放与故障注入矩阵通过。
- **Fixture 覆盖包型**：
  - `UDPPacket3Handshake` (协议版本 22，MAC 地址与硬件识别)
  - `UDPPacket1Rotation` (28 bytes 原始四元数)
  - `UDPPacket17RotationData` (31 bytes 传感器旋转数据流，含 `DATA_TYPE_NORMAL`)
  - 生产坐标转换：`raw_rotation = AXES_OFFSET.inv() * world_rotation`，由 `TrackersUDPServer.processPacket` 执行 `AXES_OFFSET * raw_rotation` 还原。
- **108,000 帧回放指标**：
  - 回放逻辑时长：30.0 分钟 (60 Hz 连续步态正弦摇摆姿态流)
  - 故障注入事件数 (`injected_fault_events`)：**948 次**
    - 60 次快速校准状态重置 (每 30 秒一次)
    - 30 次 VD 胸部断连 (每 60 秒连续 20 帧断开，共 600 帧)
    - 15 次小腿 IMU 超时掉线 (每 120 秒连续 15 帧掉线，共 225 帧)
    - 21 次 UDP 乱序报文注入 (`packetNumber = 1L` 过期重放)
  - 明确断言恢复数 (`asserted_recoveries`)：**125 次** (状态复位、重连自愈 `TrackerStatus.OK`、乱序丢弃异常捕获均通过断言验证)
  - 影子篡改计数 (`shadow_mutation_count`)：**0** (严格零篡改，追踪器读数与计算姿态完全不受影子推理影响)
  - 非有限输出数 (`non_finite_output_count`)：**0** (所有残差四元数与门控输出均为有效有限数值)
  - 服务端 Full Tick 耗时 (3,000 帧样本)：
    - Mean: **0.746 ms**, P50: **0.692 ms**, **P95: 1.106 ms** (门限 $\le 20.0$ ms, 达标率 100%), P99: **1.538 ms**, Max: **6.566 ms**
  - CorrectorBridge 推理耗时：
    - P50: **0.642 ms**, P95: **0.988 ms**, P99: **1.396 ms**
  - 内存基线与平稳性：
    - Baseline: 9.36 MB, Watermark: 30.62 MB, 回放后 Full GC 堆差: **+0.078 MB** (门限 $\le 64.0$ MB)
    - 周期性检查点：[30.62, 26.65, 23.24, 19.24, 15.68] MB (无持续单调上升趋势)
- **机器可读报告**：[`ai_corrector/artifacts/phase14_p1_udp_server_tick_benchmark_report.json`](file:///h:/program/slimevr/ai_corrector/artifacts/phase14_p1_udp_server_tick_benchmark_report.json)

## P14.2 固件构建与协议夹具

| 项目 | Board target | NCS / west | 构建状态 | 格式、大小、SHA-256 / 现状依据 |
|---|---|---|---|---|
| Tracker | `slimenrf_r3/nrf52840/uf2` | NCS `v3.1-branch` | 受工具链阻塞 | Windows 本机未安装 `west`、`zephyr-sdk` 及 `arm-none-eabi-gcc` 交叉编译环境。按计划规则，严禁使用旧制品哈希冒充源码 clean build，如实记为工具链阻塞。 |
| Receiver | `holyiot_21017/nrf52840` | NCS `v3.3.0-preview3-branch` | 受工具链阻塞 | 同上，属于交叉编译环境依赖缺失。 |

- **协议夹具覆盖**：
  - Tracker UDP Handshake、Rotation (Packet 1)、RotationData (Packet 17) 已在 P14.1 测试套件中作为黄金 fixture 验证，完整覆盖服务端解包、序列号校验与重连状态机。
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
1. **P14.1 全面达标**：Tracker UDP 协议解码至 VRServer tick 全链路 108,000 帧长程影子回放通过，故障注入 948 次、自愈 125 次，**影子篡改数为 0**，P95 耗时 1.106 ms（门限 $\le 20$ ms），堆差 +0.078 MB（门限 $\le 64$ MB），生成机器可读报告。
2. **P14.3 核心服务达标**：Server 桌面独立 Fat Jar 构建成功；模型 4 类场景容错、默认只读安全门禁、端口与 ONNX Session 释放自动化测试 100% 通过（5/5）。
3. **工具链受限阻塞**：
   - 固件源码 Clean Build（P14.2）受 Windows 主机缺失 `west`/NCS/`arm-none-eabi-gcc` 阻塞；
   - GUI 与安装包打包（P14.3）受缺失 `node`/`pnpm` 阻塞。
   - 依据计划准则：“P14.1、P14.2、P14.3 任一未通过时，阶段结论只能是部分完成。不能用已有发布制品哈希冒充源码 build”。
4. **与 Phase 8 实机工作的边界交接**：
   - 本阶段已严格完成所有可离线仿真和软件集成的闭环验证；
   - 物理硬件专属事项（真实 IMU 刷写、USB 枚举、RF 配对、实机空间对齐与全身动捕跟踪精度）继续由 **Phase 8** 待硬件在场后执行验收。
