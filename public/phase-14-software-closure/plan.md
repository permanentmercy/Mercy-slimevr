# Phase 14：无硬件软件链路闭环

> **状态**：待执行（承接 Phase 13 未完成项）  
> **规划日期**：2026-10-18  
> **目标**：在没有实体硬件时，补齐 tracker 协议输入到服务端影子推理、固件可重复构建、PC 端构建与启动冒烟的可验收证据。  
> **安全边界**：冻结模型结构和参数量；corrector 保持关闭或 shadow-only；禁止把离线构建、合成回放或协议模拟描述成实机验证。

本阶段只收敛 Phase 13 P3、P6、P7 的剩余软件缺口。Phase 13 已通过的生命周期、124D 契约、合成门禁和基础 tick 回放作为前置证据，不无故重跑。Phase 13 P5 因缺少全新独立数据继续标记数据受限；板卡枚举、无线配对、传感器实测、VD 质量和人体 30 分钟端点继续归 Phase 8。

## 执行前已知条件

- tracker 与 receiver 使用不同的 NCS manifest：`firmware/SlimeVR-Tracker-nRF/west.yml` 指向 `sdk-nrf v3.1-branch`；`firmware/SlimeVR-Tracker-nRF-Receiver/west.yml` 指向 `sdk-nrf v3.3.0-preview3-branch`。必须用隔离 west workspace，分别解析与构建。
- CI 中可复核的代表目标为 tracker `slimenrf_r3/nrf52840/uf2`（UF2）和 receiver `holyiot_21017/nrf52840`（HEX）。它们只作为源码构建目标，不证明用户手头板卡的引脚、bootloader 或可烧录性。
- 服务端 tracker 数据入口是 UDP 协议实现（`UDPProtocolParser`、`TrackersUDPServer`、`UDPPacket`），不是串口文本命令。receiver HID 格式只在找到对应的主机端解码入口后才纳入自动化验收；没有消费方时记录为接口缺口，不虚构 parser。
- `server/` 是独立 Git 仓库且存在本机未提交修改，至少包括 GUI 打包配置、Vite、pnpm workspace 和 Gradle settings。执行者必须先逐仓库记录状态，逐项审阅这些 diff；不得 reset、clean、checkout 覆盖或把本地环境 workaround 未经验证推广。
- `west`、NCS、pnpm、Gradle 或下载依赖若不可用，先记录准确缺失项与版本；需要下载时按根目录约定使用 `http://127.0.0.1:7892` 代理，并将可复现命令写入结果。

## 执行顺序与任务

### P14.0 冻结工作区和工具链

1. 保存根仓库及 `server/`、`driver/`、两个 firmware 子仓库的分支、HEAD、工作区状态；确认待执行任务不会覆盖现存改动。
2. 记录 Java/JDK、Gradle wrapper、Node、pnpm、Python、west 与 NCS toolchain 的实际版本。west/Zephyr 工具链应按两个 manifest 分开，不假设一个 SDK 能构建两套固件。
3. 复核 Phase 13 的 P3 报告和既有构建证据，列出本阶段新增断言与旧证据的边界。

**交付**：结果文档中的环境表、仓库状态摘要、执行命令和任何阻塞原因。缺工具链时先完成不依赖它的任务，并将受阻项标为未执行。

### P14.1 从 tracker UDP 数据包完成服务端长程影子回放

1. 沿生产路径核对 UDP 收包、`UDPProtocolParser` 解码、`TrackersUDPServer` tracker 更新、`VRServer.tickServer()` 和 CorrectorBridge 的调用顺序。测试必须从协议包/生产解码入口喂数据，不能直接对 mock `Tracker` 调 `setRotation()` 代替端到端输入。
2. 若现有网络循环不适合确定性测试，只增加最小的包解码/派发测试 seam；同时至少保留一项 loopback UDP 冒烟，确认实际 socket 到服务端 parser 可用。使用生产 packet serializer 或固定的十六进制黄金包生成 fixture。
3. 跑一条 60 Hz、108,000 帧逻辑时间的动态下肢回放，逐帧断言 shadow 状态不写回、tracker/模型输出为有限数值且维度正确。统计完整 tick 和 Bridge 的 P50/P95/P99/最大耗时，P95 门限沿用 **≤20 ms**。
4. 用短故障用例覆盖四路腿 tracker 的缺失、过期、断开与恢复，HMD/VD 缺失，重复/乱序 packet number、错误包长/未知 packet、非有限姿态、reset、断线重连，以及模型缺失/加载失败/无效输出。服务端 UDP 包没有采样时间戳时，不得宣称验证了远端时间戳；应在实际时钟/采样 seam 注入重复、倒退和大间隔时间，验证差分状态清理。
5. 每个故障用例都检查状态被清理、旁路原因、恢复后的 tracker 状态和有限输出。报告分别统计 `injected_fault_events` 与 `asserted_recoveries`；只有满足明确恢复断言的事件才能计入恢复数。连续异常输出必须旁路，不污染后续有效帧。
6. 回放前预热后测稳定内存基线，报告周期性堆占用、高水位、GC 次数和回放后值；回放后 Full GC 堆差 **≤64 MB**，并检查增长曲线没有持续单调上升。不得仅凭一次 Full GC 前后差值声称无泄漏。

**交付与验收**：可重复的 Gradle 测试、协议黄金 fixture、机器可读回放报告及结果摘要；108,000 帧完成、shadow 改写数为零、输出有效、P95 达标、故障恢复断言通过、内存门限通过。不能从协议层注入数据时，P14.1 保持部分完成。

### P14.2 tracker 与 receiver 固件可重复源码构建

1. 从各仓库 CI workflow、board 定义和 `west.yml` 提取实际参数；分别在隔离 west workspace 执行 pristine build。基准目标使用 tracker `slimenrf_r3/nrf52840/uf2` 和 receiver `holyiot_21017/nrf52840`；如果 NCS manifest 或 workflow 不再支持，先依据当前源码修订 target 并记录依据。
2. 对两个目标各完成一次干净构建；保存 west/NCS/toolchain 版本、manifest revision、完整命令、日志、产物相对路径、格式、大小和 SHA-256。再次构建并比较配置、map/ELF 摘要或二进制哈希；若二进制不可重现，解释时间戳等非确定性来源。
3. 静态核对 UF2/HEX 文件格式和构建 target；只证明文件与源码构建匹配，不宣称 UF2 磁盘枚举、刷写成功、设备启动、IMU ID 正确或 RF 配对成功。
4. 增加固件协议 fixtures：优先覆盖服务端实际消费的 tracker UDP handshake、rotation/rotation+acceleration、设备状态和错误包边界；检查包序列号、tracker 映射、丢包/重连语义。对于 receiver HID pairing/info 报文，仅在确认 server/host 有相应消费代码后加入端到端 parser 测试，并注明覆盖层级。

**交付与验收**：两套固件 clean build 日志与哈希、目标/格式对照表、UDP/HID fixture 测试结果及失败分流。west/NCS 无法配置时保留部分完成，不能用已有发布制品哈希冒充源码 build。

### P14.3 PC 端干净构建、协议集成和启动冒烟

1. 按 `server/` 当前 wrapper 和 lockfile 做 server desktop Jar、GUI build 与 Windows 安装包构建；driver 若 Phase 13 之后源码没有变化，可复用 Phase 13 构建凭据，否则重新构建。执行清理或打包前检查脚本会删除/覆盖的确切目录，避免触碰本机未提交文件。
2. 启动本地 Server 测试实例，通过 P14.1 的 UDP fixture/loopback 数据输入驱动 tracker 状态更新、tick、影子推理和关闭流程；验证 tracker 缺失提示、端口释放和 ONNX Session 释放。
3. 覆盖有效模型、模型缺失、模型损坏/输出无效、HMD/VD 缺失四种启动或运行情况。corrector 必须默认关闭或 shadow-only；没有有效模型时原生 Server 保持可用并安全降级。
4. 检查 GUI/安装包资源是否包含预期 Jar、driver/bindings、默认配置和模型配置说明；在没有 SteamVR/Quest/实体 tracker 时只验程序启动、协议输入和 graceful fallback，不声称完成完整 VR 体验验收。

**交付与验收**：server/GUI/package 构建命令及版本、制品 hash、启动/关闭冒烟报告、模型四场景结果、默认安全配置证据。若安装包必须下载依赖且网络阻断，先按 7892 代理重试并记录结果。

### P14.4 阶段验收与公共文档归档

1. 将 P14.1–P14.3 逐项标记通过、部分完成、未执行或失败；每条状态链接到测试、机器报告或清晰的命令输出摘要。
2. 更新本目录 `results.md`，并同步 [`public/plan.md`](../plan.md)、[`public/README.md`](../README.md)、[`public/results.md`](../results.md)。保留 Phase 13 的原裁定，链接到本阶段补齐的证据。
3. 更新根 README/构建报告仅限发现的事实错误；不要把构建通过写为实机算法精度通过，不要增加模型参数或打开姿态写回。

## 阶段完成门槛

- P14.1 的协议输入、tick、影子推理/安全旁路与故障恢复有可复现自动测试；shadow 姿态改写为零。
- P14.2 两个 manifest 对应的固件目标均有源码 clean build 证据，或明确记为受工具链阻塞；已有 UF2/HEX 不替代构建。
- P14.3 有 server/GUI/package 的构建与启动/关闭结果；所有正确器场景默认只读安全。
- 每项结论都带命令、版本和证据；硬件专属事项仍指向 Phase 8，数据受限仍指向 Phase 6/11 的新独立数据要求。

P14.1、P14.2、P14.3 任一未通过时，阶段结论只能是部分完成。无硬件不会阻止软件阶段结项，但 Phase 8 的实机验收必须继续保持待执行。
