# Phase 13：无硬件软件闭环与可复现交付准备

> **状态**：**部分完成**。P0–P2 与 P4 有通过记录；P3 的动态 tick mock 回放、时延和 shadow 不写回通过，但时间/采样间隔异常、完整 tracker 故障矩阵、reset 后状态与输出有效性断言仍缺；P6 只完成固件配置/制品核验和 Runbook，尚无 tracker/receiver 源码 clean build 与协议 parser fixture；P7 完成 driver、Desktop Jar 和 Corrector 测试，但协议层模拟输入、GUI/安装包本轮构建及启动冒烟仍缺证据。硬件专属验收归 Phase 8。
> **约束**：不增加模型参数或扩大网络；不启用姿态写回；不得把合成/回放结果表述为实机有效性。  
> **规划日期**：2026-10-18

## 当前验收状态（2026-10-18 最新）

| 子任务 | 复核状态 | 证据与剩余工作 |
|---|---|---|
| P0 | 通过 | 工具链与五大仓库差异冻结，基线测试全通。 |
| P1 | 通过 | `contract_v2.py` 与 `CorrectorBridge.writeStream17d` 契约严格对齐（位置在前、旋转在后），124D 单测全通。 |
| P2 | 通过 | `VRServer` 实现 `AutoCloseable`，模型缺失安全旁路，7 项生命周期单测全通。 |
| P3 | 部分完成（tick mock 回放） | 动态 108,000 帧、逐帧 shadow 零写回和性能门限通过；本次复跑 P95 4.638ms、P99 7.810ms、最大 19.659ms，Full GC 后堆差 −0.38MB。仍缺时间戳异常、HMD 与四路腿 tracker 故障矩阵，以及 reset/恢复后的状态和输出有效性断言。 |
| P4 | 通过（合成条件） | 120 条会话（216万合成帧）；测试设定 ±0.14°/min 下静坐漂移压制 96.5%（终点 0.57°，胜率 100%）；严格零漂 endpoint 0.008933°、静坐保持 0.000000°。不能外推为硬件标称漂移或服务端精度。 |
| P5 | 按计划延期 | 严格执行计划 P5.2，无全新独立 source/subject 隔离数据不做确认性评测；不扩大模型容量，标称 1.54M 的精确参数计数待 Phase 15 核对；待硬件到货采集。 |
| P6 | 部分完成 | 已审阅两套 `west.yml`、targets、6 个既有固件制品哈希并整理 Runbook；当前环境无 `west` 命令/构建目录，两个固件源码 clean build 和 UDP/HID 协议 fixture 均未执行。 |
| P7 | 部分完成 | driver 与 `slimevr.jar` 有源码构建记录，Corrector 10 项定向测试通过；本轮没有 GUI/安装包 clean build、真实协议层 tracker 输入或 GUI/服务启动冒烟证据。 |
| P8 | 通过 | 已建立硬件到货 8 步 bring-up 验收清单；全局文档状态同步，硬件专属测试交接 Phase 8。 |

**后续执行顺序**：后续闭环由 [Phase 14](../phase-14-software-closure/plan.md) 承接：1）从 tracker UDP 生产解析/派发入口建立协议回放，并按真实采样时钟语义验证重复、倒退和大间隔；JSON 区分注入事件与已断言恢复；2）配置两个 manifest 对应的隔离 NCS/west 环境，构建 tracker/receiver，并为服务端实际消费的 UDP 包建立 fixture；若发现 receiver HID 主机解码路径，再覆盖相应 HID fixture；3）clean build GUI/安装包，用协议 fixture 驱动 Server 启停、状态更新、缺失设备降级和资源释放；4）同步全局文档。P4 的 ±0.14°/min 只作为合成敏感度档；P5 等新隔离数据，实机项归 Phase 8。

## 目标

在没有实体硬件时，整理并完成软件链路中可独立验收的事项：服务端影子推理生命周期、输入语义、完整回放、算法评测可信度、追踪器与接收器固件可重复构建、PC 端模拟集成，以及新环境复现说明。完成后，硬件到手即可从烧录与串口调试开始，不必先修软件工程缺口。

本阶段承接 Phase 12 R1，但不取代 [Phase 8 实机采集计划](../../hardware-validation/phase-08-device-validation/plan.md)、[Phase 6 D3d](../../model-development/phase-06-accuracy-baseline/plan.md) 或 Phase 9 的硬件验收。

## 执行入口

- **Python 契约/模型/评测**：`ai_corrector/corrector/contract_v2.py`、`contract_v3.py`、`session_simulator_v3.py`、`bias_estimator_v3.py`、`hybrid_composer.py`、`evaluate_phase12_a1_physics_audit.py`、`evaluate_phase12_a2_p6_incremental_value.py`。
- **Kotlin 运行时**：`server/server/core/src/main/java/dev/slimevr/VRServer.kt` 与 `tracking/processor/corrector/{CorrectorBridge,OnnxCorrectorEngine,CorrectorConfig}.kt`；已有契约、推理和回放测试在同级 `src/test/.../corrector/`。
- **固件**：`firmware/SlimeVR-Tracker-nRF/`、`firmware/SlimeVR-Tracker-nRF-Receiver/`，重点看各自 `west.yml`、`.github/workflows/`、`boards/` 与 README；CI target/artifact 设置优先于旧的 `BUILD-REPORT.md` 口述记录。
- **PC 集成/构建**：`server/`、`driver/`、`scripts/msvc-cmake.ps1`、`BUILD-REPORT.md`、根 `.gitignore`。它们是独立仓库或本机环境定制，先看各自 git 状态和已有 diff。

实际构建/测试命令必须从锁定后的 Gradle、west manifest 和 CI workflow 复核，不从历史报告盲目复制版本或 target。

## 当前基线与必须保留的限制

- Phase 12 的 124D Kotlin/Python 契约测试、ONNX 推理测试与特征组装回放已有通过记录；回放没有经过完整 `VRServer.tick()`，服务退出时的 ONNX Session 释放没有接通。
- 在线下肢角速度目前由融合姿态差分推导；它不能称作原始陀螺读数，也不能直接充当 ZARU 原始角速度观测。先统一训练、仿真、Kotlin 特征和评测所使用的信号定义。
- Phase 12 A1 的静态姿势门禁没有实际执行，且模拟角速度语义和服务端输入不一致。旧 A1 通过字段与数字只保留为历史探索记录，不用于验收。
- Phase 12 A2 重用了旧 FinalConfirm；P9-only 与 Raw 完全相同，FK/脚滑是简化代理。旧结果不是独立确认，也没有证明融合增益。
- 根仓库忽略 `server/`、`driver/`、`firmware/` 等独立仓库；`server/` 目前另有未提交的本机构建改动。执行前须分别查看各仓库状态，不能覆盖或混入既有改动。
- nRF52840 的 USB UF2 磁盘模式、SWD 烧录、IMU ID、无线配对和 VD 追踪质量都依赖具体板卡/固件/实机，只能列为到货后的实测项。

## 执行清单

按依赖顺序执行。固件构建准备可与服务端工作并行；算法评测必须等输入语义与门禁修正后再运行。

### P0：冻结仓库与工具链基线

1. 分别记录根仓库、`server/`、`driver/`、`firmware/SlimeVR-Tracker-nRF/`、`firmware/SlimeVR-Tracker-nRF-Receiver/` 的当前分支、HEAD、工作区差异和 remote；保存原有本机改动清单。
2. 固化可复现构建所需版本：JDK/Gradle、Node/pnpm、CMake/MSVC、Python/torch/ONNX Runtime、两套 NCS/west 工作区版本及其 manifest revision。敏感路径只写成相对路径或可配置变量。
3. 建立“干净 checkout 可执行”清单：源代码仓库与大模型/数据/本机二进制分开；明确忽略文件、需要用户自行提供的模型/数据，以及下载步骤。生成物留在已忽略的 build/artifact 目录，只有可复现的源码、配置和小型 fixture 入 Git。

**交付与验收**：仓库/依赖版本清单和干净环境执行记录；任何构建失败都能区分缺依赖、网络、配置或源码问题。不得在未审阅现有 diff 前执行 reset、clean、覆盖式 checkout 或批量删除。

### P1：锁定运行时输入契约与模型 ABI

1. 对 124 个输入维度逐字段记录来源、单位、坐标系、采样时间戳、有效性/age、缺失策略、差分/滤波规则；逐一对照 Python 数据准备、模拟器、契约适配器和 Kotlin `CorrectorBridge`。
2. 明确角速度通道究竟是协议提供的原始陀螺观测，还是由融合姿态作因果差分得到的代理量。若当前只有姿态差分，则全链路统一称为“姿态差分角速度”；禁止用它声称检测了硬件陀螺零偏。若协议层存在原始陀螺字段，再单独接入并增加黄金 fixture；未经训练/数据生成对齐，不替换现有模型输入。
3. 核验相邻帧时间间隔、四元数符号连续性、tracker 重连/reset、空间重定位及长时间断流时的差分和状态清理；无效或异常时间戳不得生成有效角速度。
4. 固定部署模型 ABI：文件名/路径配置、SHA-256、输入输出名与维度、递归状态尺寸、Opset、模型导出命令和版本。缺模型、hash 不符、shape 错误、NaN/Inf 均须清晰记录并安全旁路。

**交付与验收**：更新后的字段来源表、124D 哨兵跨语言逐元素比较、模型 manifest/hash、差分角速度边界测试。契约含糊或训练/服务端定义不一致时停止后续精度结论。

### P2：补齐 ONNX 生命周期与影子模式运行安全

1. 在服务初始化时加载一个 ONNX Session，在 `VRServer`/服务关闭路径显式关闭；异常启动、部分初始化失败、重复关闭都要安全。
2. 使用相对项目路径或用户配置路径查找模型，不依赖某一台机器的绝对路径；配置默认保持禁用/影子模式，缺少模型不阻断原生 SlimeVR 服务。
3. 覆盖正常输入、模型缺失、加载失败、推理异常、非有限数值、过期 tracker、重复 reset、空间切换、Session close 的 Kotlin 测试。
4. 通过快照/逐帧比较证明 shadow-only 不改 Tracker、骨架求解结果或 Avatar 输出；故障日志须包含旁路原因，但不得每帧刷屏。

**交付与验收**：生命周期代码与测试；模型有效/无效两种配置下均能启动/退出；姿态写回差异严格为零。继续禁止调用姿态应用路径。

### P3：建立真实 `VRServer.tick()` 的无硬件连续回放

1. 用测试替身或现有 replay 接口注入有时间戳的虚拟 tracker 流，使样本走过 tracker 更新、服务端 tick、Bridge 采样、ONNX 推理和影子记录；不能只直接调用特征组装函数作为端到端验收。
2. 覆盖 60 Hz 连续帧、4 路腿 tracker 缺失/过期、HMD/VD 缺失、重复/跳变时间戳、快速与完整 reset、断线重连、模型加载失败和连续异常输出。
3. 回放至少 108,000 帧逻辑时间；逐次核对状态重置、异常旁路、输出有效性和影子姿态恒等。报告 Bridge 和完整服务 tick 的 P50/P95/P99/最大耗时。
4. 运行前先测量无模型/预热后的稳定内存基线，再写明并自动执行堆/常驻状态增长门限；只报告强制 GC 前后数字不算内存验收。门限需在查看新回放结果前冻结。

**交付与验收**：可重跑的 Gradle 测试和机器可读 JSON 报告；完整 `VRServer.tick()` 处理延迟 P95 ≤20 ms；无状态泄漏、无 NaN/Inf、无跨 reset 污染、shadow 姿态 0 改动。若测试构造无法覆盖真实 tick，明确标记未完成，禁止称为端到端。

### P4：修复 Phase 9 物理仿真与安全门禁（Phase 12 A1 重做）

1. 将世界坐标逆偏航修正按矩阵乘法顺序独立验证：`R_raw = R_drift_world · R_clean · R_mount`；分别测试四路正负漂移、安装旋转和非中立姿势，包含不使用网络的解析 Oracle 单测。
2. 将数据生成拆分为可读阶段：干净动作/安装变换、陀螺偏置与噪声、融合姿态生成、在线可见特征。每种信号注明仿真假设；若模拟直接生成带偏置的角速度，它只验证该陀螺估计器模型，不证明板载 AHRS 或当前服务端姿态差分输入下的效果。
3. 删除所有硬编码 PASS。每个静态/零漂门禁必须从记录的逐帧与逐 tracker 数据计算并输出；严格零漂关闭随机游走，并加入盘腿/坐姿/转身保持等静态姿势，以及持续运动、无静止窗、VD 缺失场景。
4. 用 Raw、ZARU-only、步态辅助-only、Full 做配对会话比较；覆盖正负常偏置、零偏置、随机游走与缓慢变化。主指标采用 30 分钟会话起点至终点四路累计偏航误差增长总量，不以全帧平均代替。每个主要工况至少 30 个独立会话/随机种子，报告均值、中位数、胜率、95% CI、逐路结果与最差案例。
5. 在新确认集运行前冻结算法、噪声范围与门限。严格零漂 endpoint 四路合计误改 ≤0.05°，静态保持姿势增量 ≤0.10°；两项机器判定缺数据、未执行或失败均不得 PASS。

**交付与验收**：新命名脚本/JSON、仿真假设说明、解析坐标单测和审计结果。所有结论只称“该仿真模型范围内”；只有原始传感器与板载融合链路实测才能进入硬件有效性结论。此项不扩大模型参数。

### P5：重新建立 Phase 11 增量确认（Phase 12 A2 重做，可条件跳过）

1. 审核 AMASS/现有序列身份、主体/动作重复、此前 Train/Val/Test、Phase 11 和旧 FinalConfirm 的使用历史。只从此前从未用于训练、选模、阈值调整或报告确认的 source/subject 生成新 manifest，记录源 ID、原文件哈希、生成种子、漂移配对 ID 和 split。
2. 若找不到足够新的独立源序列，不重用旧 FinalConfirm，不做“确认性”运行；记录数据限制与需要新增的数据，而不是降低隔离标准。
3. 冻结当前网络与参数量，不新增容量；同条件比较 Raw、P9-only、P6-only、P9+P6，证明 P9 产生非零因果修正后再谈融合收益。评价世界偏航 endpoint、Swing/全姿态安全和逐序列差值；报告各路、最差会话与 CI。
4. 重做几何评价：先用已知人工旋转单测骨架 FK，再使用一致骨架层级/骨长与世界 root 运动计算关节误差、脚在世界坐标中的支撑滑移。缺独立真值或可信映射时将该指标标为未验，不以固定骨长代理命名 SMPL-H/世界足滑。
5. 报告 VD 质量/年龄/丢帧/噪声/安装差异的敏感性范围；无实测统计时将其称为压力测试范围，不伪称真实分布。

**交付与验收**：新隔离 manifest、可重建数据配方、单次确认报告及失败样本清单。无新数据时本项状态是“数据受限/未执行”，不是通过。此项不扩大模型参数。

### P6：追踪器与接收器固件的无硬件可重复构建

1. 分别审阅两个固件仓库的 `west.yml`、CI workflow、板卡定义和 release 文件格式；以仓库实际配置为准，固定各自 NCS/west 版本和 revision，继续使用独立工作区。
2. 固定试做板卡—target 对照表（SuperMini、nice!nano、ProMicro、HolyIOT 接收器），核对 pin、I2C/SPI 变体、IMU INT、USB 和无线配置；不把板卡同名/兼容 target 当成引脚已实测。
3. 在可重建环境中分别构建 tracker 和 receiver 的实际目标；保存命令、版本、日志摘要、产物格式、板卡 target、大小和 SHA-256。重复构建至少验证固件行为/配置一致；若二进制不可复现，记录非确定性来源。
4. 对 `.uf2` 只做离线格式/target/文件名校验，并核对 bootloader/flash layout 是否匹配；对 HEX/其他格式记录对应 SWD 烧录路径。文档明确 UF2 bootloader 常提供 USB Mass Storage，但具体 SuperMini 必须到货后确认双击 Reset 是否枚举磁盘。
5. 从当前实际接口建立包 fixture：优先测试 tracker UDP parser/派发链路及固件 `info`、tracker 状态和包序号等字段；receiver HID 报文仅在核实存在对应 host decoder 后覆盖。不要把 tracker UDP/HID 报文称为串口文本命令，也不声称完成真实 RF 配对或芯片 WHO_AM_I 验证。

**交付与验收**：两个固件 clean build 记录、target/接口矩阵、烧录 Runbook、产物 hash 与适用的 UDP/HID fixture 测试。成功编译不等于设备能启动、传感器可识别或无线可配对。

### P7：PC 软件的干净构建与模拟数据集成

1. 对 Server、SteamVR driver、GUI 和安装包按锁定工具链建立可重复的构建顺序；显式保留当前 `server/` 中已有的本机 MSVC/pnpm/Gradle 修改，分别审阅，不把环境 workaround 未经测试地推广成项目要求。
2. 使用协议层虚拟 tracker/回放 fixture 运行 Server 无硬件冒烟：启动、接收模拟数据、状态更新、缺失设备提示、退出资源释放；检查 GUI 配置能保持 corrector 默认关闭/影子模式。
3. 测试模型正常、模型缺失、模型无效、上身参考缺失四种启动/运行场景。安装包应明确模型是随包提供还是用户指定；当前被 ignore 的 ONNX 不能被假定在干净 clone 中存在。
4. 验证 driver/GUI/server 能安装构建并启动到可用状态；无 SteamVR/Tracker 时只验启动和 graceful fallback，不伪装完整 VR 应用验收。

**交付与验收**：一个可重复的本机/CI构建入口、模拟数据端到端记录、默认安全配置检查和新环境操作说明。无实体硬件的部分与依赖 SteamVR/Quest 的部分清楚分界。

### P8：同步总文档并准备硬件到货验收包

1. 每项 P0–P7 完成后分别更新本计划的状态、阶段 `results.md`、[`public/plan.md`](../../plan.md)、[`public/README.md`](../../README.md) 和 [`public/results.md`](../../results.md)。结果记录命令、软件版本、日志/JSON 链接、提交 SHA、限制及未通过事项。
2. 对照根 `README.md` 与 [`BUILD-REPORT.md`](../../../BUILD-REPORT.md)，修正过期架构、指标、版本和产物格式；全局只保留被阶段结果支持的结论。
3. 准备到货检查单：板卡丝印/variant、首次 SWD bootloader、UF2 USB 磁盘模式、固件刷写、串口 IMU ID/陀螺读数/时间戳、接收器配对、服务器设备映射、VD 胸/肘观测映射与失效年龄、reset/丢包和 30 分钟会话采集。每项附操作、期望日志、失败分流和证据位置。

**交付与验收**：新 checkout 执行指南与一页硬件 Bring-up 表；无实际板卡记录的字段保持“待实测”。

## 硬件依赖、暂不执行的验收

以下事项不纳入本阶段软件 PASS：实体 USB 枚举/双击 Reset、DAPLink SWD 接线与恢复、芯片 WHO_AM_I 和原始陀螺/加速度读数、实际电源/电池续航、无线配对与丢包、Quest/VD 胸肘质量与时间同步、人体动作下姿态效果、至少 30 分钟无中途 reset 的误差端点和独立光学真值。硬件到货后按 Phase 8 启动，先影子采集和测量，再评估是否开放在线修改。

## 总体完成条件

- P0–P3、P6–P8 的软件交付均有干净环境复现记录和自动测试；内存与完整 tick 性能有预先冻结的门限。
- P4 仅在所有安全门禁实际计算且输入定义对齐时通过；否则公开保留阻塞和原因。
- P5 仅在全新 source/subject 隔离数据存在时进行确认；没有新数据则明确记录未执行。
- Tracker、Skeleton、Avatar 影子差异为零，corrector 默认不在线写入。
- 没有硬件的限制在总文档和每份报告中一致；不扩大网络参数量。

## 给执行者的顺序

1. 执行 P0，先保护现有嵌套仓库改动并锁定环境。
2. 顺序执行 P1 → P2 → P3；输入 ABI 或 Session 生命周期失败时不继续端到端/算法宣称。
3. 执行 P4；修复观测语义后才能尝试 P5。没有全新确认集就跳过 P5 的计算，只记录数据缺口。
4. 固件 P6 可与 P1–P4 并行，但不得把构建通过写成硬件验证通过。
5. 完成 P7 集成和 P8 文档/bring-up；仅当对应项目的结果文档更新后才标记子任务完成。
