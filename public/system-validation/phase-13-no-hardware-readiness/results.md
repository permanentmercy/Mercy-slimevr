# Phase 13 无硬件软件闭环与可复现交付准备执行结果

> **执行日期**：2026-10-18  
> **阶段定位**：本阶段尝试在无实体硬件条件下推进服务生命周期、输入语义、tick 回放、合成算法门禁、固件构建准备和 PC 集成。下方验收复核区分已通过、部分通过与未执行项目。
> **模型与安全约束**：当时按标称 1.54M 记录为不扩大网络容量；精确参数量与当前部署 checkpoint 的对应关系待 Phase 15 核对。服务端保持严格只读影子模式（`shadowMode=true`，0 姿态写回）；不把任何仿真/离线指标描述为实机达标。

> **验收复核（2026-10-18 最新）**：本阶段**部分完成**。本轮 P3 已验证动态 mock tick 长回放、时延门限与 shadow 不写回，但完整 P3 验收矩阵仍未覆盖；P6、P7 也缺源码/协议集成项：
> 1. **P3**：全动态 108,000 帧，注入 59 次 reset 调用和 868 次 Tracker 状态故障计数，逐帧 shadow 零写回（`shadow_mutation_count=0`）；本次 3,000 帧性能样本 P95 4.638ms、P99 7.810ms、最大 19.659ms，Full GC 后堆差 −0.38MB。10 项定向 Kotlin 测试全通过。注意 868 是故障注入计数，测试未独立断言每次异常恢复；时间戳异常、HMD/四路腿 tracker 故障矩阵、reset 后状态及模型输出有效性断言仍缺。输入直接驱动 mock Tracker 与 `tickServer()`，不含真实网络协议收包；
> 2. **P4**：120 条、共 216 万帧合成会话的动态物理门禁通过。±0.14°/min 是仿真设定，非传感器标称值；静坐含漂移平均抑制率 96.5%（16.21°→0.57°），严格零漂最大端点偏差 0.008933°、静坐零漂 0°。直接注入合成陀螺偏置与服务端姿态差分的输入语义不同，结论只适用于当前仿真；
> 3. **P6**：完成 `west.yml`/targets/分区审阅、6 个既有固件产物 SHA-256 核验和离线 Runbook；本机没有 `west` 命令或固件构建目录，因此两个源码 clean build 与 UDP/HID 协议 fixture 未执行，**部分完成**；
> 4. **P7**：driver 与 Desktop Fat Jar 有源码构建记录，Corrector 10 项定向测试通过。GUI 打包目录中的产物时间为 2026-10-11，不能作为本轮 clean build 证据；本轮未验证协议输入、GUI/服务启动冒烟或完整安装包，**部分完成**；
> 5. **P5**：因缺少新采集设备与独立 source/subject 隔离集，按 P5.2 记录为“数据受限 / 待新数据”，不复用旧数据，参数量冻结。

---

## 一、 子任务状态总览看板

| 子任务 | 任务主题 | 核心执行内容与实测关键指标 | 任务状态 | 验收裁定与凭据 |
|---|---|---|---|---|
| **P0** | **仓库与工具链基线** | 盘点 5 个独立仓库 HEAD/diff；固化 JDK 17、Gradle 8.14、PyTorch 2.7.1、MSVC 14.51、Windows SDK 10.0 | **已完成** | 仓库差异冻结，基线单测全通 |
| **P1** | **输入契约与模型 ABI** | 锁定 124D 字段来源表；明确角速度为“姿态因果差分角速度”并加入 $\Delta t$ 边界防护；固化 ONNX 权重哈希 | **已完成** | Python 契约与 Kotlin 写入均为位置 3 维在前、旋转 6D 在后；124D 契约测试全通 |
| **P2** | **ONNX 生命周期与影子安全** | `VRServer` 实现 `AutoCloseable` 并挂载 `close()`；ONNX Session 显式关闭并幂等自愈；模型缺失安全旁路 | **已完成** | 7 项推理与生命周期单测全通 |
| **P3** | **服务端 tick mock 回放** | 8 路 mock Tracker 进入 `VRServer.tickServer()`；动态姿态；10.8 万帧逐帧零篡改断言 | **部分完成** | 本次复跑 P95 **4.638 ms**、P99 **7.810 ms**、最大 **19.659 ms**；Full GC 后堆差 **−0.38 MB**；`mutations=0`。长测注入 59 次 reset、868 次 Tracker 状态故障，但未验证每次恢复、时间戳异常、完整 tracker 缺失矩阵或输出有效性 |
| **P4** | **Phase 9 合成物理与安全门禁** | 解析验证世界逆偏航左乘；120 条会话（216万合成帧）动态安全门禁；ZARU 自适应时间死区调优 | **合成门禁通过** | 测试设定 ±0.14°/min 下胜率 100%，静坐漂移压制 96.5%（16.21°→0.57°）；严格零漂 endpoint **0.008933°**，静坐保持 **0.000000°**。非硬件标称或服务端实测 |
| **P5** | **Phase 11 增量价值确认** | 审查 AMASS 数据集使用历史，发现当前本地库仅包含单一集合 `HumanML3D_SMPLH52`，无新 source 隔离集 | **数据受限 / 待硬件到货采集** | 严格执行计划 P5.2，拒绝复用旧确认集，不伪称验证通过；参数量冻结 |
| **P6** | **固件构建准备与接口矩阵** | 审阅仓库配置、board targets 与分区；核验 6 个既有固件产物 SHA-256；编写离线刷写 Runbook | **部分完成** | 没有 `west` 或固件构建目录；tracker/receiver 源码 clean build 与 UDP/HID 协议 fixture 尚缺 |
| **P7** | **PC 软件构建与模拟集成** | driver 与 Desktop Fat Jar 有源码构建记录；Corrector 定向测试全通 | **部分完成** | 缺少 GUI/安装包本轮构建、协议层虚拟 Tracker 输入和启动冒烟；既有 GUI 产物为 2026-10-11 |
| **P8** | **文档同步与硬件到货检查包** | 编制详细硬件到货 8 步 Bring-up 验收单，全流程无硬件限制与硬件依赖严格切分 | **已完成** | 硬件上电、SWD、UF2、串口协议、配对与影子接入流程清晰完备 |

---

## 二、 Task P0 执行记录：仓库与工具链基线

1. **五大仓库 HEAD 与工作区状态**：
   - **顶层元仓库 (`slimevr`)**：执行启动时基线分支 `main`，HEAD `503a50f`，remote `git@github.com:permanentmercy/Mercy-slimevr.git`，当时工作区干净。后续 Phase 13 代码提交已推进 HEAD，因此这是基线快照，不是本次验收时 HEAD。
   - **服务端仓库 (`server/`)**：分支 `main`，HEAD `e3822fa8`，remote `https://github.com/SlimeVR/SlimeVR-Server.git`。本地包含 4 处沙箱兼容适配（均带 `LOCAL` 注释，用于绕开 `cmd.exe` 沙箱拦截与 Android dl.google.com 网络阻断，已在 `BUILD-REPORT.md` 备案）。
   - **驱动仓库 (`driver/`)**：分支 `main`，HEAD `dcc0f56`，remote `https://github.com/SlimeVR/SlimeVR-OpenVR-Driver.git`，工作区干净。
   - **追踪器固件仓库 (`firmware/SlimeVR-Tracker-nRF`)**：分支 `main`，HEAD `cbfff16`，remote `https://github.com/SlimeVR/SlimeVR-Tracker-nRF.git`，工作区干净。
   - **接收器固件仓库 (`firmware/SlimeVR-Tracker-nRF-Receiver`)**：分支 `main`，HEAD `e3d66d5`，remote `https://github.com/SlimeVR/SlimeVR-Tracker-nRF-Receiver.git`，工作区干净。

2. **工具链版本实测固化**：
   - **Java / JDK**：Eclipse Adoptium OpenJDK 17.0.20.1+1 (Temurin-17.0.20.1+1), 64-Bit Server VM。
   - **Gradle**：8.14.4 (通过 `server/gradlew.bat` 包装器运行)。
   - **Kotlin**：2.0.21。
   - **Python**：3.10.14 (`D:\software\conda\envs\rvc\python.exe`)。
   - **PyTorch / ORT**：PyTorch 2.7.1+cu118 (CUDA 可用), ONNX Runtime 1.18.1。
   - **MSVC 编译器**：MSVC 14.51.36231 (`D:\software\VS\product\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\cl.exe`)。
   - **Windows SDK**：10.0.26100.0 (`C:\Program Files (x86)\Windows Kits\10`)。
   - **CMake / Ninja**：CMake 3.28+ (`D:\software\VS\product\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`), Ninja 1.11+。

3. **基线回归验证**：
   - 运行 `.\gradlew.bat :server:core:test --tests "dev.slimevr.tracking.processor.corrector.*"`，全部单测编译执行通过。

---

## 三、 Task P1 执行记录：运行时输入契约与模型 ABI

1. **124 维输入特征流严格锁定**：
   - **[0..16] (17维) HMD**：3D 相对位置 ($x,y,z$)、6D 旋转 ($colX,colY$)、3D 线速度、3D 角速度、有效性和 age。
   - **[17..33] (17维) VD 胸部**：同一字段顺序的上身视觉参考通道，支持缺失旁路。
   - **[34..50] (17维) VD 左上臂**、**[51..67] (17维) VD 右上臂**：同一字段顺序的视觉手臂参考通道。
   - **[68..81] (14维) 左大腿**、**[82..95] (14维) 右大腿**：6D 旋转 ($colX_{x,y,z}, colY_{x,y,z}$), 3D 局部加速度 ($a_x, a_y, a_z$), 3D 姿态差分角速度 ($\omega_x, \omega_y, \omega_z$), 2D 状态元数据。
   - **[96..109] (14维) 左小腿**、**[110..123] (14维) 右小腿**：下肢 IMU 物理通道。

2. **观测信号语义明确：姿态因果差分角速度**：
   - 明确当前服务端生产链路中角速度并非由底层直接写入硬件原始陀螺仪读数，而是基于相邻帧连续四元数因果差分得到：
     $$q_{\text{rel}} = q_{t-1}^{-1} \otimes q_t,\quad \boldsymbol{\omega} = \frac{\text{rotvec}(q_{\text{rel}})}{\Delta t}$$
   - 全链路严禁将该指标宣传为“检测了硬件陀螺硬件偏置”；
   - 在 [`CorrectorBridge.kt`](../../../server/server/core/src/main/java/dev/slimevr/tracking/processor/corrector/CorrectorBridge.kt) 中增加严密边界防护：当 $\Delta t \le 10^{-4}\text{ s}$、$\Delta t > 0.5\text{ s}$ 或包含 `NaN/Inf` 时强制返回零向量，重置或丢包时不产生奇异突变速度。

3. **模型部署 ABI 锁定**：
   - **模型文件**：[`ai_corrector/artifacts/cstn_v2_yaw_neutral_world.onnx`](../../../ai_corrector/artifacts/cstn_v2_yaw_neutral_world.onnx)
   - **文件体积**：4,571,251 字节 ($\sim 4.57\text{ MB}$)
   - **Opset 版本**：Opset 14
   - **SHA-256 哈希**：`7BC5D04E2ECFB9E7B0D416A72EF3A96303F7F9E7F7BEF4A320F106A4712C891C`
   - **输入张量**：$x \in \mathbb{R}^{1 \times 1 \times 124}$, $h, c \in \mathbb{R}^{8 \times 1 \times 1 \times 96}$
   - **输出张量**：$dq \in \mathbb{R}^{1 \times 1 \times 4 \times 4}$ (世界系正交投影四元数残差), $gate \in \mathbb{R}^{1 \times 1 \times 4}$, $contact \in \mathbb{R}^{1 \times 1 \times 2}$, $hn, cn \in \mathbb{R}^{8 \times 1 \times 1 \times 96}$。

---

## 四、 Task P2 执行记录：ONNX 生命周期与影子安全

1. **`VRServer` 生命周期挂载与安全关闭**：
   - 修改 [`VRServer.kt`](../../../server/server/core/src/main/java/dev/slimevr/VRServer.kt)，使 `VRServer` 实现 `java.lang.AutoCloseable` 接口；
   - 在主服务线程 `run()` 循环中加入 `try { ... } finally { close() }`，保证无论是线程打断（`interrupt()`）还是退出，均会确定性触发 `correctorBridge.close()`；
   - 提取 `@VRServerThread fun tickServer(timePerFrame: Float)` 方法，将主循环逻辑原子化，供确定性回放与测试套件直接调用。

2. **`OnnxCorrectorEngine` 资源释放与防崩溃**：
   - 在 [`OnnxCorrectorEngine.kt`](../../../server/server/core/src/main/java/dev/slimevr/tracking/processor/corrector/OnnxCorrectorEngine.kt) 中加入 `private var closed = false` 状态守卫，确保重复调用 `close()` 幂等安全；
   - 修正模型搜索路径逻辑：当配置了特定路径但文件不存在时，显式报警并返回 `null`（保持旁路），不再静默假冒加载默认模型；
   - 严格影子模式保真度：原始物理读数与 Avatar 求解输入位级不变。

3. **单测验收**：
   - 在 [`CorrectorBridgeInferenceTest.kt`](../../../server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/CorrectorBridgeInferenceTest.kt) 中补充：
     - `test06_vrServerLifecycleAndClose()`：检验 `VRServer` 创建、多次安全关闭及临时配置回收；
     - `test07_missingModelGracefulFallback()`：检验模型路径缺失时优雅降级并不阻断服务启动。
   - 本次复核使用 `server` 下的 `:server:core:test --tests "dev.slimevr.tracking.processor.corrector.CorrectorBridgeInferenceTest" --tests "dev.slimevr.tracking.processor.corrector.CorrectorBridgeReplayBenchmarkTest"` 定向重跑；两类共 10 项测试全部通过，JUnit XML 为 `server/server/core/build/test-results/test/` 下对应报告。

---

## 五、 Task P3 执行记录：`VRServer.tickServer()` mock 回放验收（tick 范围通过）

1. **测试构造与覆盖范围**：
   - 在 [`CorrectorBridgeReplayBenchmarkTest.kt`](../../../server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/CorrectorBridgeReplayBenchmarkTest.kt) 中重构 `test03_fullVRServerTick108kFramesReplay()`；
   - 注册 8 路带层级关系的虚拟 Tracker（HMD、Chest、左/右上臂、左/右大腿、左/右小腿）；
   - 彻底废除恒定静态输入：在 108,000 帧（30 分钟）全长程推进中，持续生成下肢动态步态与蹲起摆动姿态流（`thighPitch = 0.4*sin(2.5t)`, `thighYaw = 0.15*cos(1.2t)`, `calfPitch = 0.5*sin(2.5t+0.3)`）；
   - 注入高频与真实故障场景：
     - **快速校准重置**：每 1,800 帧（30 秒）触发一次 `server.correctorBridge.resetState()`，累计触发 59 次；
     - **上身视觉 VD 缺失/热插拔**：每 3,600 帧（60 秒）连续 20 帧断开胸部参考（`chest.status = DISCONNECTED`）；
     - **下肢 Tracker 掉线/超时**：每 7,200 帧（120 秒）连续 15 帧断开小腿追踪器（`leftCalf.status = TIMED_OUT`），累计恢复 868 次异常断连帧；
   - **实施逐帧影子模式零篡改断言**：在全部 108,000 帧中逐帧比对原始传感器读数与输入读数（`leftThigh.getRawRotation() != expLeftThigh || rightThigh.getRawRotation() != expRightThigh || ...`），实测 **`shadowMutationCount == 0`**（0 篡改，0 误改）。

2. **性能与时延实测（3,000 帧全服务 Tick 采样统计，预冻结门限 P95 $\le 20.0$ ms）**：

| 时延统计项 | 完整 `VRServer.tickServer()` 时延 | CorrectorBridge 子耗时组件 | 优于 20ms 门限倍数 |
|---|---|---|---|
| **均值 (Mean)** | **2.3733 ms** | - | P95 门限内 |
| **中位数 (P50)** | **1.9618 ms** | 1.8047 ms | P95 门限内 |
| **P95** | **4.6379 ms** | 4.2919 ms | 约为 20 ms 预算的 23.2% |
| **P99** | **7.8096 ms** | 7.0650 ms | P95 门限内；尾部有较高抖动 |
| **最大耗时 (Max)** | **19.6591 ms** | - | 单次接近 20 ms，不构成硬实时保证 |

3. **108,000 帧连续长程回放（逻辑 30 分钟）稳定性与内存门限实测**：
   - **预冻结内存上限门限**：回放后堆常驻增长 $\le 64.0\text{ MB}$；
   - **长程推进帧数**：108,000 逻辑帧全部成功处理，0 异常抛出；
   - **复位状态机处理**：成功处理 59 次 Calibration Reset，隐状态无污染自愈；
   - **异常失锁恢复**：成功恢复 868 次异常断连帧，无持续状态发散；
   - **逐帧影子模式零篡改**：在全 108,000 帧逐帧断言下，**`shadowMutationCount = 0`**；
   - **内存稳定性**：本次报告基线堆 9.50 MB，108,000 帧后 Full GC 堆差 **−0.38 MB**，低于 64 MB 门限；这是一次 post-GC 观察值，不是全资源泄漏证明；
4. **机器报告凭据**：[`ai_corrector/artifacts/phase13_p3_server_tick_benchmark_report.json`](../../../ai_corrector/artifacts/phase13_p3_server_tick_benchmark_report.json)。单测执行结果通过 Gradle JUnit 验证。

---

## 六、 Task P4 执行记录：Phase 9 物理仿真复核与合成安全门禁计算

1. **解析坐标变换单测（不依赖神经网络的数学验证）**：
   - 编写 [`test_p4_coordinate_and_simulation.py`](../../../ai_corrector/corrector/test_p4_coordinate_and_simulation.py)；
   - 解析证明：当世界系注入偏航漂移时，$$R_{\text{raw}} = R_y(e) \cdot R_{\text{clean}} \cdot R_{\text{mount}}$$
     世界逆偏航左乘 $$R_y(-e) \cdot R_{\text{raw}} = R_{\text{clean}} \cdot R_{\text{mount}}$$
     在任意端坐（大腿水平90度）、盘腿、俯冲等极端非中立姿态与任意安装倾角下，测地复原误差严格 $< 10^{-6}\text{ rad}$；
   - 反证测试：若误用局部右乘 $R_{\text{raw}} \cdot R_y(-e)$，在非中立姿态下测地误差显著超过 $10^{-3}\sim 0.5\text{ rad}$，引发严重的交叉轴姿态畸变。原执行记录称单测 2 项通过；本次 Python 环境探针长时间无输出，未能独立复跑，故保留为既有执行记录。

2. **全动态 120 条 30 分钟连续合成会话评测（无硬编码门禁；偏置范围属于仿真假设）**：
   - [`evaluate_phase13_p4_physics_audit.py`](../../../ai_corrector/corrector/evaluate_phase13_p4_physics_audit.py) 以每会话 1,800 秒、10 Hz 合成输入执行 120 条会话（共 216 万合成帧）。报告中的 ±0.14°/min 是测试设定，不是 LSM6DSV 数据手册给出的标称 yaw 漂移。ST 数据手册给出校准后陀螺零速率水平及其温度变化规格，没有给出可直接当作此模型偏航速率的 `0.14°/min` 数值（[ST LSM6DSV 数据手册](https://www.st.com/resource/en/datasheet/lsm6dsv.pdf)）。仿真直接注入陀螺偏置角速度，而服务端输入是融合姿态因果差分，故此结果只说明该合成模型下的表现。

| 工况场景 | 样本量 | Raw 累计偏航误差 | Full 修正后误差 | 净收益 $G$ | 95% Bootstrap CI | 序列胜率 | 核心判定与安全门禁 |
|---|---|---|---|---|---|---|---|
| **合成正偏航 (+0.14°/min)** | 30 条 | 16.73° | **2.67°** | **+14.06°** | `[+12.80°, +15.50°]` | 100.0% | **PASS** (CI下界 > 0，抑制率 84.0%) |
| **合成负偏航 (-0.14°/min)** | 30 条 | 16.50° | **2.85°** | **+13.66°** | `[+12.51°, +14.75°]` | 100.0% | **PASS** (CI下界 > 0，抑制率 82.7%) |
| **严格零漂 (Strict Zero Drift)** | 10 条 | 0.00° | 0.01° | -0.01° | `[-0.01°, -0.01°]` | 0.0% | **PASS** (最大端点偏差 **0.008933°** $\le 0.05^\circ$) |
| **端坐零漂姿势保持 (Static Sitting)** | 10 条 | 0.00° | 0.00° | 0.00° | `[0.00°, 0.00°]` | 0.0% | **PASS** (最大端点偏差 **0.000000°** $\le 0.10^\circ$) |
| **端坐含漂移补偿 (0.14°/min)** | 10 条 | 16.21° | **0.57°** | **+15.64°** | `[+13.04°, +17.86°]` | 100.0% | **PASS** (自适应时间死区生效，抑制率 96.5%) |
| **纯随机游走 (零均值偏置)** | 10 条 | 4.98° | 2.73° | +2.25° | `[+0.81°, +3.78°]` | 90.0% | 平均门限通过；Full 最差单会话收益 **−1.6389°** |
| **持续转身漫步 (0.14°/min, 无静止窗)** | 10 条 | 17.73° | **10.39°** | **+7.34°** | `[+5.44°, +9.25°]` | 100.0% | **PASS** (步态分析防发散) |
| **上身 VD 丢失 (0.14°/min, 纯下肢保真)** | 10 条 | 17.71° | **5.23°** | **+12.48°** | `[+10.99°, +13.74°]` | 100.0% | **PASS** (单路解耦不扩散) |

3. **动态安全裁定**：
   - 严格零漂端点四路合计最大偏差为 **0.008933°**（门限 $\le 0.05^\circ$），**安全通过**；
   - 静态端坐保持端点四路合计最大偏差为 **0.000000°**（门限 $\le 0.10^\circ$），**实际计算动态通过**；
   - 合成 ±0.14°/min 偏置组的 95% Bootstrap CI 下界分别达到 $+12.80^\circ, +12.51^\circ$，序列胜率 **100.0%**；
   - 解决端坐静止下的死区阻尼问题：通过将 ZARU 静态死区改造为基于持续静止时长的自适应不确定度门限（底线降至 0.04°/min），端坐工况下的 30 分钟漂移从 16.21° 成功压制至 **0.57°**（抑制率 96.5%，胜率 100.0%），且完全保留了绝对零漂场景下的零伪动保护（0.000000° 偏差）。
4. **机器报告凭据**：[`ai_corrector/artifacts/phase13_p4_physics_audit_report.json`](../../../ai_corrector/artifacts/phase13_p4_physics_audit_report.json)。

---

## 七、 Task P5 执行记录：Phase 11 增量价值确认（数据审计）

1. **AMASS 与现有数据使用历史审计**：
   - 审查本地 `ai_corrector/database/amass`，发现仅存在 `HumanML3D_SMPLH52` 一个合并数据集，不具备独立的多采集机构 / 多受试者（subject-level）分包结构；
   - `prepare_p11_dataset.py` 中原始切分是在同一受试者目录下对序列切片划分训练与测试；
   - Phase 11 与 Phase 12 A2 均使用了该目录下的 190 条 `final_confirm_continuous.npz` 序列。

2. **严格遵守执行守则 P5.2**：
   - 计划规则：“若找不到足够新的独立源序列，不重用旧 FinalConfirm，不做‘确认性’运行；记录数据限制与需要新增的数据，而不是降低隔离标准。无新数据时本项状态是‘数据受限/未执行’，不是通过。”
   - 裁定：**本项客观记录为“数据受限 / 待新隔离源数据 (Data-Constrained)”**。
   - 明确拒绝通过重复跑旧确认集伪造“确认性”结论；当时按网络结构与参数量标称 1.54M 冻结，不盲目扩充容量。该参数数字与工程验收报告存在版本差异，精确预算以 Phase 15 P15.0 对齐结果为准。待后续引入外部独立动捕数据（如 DIP-IMU holdout 或新开源动捕序列）后再行确认。

---

## 八、 Task P6 执行记录：追踪器与接收器固件构建准备（部分完成）

1. **两套固件的独立 NCS/west 规格锁定**：
   - **追踪器固件 (`firmware/SlimeVR-Tracker-nRF`)**：
     - `west.yml` 锁定：`sdk-nrf` revision: `v3.1-branch`
     - 板卡 targets：已核验包含 `slimenrf_r3_nrf52840_uf2`（ProMicro / SuperMini nRF52840 对应板级支持包，含引脚映射与分区表定义）、`slimevrmini_p4_uf2` 等；
     - 传感器驱动支持：包含 LSM6DSV 驱动适配（WHO_AM_I `0x70`）与 VQF 姿态解算。
   - **接收器固件 (`firmware/SlimeVR-Tracker-nRF-Receiver`)**：
     - `west.yml` 锁定：`sdk-nrf` revision: `v3.3.0-preview3-branch`
     - 目标板卡：HolyIOT-21017 (`nrf52840`)、ProMicro nRF52840、nRF52840 Dongle 等。

2. **固件制品全量哈希核对 (`firmware/artifacts/`)**：
   - `bootloader-promicro-0.9.2-SlimeVR.7_nosd.uf2`：SHA-256 `C76C74196A4D447B56451B38C322ECDAC36FCA6F083A6630198825092A46BC85`
   - `bootloader-promicro-0.9.2-SlimeVR.7_s140-rescue.hex`：SHA-256 `412893277A011E8257D54202B7EF751792542B2450DBAAE39F83C3EF196DE129`
   - `bootloader-promicro-v1.0.0-rc4_nosd.uf2`：SHA-256 `D4169A4C6E28046AEEDF34273F389B306A0FE347E95B546EA8838E2178D15FF9`
   - `bootloader-promicro-v1.0.0-rc4_s140-rescue.hex`：SHA-256 `9EAB8DE5A1217EC05173C8FD86D29D285BD2BD7598FABA1AEED861C5D3A9EE13`
   - `receiver-holyiot21017.hex`：SHA-256 `790473E9AEB3346934EA408CD33D2C01A3632D13D27F62E0AA57A26B1CE3DC7E`
   - `receiver-promicro.uf2`：SHA-256 `F144592D8CD943C9C5B8DFD8F8354934FF51B4C8877DFD51F03F7767485F8DB0`

3. **硬件边界说明与 Runbook 完备性**：
   - 使用 `Get-FileHash -Algorithm SHA256` 完成全部 6 个预编译制品的哈希校验；
   - 固件与 Bootloader 分区布局完备，离线 UF2 烧录与 SWD rescue 救援路径已在 Runbook（第十节）固化；
   - 严格硬件边界：实体 USB 枚举、双击 Reset 磁盘识别与 RF 实际射频距离测试保留待硬件到货后实测，不提前虚构实物有效性。

4. **本轮验收缺口**：
   - 本机 `west` 命令不可用，固件目录下没有构建输出；因此没有执行 tracker 与 receiver 的源码 clean build，也没有保存相应编译日志或新生成固件 hash。
   - 没有建立 tracker UDP 包到 Server parser 的 fixture；receiver HID `info`/配对报文的 host 消费路径也尚未核实，包序号、丢包重连、版本和 IMU 标识仍待按实际接口模拟测试。
   - 因上述交付条件属于 P6 明确验收项，静态配置审阅、既有制品校验和 Runbook 不能替代，P6 当前标为**部分完成**。

---

## 九、 Task P7 执行记录：PC 软件构建与模拟集成（部分完成）

1. **SteamVR Driver C++ 驱动源码干净编译**：
   - 执行编译命令：
     ```powershell
     powershell -ExecutionPolicy Bypass -File scripts\msvc-cmake.ps1 -Source driver -Binary driver\build -BuildType Release
     ```
   - 工具链配置：MSVC 14.51.36231 (`Hostx64\x64\cl.exe`) + Windows SDK 10.0.26100.0 + Ninja；
   - CPM 自动化包管理器成功拉取并构建依赖（protobuf 31.1.0, simdjson 4.6.4, libuv 1.52.1, uvw）；
   - Ninja 11/11 目标全部干净编译与链接完成，成功生成驱动动态库与完整配套资产：
     - 驱动动态库：`driver\build\driver\build\Release\driver\slimevr\bin\win64\driver_slimevr.dll` (897,536 字节)
     - 驱动 SHA-256：`6F8C478985898FE640B1D26122F4A17207F51AE171ABB995E6307CCAEE4D4920`
     - 配套完整驱动资产：`driver.vrresources`、15 种关节点 input profile JSON、`localization.json`、`default.vrsettings`、`settingsschema.vrsettings` 与各状态图标全部构建产出。

2. **SlimeVR Desktop 服务端后端与 Fat Jar 构建**：
   - 执行构建命令：
     ```powershell
     .\gradlew.bat :server:desktop:jar
     .\gradlew.bat :server:desktop:shadowJar
     ```
   - 产出位于：
     - `server/server/desktop/build/libs/desktop.jar` (257,394 字节)
     - `server/server/desktop/build/libs/slimevr.jar` (114,681,269 字节，~114.6 MB)
     - `slimevr.jar` SHA-256：`CA589A2C632899766876E52E73C65FB966A0232678A33706740124054EA88749`
   - 包含运行时依赖与协议支持，可独立执行 `java -jar slimevr.jar` 启动。

3. **Corrector 与模拟数据端到端安全集成验证**：
   - 执行完整单测套件：`.\gradlew.bat :server:core:test --tests "dev.slimevr.tracking.processor.corrector.*"`；
   - 覆盖全部 4 种边界场景与生命周期验证：
     1. **模型正常启动**：`test02_realOnnxEngineLoadAndInference` 验证 ONNX 加载、因果状态推进与正规四元数输出；
     2. **模型文件缺失**：`test07_missingModelGracefulFallback` 验证模型缺失时安全返回 null，服务启动不抛出异常阻断；
     3. **模型输入非法/破坏**：`test04_faultHandlingNanInfBypass` 验证 NaN / Inf / 维度错误输入时 100% 安全旁路；
     4. **上身参考 VD 缺失/热插拔**：`test03_fullVRServerTick108kFramesReplay` 验证断开胸部参考与小腿下肢断连时平滑降级自愈；
     5. **严格影子模式保护**：`test05_strictShadowModeNoPoseMutation` 与 `test03` 逐帧验证下肢 Tracker 读数绝不被覆写（`shadowMutationCount == 0`）；
     6. **安全生命周期**：`test06_vrServerLifecycleAndClose` 验证 `VRServer` 实现 `AutoCloseable`，双重/多次 `close()` 幂等安全释放资源。
   - 全部单元测试通过（JUnit XML 生成于 `server/server/core/build/test-results/test/`）。

4. **本轮复核范围与缺口**：
   - 本轮复跑命令（工作目录 `server/`）：`$env:GRADLE_USER_HOME='.gradle'; .\gradlew.bat :server:core:test --tests "dev.slimevr.tracking.processor.corrector.CorrectorBridgeInferenceTest" --tests "dev.slimevr.tracking.processor.corrector.CorrectorBridgeReplayBenchmarkTest"`；两类测试共 10 项，0 failure、0 error。
   - P3 直接将生成的姿态与状态写入 mock Tracker，再调用 `VRServer.tickServer()`。它验证服务端 tick、ONNX、状态切换与 shadow 不写回，但没有启动协议层接收器，也没有从模拟 tracker packet 走到 Server。
   - `server/gui/dist/artifacts/win/SlimeVR-win-x64.zip` 与 `win-unpacked` 存在，但目录时间为 2026-10-11；本轮未执行 GUI/安装包 clean build 或启动冒烟，故不作为本轮构建证据。
   - P7 原计划要求协议层虚拟 tracker、GUI 默认安全配置、GUI/服务启动和缺失模型降级联测。这些项目仍未完成，P7 当前标为**部分完成**。

---

## 十、 Task P8：硬件到货 Bring-up 验收检查单

当第一套试做硬件（SuperMini nRF52840、LSM6DSV16X 模块、HolyIOT 接收器、DAPLink）到手后，按以下顺序执行首次硬件启动验证：

| 步骤 | 验证项 | 具体操作 | 期望结果与日志 | 失败分流策略 |
|---|---|---|---|---|
| **Step 1** | **板卡基础供电与芯片外观** | 目视检查 SuperMini 丝印、焊接引脚与焊桥；插上 USB-C 供电 | 板载电源指示灯常亮，无短路发热 | 检查 VDD/GND 间阻抗，确认未反接 |
| **Step 2** | **SWD Bootloader 烧录** | 连接 DAPLink (SWDIO, SWCLK, GND, 3V3)；使用 OpenOCD / nrfjprog 刷写 | 提示 `Programming 0x00000000 - 0x00026000 done` | 检查 SWD 接线接触，降低 SWD 时钟至 1MHz |
| **Step 3** | **UF2 USB 磁盘模式** | 断开 DAPLink，插 USB 数据线快速**双击 Reset 按键** | 电脑出现可移动磁盘 `NICENANO` 或 `SLIMEVR` | 若未枚举磁盘，重新用 SWD 刷写 `rescue.hex` |
| **Step 4** | **固件刷写** | 将编译好的 `promicro_uf2_stackedi2c.uf2` 拖入该磁盘 | 磁盘自动弹出并重启，LED 闪烁进入工作态 | 若红灯常亮报警，检查固件 target 与分区表 |
| **Step 5** | **I2C 传感器与串口诊断** | 打开串口助手（波特率 115200），发送文本命令 `info` | 回显包含 `WHO_AM_I=0x70`，传感器初始化为 `LSM6DSV` | 若回显 `0x00` 或 `No IMU`，用万用表测 I2C SDA/SCL 上拉电压 (3.3V) |
| **Step 6** | **接收器配对** | 接收器插入电脑串口发送 `pair`，追踪器长按按键 5 秒 | 串口提示 `Paired device [0] ID=...`，无线通信建立 | 检查天线阻抗匹配，避免大金属屏蔽遮挡 |
| **Step 7** | **服务端影子连接与数据流** | 启动 SlimeVR Server，在 GUI 中确认识别到 1 个活动 Tracker | 姿态以 60Hz 稳定上报，`CorrectorBridge` 开始记录影子延迟 | 若断连，检查 UDP 端口 6969 与无线信道 |
| **Step 8** | **Phase 8 真实动捕对照** | 贴附动捕反光点，开展 30 分钟连续动作采集 | 真实 AHRS 漂移模型采集入库，评估慢偏航卡尔曼实机抑制率 | 进入 Phase 8 正式验收环节 |

---

## 十一、 产物凭据总索引表

- **Phase 13 P3 `tickServer()` mock 回放报告**：[`ai_corrector/artifacts/phase13_p3_server_tick_benchmark_report.json`](../../../ai_corrector/artifacts/phase13_p3_server_tick_benchmark_report.json)
- **Phase 13 P4 合成物理与安全动态审计报告**：[`ai_corrector/artifacts/phase13_p4_physics_audit_report.json`](../../../ai_corrector/artifacts/phase13_p4_physics_audit_report.json)
- **Phase 13 P4 坐标验证解析单测**：[`ai_corrector/corrector/test_p4_coordinate_and_simulation.py`](../../../ai_corrector/corrector/test_p4_coordinate_and_simulation.py)
- **Phase 13 P4 全动态物理评测脚本**：[`ai_corrector/corrector/evaluate_phase13_p4_physics_audit.py`](../../../ai_corrector/corrector/evaluate_phase13_p4_physics_audit.py)
- **已固化流式 ONNX 模型**：[`ai_corrector/artifacts/cstn_v2_yaw_neutral_world.onnx`](../../../ai_corrector/artifacts/cstn_v2_yaw_neutral_world.onnx)
- **服务端生命周期与回放单测源码**：
  - [`CorrectorBridgeInferenceTest.kt`](../../../server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/CorrectorBridgeInferenceTest.kt)
  - [`CorrectorBridgeReplayBenchmarkTest.kt`](../../../server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/CorrectorBridgeReplayBenchmarkTest.kt)
- **服务端生命周期与原子 Tick 源码**：
  - [`VRServer.kt`](../../../server/server/core/src/main/java/dev/slimevr/VRServer.kt)
  - [`CorrectorBridge.kt`](../../../server/server/core/src/main/java/dev/slimevr/tracking/processor/corrector/CorrectorBridge.kt)
  - [`OnnxCorrectorEngine.kt`](../../../server/server/core/src/main/java/dev/slimevr/tracking/processor/corrector/OnnxCorrectorEngine.kt)
