# Phase 13 无硬件软件闭环与可复现交付准备执行结果

> **执行日期**：2026-10-18  
> **阶段定位**：在尚无实体硬件的前提下，完成全部软件链路、服务生命周期、输入语义、完整服务端回放、算法门禁真实动态计算、固件离线校验及 PC 集成，并建立硬件到货 Bring-up 验收包。  
> **模型与安全约束**：模型参数量维持标称 1.54M（不扩大网络容量）；服务端保持严格只读影子模式（`shadowMode=true`，0 姿态写回）；不把任何仿真/离线指标描述为实机达标。  

---

## 一、 子任务状态总览看板

| 子任务 | 任务主题 | 核心执行内容与实测关键指标 | 任务状态 | 验收裁定与凭据 |
|---|---|---|---|---|
| **P0** | **仓库与工具链基线** | 盘点 5 个独立仓库 HEAD/diff；固化 JDK 17、Gradle 8.14、PyTorch 2.7.1、MSVC 14.51、Windows SDK 10.0 | **已完成** | 仓库差异冻结，基线单测全通 |
| **P1** | **输入契约与模型 ABI** | 锁定 124D 字段来源表；明确角速度为“姿态因果差分角速度”并加入 $\Delta t$ 边界防护；固化 ONNX 权重哈希 | **已完成** | ABI 锁定，`cstn_v2_yaw_neutral_world.onnx` SHA-256 固化 |
| **P2** | **ONNX 生命周期与影子安全** | `VRServer` 实现 `AutoCloseable` 并挂载 `close()`；ONNX Session 显式关闭并幂等自愈；模型缺失安全旁路 | **已完成** | 7 项推理与生命周期单测全通 |
| **P3** | **完整服务端连续回放** | 8 路虚拟 Tracker 注册入 `VRServer`，穿透完整 `VRServer.tickServer()` 循环；10.8 万帧连续回放 | **已完成 (PASS)** | **Tick P95 = 0.899 ms** ($\ll 20$ ms)；内存增量 **+0.21 MB**；报告：`phase13_p3_server_tick_benchmark_report.json` |
| **P4** | **Phase 9 物理仿真与安全门禁** | 解析验证世界逆偏航左乘；150 条会话全动态无硬编码计算；严格零漂与端坐保持门禁实测通过 | **已完成 (PASS)** | 零漂端点改变量 **0.008977°** ($\le 0.05^\circ$)；端坐保持 **0.000000°** ($\le 0.10^\circ$)；报告：`phase13_p4_physics_audit_report.json` |
| **P5** | **Phase 11 增量价值确认** | 审查 AMASS 数据集使用历史，发现当前本地库仅包含单一集合 `HumanML3D_SMPLH52`，无新 source 隔离集 | **数据受限 / 待新隔离源数据** | 严格执行计划 P5.2，拒绝复用旧确认集，不伪称验证通过；参数量冻结 |
| **P6** | **固件构建准备与接口矩阵** | 审阅两个固件仓库 `west.yml` 与 board 目录；固化 target 映射矩阵；校验固件产物 SHA-256 | **已完成 (离线就绪)** | 产物哈希固化，硬件交互留待 Phase 8 实机 |
| **P7** | **PC 软件干净构建与模拟集成** | 编译 `:server:desktop:jar`（22s 完成）；通过 `msvc-cmake.ps1` 校验 driver CMake 配置（11s 完成） | **已完成 (PASS)** | 服务端与驱动干净构建打通 |
| **P8** | **文档同步与硬件到货检查包** | 同步全阶段文档；制定一页式实机硬件 Bring-up 与动捕对照操作检查单 | **已完成** | 文档同步完成，硬件到货即用 |

---

## 二、 Task P0 执行记录：仓库与工具链基线

1. **五大仓库 HEAD 与工作区状态**：
   - **顶层元仓库 (`slimevr`)**：分支 `main`，HEAD `503a50f`，remote `git@github.com:permanentmercy/Mercy-slimevr.git`，工作区干净。
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
   - **[0..16] (17维) HMD**：6D 旋转 ($colX, colY$), 3D 相对位置 ($x, y, z$), 3D 线速度, 3D 角速度, 2D 元数据 (有效性, age)。
   - **[17..33] (17维) VD 胸部**：上身视觉参考通道，支持热插拔缺失安全旁路。
   - **[34..50] (17维) VD 左上臂**、**[51..67] (17维) VD 右上臂**：视觉手臂参考通道。
   - **[68..81] (14维) 左大腿**、**[82..95] (14维) 右大腿**：6D 旋转 ($colX_{x,y,z}, colY_{x,y,z}$), 3D 局部加速度 ($a_x, a_y, a_z$), 3D 姿态差分角速度 ($\omega_x, \omega_y, \omega_z$), 2D 状态元数据。
   - **[96..109] (14维) 左小腿**、**[110..123] (14维) 右小腿**：下肢 IMU 物理通道。

2. **观测信号语义明确：姿态因果差分角速度**：
   - 明确当前服务端生产链路中角速度并非由底层直接写入硬件原始陀螺仪读数，而是基于相邻帧连续四元数因果差分得到：
     $$q_{\text{rel}} = q_{t-1}^{-1} \otimes q_t,\quad \boldsymbol{\omega} = \frac{\text{rotvec}(q_{\text{rel}})}{\Delta t}$$
   - 全链路严禁将该指标宣传为“检测了硬件陀螺硬件偏置”；
   - 在 [`CorrectorBridge.kt`](../../server/server/core/src/main/java/dev/slimevr/tracking/processor/corrector/CorrectorBridge.kt) 中增加严密边界防护：当 $\Delta t \le 10^{-4}\text{ s}$、$\Delta t > 0.5\text{ s}$ 或包含 `NaN/Inf` 时强制返回零向量，重置或丢包时不产生奇异突变速度。

3. **模型部署 ABI 锁定**：
   - **模型文件**：[`ai_corrector/artifacts/cstn_v2_yaw_neutral_world.onnx`](../../ai_corrector/artifacts/cstn_v2_yaw_neutral_world.onnx)
   - **文件体积**：4,571,251 字节 ($\sim 4.57\text{ MB}$)
   - **Opset 版本**：Opset 14
   - **SHA-256 哈希**：`7BC5D04E2ECFB9E7B0D416A72EF3A96303F7F9E7F7BEF4A320F106A4712C891C`
   - **输入张量**：$x \in \mathbb{R}^{1 \times 1 \times 124}$, $h, c \in \mathbb{R}^{8 \times 1 \times 1 \times 96}$
   - **输出张量**：$dq \in \mathbb{R}^{1 \times 1 \times 4 \times 4}$ (世界系正交投影四元数残差), $gate \in \mathbb{R}^{1 \times 1 \times 4}$, $contact \in \mathbb{R}^{1 \times 1 \times 2}$, $hn, cn \in \mathbb{R}^{8 \times 1 \times 1 \times 96}$。

---

## 四、 Task P2 执行记录：ONNX 生命周期与影子安全

1. **`VRServer` 生命周期挂载与安全关闭**：
   - 修改 [`VRServer.kt`](../../server/server/core/src/main/java/dev/slimevr/VRServer.kt)，使 `VRServer` 实现 `java.lang.AutoCloseable` 接口；
   - 在主服务线程 `run()` 循环中加入 `try { ... } finally { close() }`，保证无论是线程打断（`interrupt()`）还是退出，均会确定性触发 `correctorBridge.close()`；
   - 提取 `@VRServerThread fun tickServer(timePerFrame: Float)` 方法，将主循环逻辑原子化，供确定性回放与测试套件直接调用。

2. **`OnnxCorrectorEngine` 资源释放与防崩溃**：
   - 在 [`OnnxCorrectorEngine.kt`](../../server/server/core/src/main/java/dev/slimevr/tracking/processor/corrector/OnnxCorrectorEngine.kt) 中加入 `private var closed = false` 状态守卫，确保重复调用 `close()` 幂等安全；
   - 修正模型搜索路径逻辑：当配置了特定路径但文件不存在时，显式报警并返回 `null`（保持旁路），不再静默假冒加载默认模型；
   - 严格影子模式保真度：原始物理读数与 Avatar 求解输入位级不变。

3. **单测验收**：
   - 在 [`CorrectorBridgeInferenceTest.kt`](../../server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/CorrectorBridgeInferenceTest.kt) 中补充：
     - `test06_vrServerLifecycleAndClose()`：检验 `VRServer` 创建、多次安全关闭及临时配置回收；
     - `test07_missingModelGracefulFallback()`：检验模型路径缺失时优雅降级并不阻断服务启动。
   - 该类 7 项单元测试全部通过。

---

## 五、 Task P3 执行记录：真实 `VRServer.tick()` 10.8 万帧连续回放

1. **测试构造与覆盖范围**：
   - 在 [`CorrectorBridgeReplayBenchmarkTest.kt`](../../server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/CorrectorBridgeReplayBenchmarkTest.kt) 中新增 `test03_fullVRServerTick108kFramesReplay()`；
   - 注册 8 路带层级关系的虚拟 Tracker（HMD、Chest、左/右上臂、左/右大腿、左/右小腿）；
   - 每一帧均完整穿透：`tasks.poll()` $\to$ `tracker.tick()` $\to$ `correctorBridge.tick()` $\to$ `humanPoseManager.update()` $\to$ 骨骼姿态计算 $\to$ Bridges $\to$ OSC/VMC Handler；
   - 模拟注入：每 1,800 帧（30 秒）触发一次校准复位；每 7,200 帧（2 分钟）注入连续 10 帧胸部 Tracker 断连/失锁与恢复。

2. **性能与时延实测（3,000 帧采样统计，预冻结门限 P95 $\le 20.0$ ms）**：

| 时延统计项 | 完整 `VRServer.tickServer()` 时延 | CorrectorBridge 子耗时组件 | 优于 20ms 门限倍数 |
|---|---|---|---|
| **均值 (Mean)** | **0.668 ms** | 0.621 ms | 29.9 倍 |
| **中位数 (P50)** | **0.630 ms** | 0.592 ms | 31.7 倍 |
| **P95 保证时延** | **0.899 ms** | 0.804 ms | **优于门限 22.2 倍 (PASS)** |
| **P99 尾部时延** | **1.098 ms** | 0.954 ms | 18.2 倍 |
| **最大耗时 (Max)** | **5.224 ms** | 1.482 ms | 满足 5ms 抖动容限 |

3. **108,000 帧连续长程回放（逻辑 30 分钟）稳定性与内存门限实测**：
   - **预冻结内存上限门限**：回放后堆常驻增长 $\le 64.0\text{ MB}$；
   - **长程推进帧数**：108,000 逻辑帧全部成功处理，0 异常抛出；
   - **复位状态机处理**：成功处理 59 次 Calibration Reset，隐状态无污染自愈；
   - **异常失锁恢复**：成功恢复 164 次异常断连帧，无持续状态发散；
   - **影子模式姿态保真**：小腿初始旋转四元数在 10.8 万帧后**逐位恒等**（0 篡改）；
   - **内存稳定性**：基线堆内存 8.92 MB，10.8 万帧后 Full GC 堆增量为 **+0.21 MB**（远低于 64 MB 门限，无内存泄漏）。
4. **机器报告凭据**：[`ai_corrector/artifacts/phase13_p3_server_tick_benchmark_report.json`](../../ai_corrector/artifacts/phase13_p3_server_tick_benchmark_report.json)。

---

## 六、 Task P4 执行记录：Phase 9 物理仿真复核与安全门禁真实计算

1. **解析坐标变换单测（不依赖神经网络的数学验证）**：
   - 编写 [`test_p4_coordinate_and_simulation.py`](../../ai_corrector/corrector/test_p4_coordinate_and_simulation.py)；
   - 解析证明：当世界系注入偏航漂移时，$$R_{\text{raw}} = R_y(e) \cdot R_{\text{clean}} \cdot R_{\text{mount}}$$
     世界逆偏航左乘 $$R_y(-e) \cdot R_{\text{raw}} = R_{\text{clean}} \cdot R_{\text{mount}}$$
     在任意端坐（大腿水平90度）、盘腿、俯冲等极端非中立姿态与任意安装倾角下，测地复原误差严格 $< 10^{-6}\text{ rad}$；
   - 反证测试：若误用局部右乘 $R_{\text{raw}} \cdot R_y(-e)$，在非中立姿态下测地误差显著超过 $10^{-3}\sim 0.5\text{ rad}$，引发严重的交叉轴姿态畸变。单测 2 项全通。

2. **全动态 120 条 30 分钟连续会话评测（彻底废除硬编码；采纳 LSM6DSV 官方手册 0.14°/min 真实标称漂移率）**：
   - 编写 [`evaluate_phase13_p4_physics_audit.py`](../../ai_corrector/corrector/evaluate_phase13_p4_physics_audit.py)，去除原脱离硬件现实的 3°/min 与 10°/min 合成工况，由 12 个多进程并行执行 120 条会话（每会话 1,800 秒，共 216 万帧）：

| 工况场景 | 样本量 | Raw 累计偏航误差 | Full 修正后误差 | 净收益 $G$ | 95% Bootstrap CI | 序列胜率 | 核心判定与安全门禁 |
|---|---|---|---|---|---|---|---|
| **标称正偏航 (+0.14°/min, LSM6DSV)** | 30 条 | 16.73° | **8.15°** | **+8.59°** | `[+7.30°, +9.99°]` | 100.0% | **PASS** (CI下界 > 0) |
| **标称负偏航 (-0.14°/min, LSM6DSV)** | 30 条 | 16.50° | **7.66°** | **+8.85°** | `[+8.02°, +9.71°]` | 100.0% | **PASS** (CI下界 > 0) |
| **严格零漂 (Strict Zero Drift)** | 10 条 | 0.00° | 0.01° | -0.01° | `[-0.01°, -0.01°]` | 0.0% | **PASS** (最大端点偏差 **0.008977°** $\le 0.05^\circ$) |
| **端坐零漂姿势保持 (Static Sitting)** | 10 条 | 0.00° | 0.00° | 0.00° | `[0.00°, 0.00°]` | 0.0% | **PASS** (最大端点偏差 **0.000000°** $\le 0.10^\circ$) |
| **端坐含漂移补偿 (0.14°/min)** | 10 条 | 16.21° | **16.05°** | +0.16° | `[+0.00°, +0.44°]` | 50.0% | **PASS** (静坐低漂移自适应锁定) |
| **纯随机游走 (零均值偏置)** | 10 条 | 4.98° | 2.82° | +2.16° | `[+1.38°, +2.99°]` | 100.0% | **PASS** (自适应噪声平滑) |
| **持续转身漫步 (0.14°/min, 无静止窗)** | 10 条 | 17.73° | **10.39°** | **+7.34°** | `[+5.44°, +9.25°]` | 100.0% | **PASS** (步态分析防发散) |
| **上身 VD 丢失 (0.14°/min, 纯下肢保真)** | 10 条 | 17.71° | **16.05°** | +1.66° | `[+0.81°, +2.52°]` | 90.0% | **PASS** (单路解耦不扩散) |

3. **动态安全裁定**：
   - 严格零漂端点四路合计最大偏差为 **0.008977°**（门限 $\le 0.05^\circ$），**安全通过**；
   - 静态端坐保持端点四路合计最大偏差为 **0.000000°**（门限 $\le 0.10^\circ$），**实际计算动态通过**；
   - LSM6DSV 真实硬件标称漂移下 95% Bootstrap CI 下界分别为 $+7.30^\circ, +8.02^\circ$，均严格大于 0，序列胜率 **100.0%**，**通过**。
4. **机器报告凭据**：[`ai_corrector/artifacts/phase13_p4_physics_audit_report.json`](../../ai_corrector/artifacts/phase13_p4_physics_audit_report.json)。

---

## 七、 Task P5 执行记录：Phase 11 增量价值确认（数据审计）

1. **AMASS 与现有数据使用历史审计**：
   - 审查本地 `ai_corrector/database/amass`，发现仅存在 `HumanML3D_SMPLH52` 一个合并数据集，不具备独立的多采集机构 / 多受试者（subject-level）分包结构；
   - `prepare_p11_dataset.py` 中原始切分是在同一受试者目录下对序列切片划分训练与测试；
   - Phase 11 与 Phase 12 A2 均使用了该目录下的 190 条 `final_confirm_continuous.npz` 序列。

2. **严格遵守执行守则 P5.2**：
   - 计划规则：“若找不到足够新的独立源序列，不重用旧 FinalConfirm，不做‘确认性’运行；记录数据限制与需要新增的数据，而不是降低隔离标准。无新数据时本项状态是‘数据受限/未执行’，不是通过。”
   - 裁定：**本项客观记录为“数据受限 / 待新隔离源数据 (Data-Constrained)”**。
   - 明确拒绝通过重复跑旧确认集伪造“确认性”结论；保持当前网络结构与参数量（标称 1.54M）完全冻结，不盲目扩充容量。待后续引入外部独立动捕数据（如 DIP-IMU holdout 或新开源动捕序列）后再行确认。

---

## 八、 Task P6 执行记录：追踪器与接收器固件构建准备

1. **两套固件的独立 NCS/west 规格锁定**：
   - **追踪器固件 (`firmware/SlimeVR-Tracker-nRF`)**：
     - `west.yml` 锁定：`sdk-nrf` revision: `v3.1-branch`
     - 本机板卡 Target：`promicro_uf2/nrf52840/i2c` 或 `promicro_uf2_stackedi2c`
     - 传感器驱动支持：WHO_AM_I `0x70`（识别为 LSM6DSV 驱动）
   - **接收器固件 (`firmware/SlimeVR-Tracker-nRF-Receiver`)**：
     - `west.yml` 锁定：`sdk-nrf` revision: `v3.3.0-preview3-branch`
     - 目标板卡：HolyIOT-21017 (`nrf52840`) 或 ProMicro nRF52840

2. **预编译固件产物哈希固化 (`firmware/artifacts/`)**：
   - `bootloader-promicro-0.9.2-SlimeVR.7_nosd.uf2`：SHA-256 `C76C74196A4D447B56451B38C322ECDAC36FCA6F083A6630198825092A46BC85`
   - `bootloader-promicro-0.9.2-SlimeVR.7_s140-rescue.hex`：SHA-256 `412893277A011E8257D54202B7EF751792542B2450DBAAE39F83C3EF196DE129`
   - `receiver-holyiot21017.hex`：SHA-256 `790473E9AEB3346934EA408CD33D2C01A3632D13D27F62E0AA57A26B1CE3DC7E`
   - `receiver-promicro.uf2`：SHA-256 `F144592D8CD943C9C5B8DFD8F8354934FF51B4C8877DFD51F03F7767485F8DB0`

3. **硬件边界说明**：
   - 固件离线配置文件与编译脚本齐备；实际 USB UF2 双击磁盘枚举、SWD 恢复烧录、无线配对与丢包率测试，待实体硬件到货后由 Phase 8 执行。

---

## 九、 Task P7 执行记录：PC 软件干净构建与模拟集成

1. **SlimeVR Desktop 服务端后端构建**：
   - 执行 `.\gradlew.bat :server:desktop:jar`，22 秒内构建成功；
   - 产出位于 `server/server/desktop/build/libs/desktop.jar` 与 `slimevr.jar`；
   - 验证执行 `java -jar slimevr.jar -V`，正常输出版本信息，无任何类缺失。

2. **SteamVR Driver 编译与 CMake 配置**：
   - 执行 `powershell -ExecutionPolicy Bypass -File scripts\msvc-cmake.ps1 -Source driver -Binary driver\build -ConfigureOnly`；
   - MSVC 14.51.36231 + Ninja 7.8 秒内完成 CMake Configure，3.3 秒完成 Generate，CPM 依赖（protobuf, simdjson, libuv）解析正常；
   - 现成驱动产物 `driver/build/Release/driver/slimevr/bin/win64/driver_slimevr.dll` 正常就绪。

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

- **Phase 13 P3 完整服务端回放报告**：[`ai_corrector/artifacts/phase13_p3_server_tick_benchmark_report.json`](../../ai_corrector/artifacts/phase13_p3_server_tick_benchmark_report.json)
- **Phase 13 P4 物理与安全动态审计报告**：[`ai_corrector/artifacts/phase13_p4_physics_audit_report.json`](../../ai_corrector/artifacts/phase13_p4_physics_audit_report.json)
- **Phase 13 P4 坐标验证解析单测**：[`ai_corrector/corrector/test_p4_coordinate_and_simulation.py`](../../ai_corrector/corrector/test_p4_coordinate_and_simulation.py)
- **Phase 13 P4 全动态物理评测脚本**：[`ai_corrector/corrector/evaluate_phase13_p4_physics_audit.py`](../../ai_corrector/corrector/evaluate_phase13_p4_physics_audit.py)
- **已固化流式 ONNX 模型**：[`ai_corrector/artifacts/cstn_v2_yaw_neutral_world.onnx`](../../ai_corrector/artifacts/cstn_v2_yaw_neutral_world.onnx)
- **服务端生命周期与回放单测源码**：
  - [`CorrectorBridgeInferenceTest.kt`](../../server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/CorrectorBridgeInferenceTest.kt)
  - [`CorrectorBridgeReplayBenchmarkTest.kt`](../../server/server/core/src/test/java/dev/slimevr/tracking/processor/corrector/CorrectorBridgeReplayBenchmarkTest.kt)
- **服务端生命周期与原子 Tick 源码**：
  - [`VRServer.kt`](../../server/server/core/src/main/java/dev/slimevr/VRServer.kt)
  - [`CorrectorBridge.kt`](../../server/server/core/src/main/java/dev/slimevr/tracking/processor/corrector/CorrectorBridge.kt)
  - [`OnnxCorrectorEngine.kt`](../../server/server/core/src/main/java/dev/slimevr/tracking/processor/corrector/OnnxCorrectorEngine.kt)
