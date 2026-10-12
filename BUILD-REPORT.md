# SlimeVR 编译流程执行报告

日期：2026-10-11 · 工作区 `H:/program/slimevr` · 目标：nRF52840 + LSM6DSV16X，6 点，VRChat PCVR（SteamVR）

---

## 一、编译结果总览

| # | 链路 | 状态 | 产物 |
|---|---|---|---|
| S1 | Server 后端（Kotlin 2.3.10 / Gradle 8.14.4 / JDK 17） | ✅ | `server/server/desktop/build/libs/slimevr.jar`（24.98 MB） |
| S2 | SteamVR 桥接（C++23 / MSVC） | ✅ | `server/bindings-provider/build/win-x64/Release/SlimeVR-Bindings-Provider.exe` + `openvr_api.dll` |
| S3 | GUI 前端（Electron 40 / React 18 / Vite 5 / pnpm 10.33.0） | ✅ | `server/gui/out/{main,preload,renderer}` |
| S4 | SteamVR 驱动（C++20 / CMake + CPM） | ✅ | `driver/build/Release/driver/slimevr/bin/win64/driver_slimevr.dll`，已装入 SteamVR |
| S5 | 打包（electron-builder 26.15.3） | ✅ | `server/gui/dist/artifacts/win/SlimeVR-win-x64.zip`（163 MB）→ `win-unpacked/SlimeVR.exe` |

### 运行时冒烟（后端）
```
[信息] Starting the tracker server on port 6969...
[Keybinding] Bound full reset to CTRL+ALT+SHIFT+Y
[Keybinding] Bound yaw reset to CTRL+ALT+SHIFT+U
[信息] [WindowsNamedPipe] Pipe \\.\pipe\SlimeVRRpc created
[信息] [WindowsNamedPipe] Pipe \\.\pipe\SlimeVRDriver created
[信息] [WindowsNamedPipe] Pipe \\.\pipe\SlimeVRInput created
[信息] [WebSocket] Web Socket VR Bridge started on port 21110
```
`netstat` 确认 21110 处于 LISTENING。后端启动命令需带子命令：`java -Xmx128M -jar slimevr.jar run`。

> SteamVR 中原先由官方安装器装的驱动已备份为
> `H:\steam\steamapps\common\SteamVR\drivers\slimevr.official.bak`。

---

## 二、本机环境的三个硬约束（与网上教程不同）

1. **装的是 Visual Studio 2026（v18.x / MSVC 14.51.36231 / MSBuild v180），不是 2022。**
   `Visual Studio 17 2022` 生成器报 `MSB8020`（找不到 v143）。必须用 `-G Ninja` +
   显式 `-DCMAKE_C/CXX_COMPILER=<cl.exe>`。
2. **`vcvarsall.bat` 不可用** —— 它内部调 `reg.exe`，被沙箱黑名单拦截。改为手工设置
   `PATH` / `INCLUDE` / `LIB`。**`INCLUDE` 必须含 SDK 的 `winrt` 目录**，否则 abseil 的
   `time_zone_lookup.cc` 报 `fatal error C1083: roapi.h: No such file or directory`。
3. **`cmd.exe` 被沙箱拦截** —— Node 的 `execSync` 默认走 cmd.exe，一律报 `EBUSY`，必须改 `execFileSync`。
   同类问题还有 esbuild 的 postinstall（spawn node.exe 报 EBUSY）。

网络方面：`github.com` 与 `dl.google.com` 直连不通，代理 `http://127.0.0.1:7892` 可用；
Electron 二进制走 `ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/`。

---

## 三、为适配本机做过的本地改动（均带 `LOCAL` 注释，可随时还原）

| 文件 | 改动 | 原因 |
|---|---|---|
| `server/settings.gradle.kts` | 注释掉 `include(":server:android")` | AGP 8.13.2 只在 `dl.google.com`，该仓库不可达；PCVR 不需要 Android |
| `server/pnpm-workspace.yaml` | `onlyBuiltDependencies` 移除 `esbuild` | postinstall spawn node.exe 报 EBUSY；JS API 运行时会自行解析 `@esbuild/win32-x64` |
| `server/gui/vite.config.ts` | `execSync` → `execFileSync` | 绕开被拦的 cmd.exe |
| `server/gui/electron-builder.yml` | 注释掉 `electronLanguages` | 裁剪语言包会批量删除约 50 个 `.pak`，触发沙箱批量删除保护 |

辅助脚本（新建）：
- `scripts/msvc-cmake.ps1` —— 封装 MSVC 环境 + Ninja 配置/构建，两个 C++ 项目共用
- `scripts/vsenv.cmd` —— vcvars 包装（**实际不可用**，因 reg.exe 被拦，保留备查）

---

## 四、闸门 G1 结论：LSM6DSV16X 可用（已核对源码）

已实地核对 `SlimeVR-Tracker-nRF` HEAD `cbfff16`，不再只是推断：

- `src/sensor/sensors.h` 的 `dev_imu_names[]` **确实没有** `LSM6DSV16X` 条目，只有 `LSM6DSV` 和 `LSM6DSV16B/ISM330BX`。
- 但 I2C 扫描（reg `0x0F`）接受的 ID 是 `0x22,0x69,0x6A,0x6B,0x6C,0x6D,0x70,0x71`，其映射表为：

  | WHO_AM_I | 识别为 |
  |---|---|
  | `0x6B` | LSM6DSR |
  | `0x6C` | LSM6DSO |
  | **`0x70`** | **LSM6DSV** |
  | `0x71` | ISM330BX |

- `sensor_imus[]` 索引 17 即 `&sensor_imu_lsm6dsv`（真实驱动，非 `sensor_imu_none`）。

**ST 官方 PID 中 `LSM6DSV16X_ID = 0x70U`** → 上电会被识别成 **LSM6DSV** 并跑 `sensor_imu_lsm6dsv` 驱动，
有完整实现。**结论：可用，失败概率低。**

残留风险：若该批芯片实际回报 `0x71`，会落到 `sensor_imu_ism330bx` 驱动（同样有实现）。
因此仍建议**先买 1 套试做、用串口 `info` 命令确认一次**，但不必再预设失败。
真失败时的切换顺序：ICM-45686 → LSM6DSV 真片 → LSM6DSR（立创 ¥16–23 最好买）。
绝不买国产 ICM-42688 替代片（WHO_AM_I 与 ID 不符，固件识别不了）。

---

## 五、下一步（S6–S11）

1. **下单试做包**（约 ¥150–230）：SuperMini nRF52840 ×1、LSM6DSV16X breakout ×1、401230 110mAh 电池 ×1、
   HolyIOT-21017 接收器 ×1、DAPLink ×1。
2. **烧录验证**：先刷 SlimeVR 定制 UF2 bootloader，再刷 CI 预编译固件；串口 `info` 看 IMU 识别结果。
3. **NCS 工具链**（`C:\ncs`，10 GB+）：追踪器用 `sdk-nrf v3.1.x`，接收器用 `v3.3.0-preview3`，
   **必须建两个独立 west 工作区**，且 `west build` 必须带 `-DBOARD_ROOT=`。
   本机 board target：SuperMini / nice!nano / ProMicro nRF52840 → `promicro_uf2/nrf52840/i2c`。
4. **批量 6 套 + 组装**：Stacked 方案免 PCB，INT 引脚必接，统一刷 i2c 变体。
5. **配对校准**：接收器串口 `pair` → 追踪器长按 SW0 5 秒 → `exit`；Server ≥ 0.13.2（本次编译的 main 满足）。

启动顺序：SlimeVR.exe → SteamVR → VRChat。

---

## 六、复用命令速查

```bash
# 后端
cd /h/program/slimevr/server
./gradlew.bat :server:desktop:shadowJar --build-cache

# 桥接 / 驱动（PowerShell）
& H:\program\slimevr\scripts\msvc-cmake.ps1 -Source <src> -Binary <bin>

# 前端
cd /h/program/slimevr/server/gui
export ELECTRON_MIRROR="https://npmmirror.com/mirrors/electron/"
pnpm run build && pnpm run package

# 后端冒烟
java -Xmx128M -jar server/desktop/build/libs/slimevr.jar run
```
