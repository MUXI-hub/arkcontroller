# 更新日志 (Changelog)

本项目严格遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 变更日志规范，并采用 [语义化版本 2.0.0](https://semver.org/lang/zh-CN/)。

## [0.1.1] - 2026-11-21

### 新增 (Added)
- **左摇杆代替十字键导航 (`arkcontroller.input` & `arkcontroller.control`)**：
  - `GestureEngine` 新增带滞回阈值（触发 `0.50` / 回中 `0.30`）与按住连发的 `StickDirection` 离散四向事件，默认 `left_stick_mode="dpad"` 下左摇杆可直接替代十字键完成全流程吸附跳格。
  - 新增按下左摇杆 **`LS`** 一键切换「槽位吸附导航 ↔ 自由虚拟光标」模式，并在系统托盘菜单新增热切换开关。
- **商店展开/收起联动与多界面缩放态补偿 (`arkcontroller.config` & `arkcontroller.control` & `arkcontroller.overlay`)**：
  - 新增 `LayoutViewState`（`normal` / `shop_open` / `unit_selected`）、`ViewportTransform` 与 `ViewLayoutOverride` 模型，支持在展开商店或选取角色导致镜头缩放/平移时自动切换对应透视棋盘与槽位坐标，解决框选错位。
  - 局内切入商店区自动展开商店、切入棋盘或抓取干员准备部署时自动收起商店；支持 **`LT + LB`** 随时手动展开/收起商店、商店区按 **`B`** 收起并退回整备区。
  - 休整期整备区/棋盘按 **`RT + A`** 主动查看角色详情并激活 `unit_selected` 缩放补偿，按 **`B`** 退出详情恢复标准视角；同时自适应底部横向装备栏与侧边竖向装备栏的跨区导航。
  - 可视化校准向导 `CalibrationDialog` 新增「校准视图态」切换（支持分别拖拽校准 `normal` / `shop_open` / `unit_selected` 透视网格）及 `shop_open` / `unit_selected` 视觉识别模板一键采集。

### 修复 (Fixed)
- 修复 `DeployFSM.pick_unit()` / `pick_equip()` 抓取时提前单击源槽位导致诱发游戏“选取角色 -> 界面缩放”并造成拖拽起终点错位的问题（改为默认无点击虚拟锁定，仅在原生快捷键出售前补发单击选中）。
- 修复 Windows Qt6 启动时未设正字号默认字体触发 `QFont::setPointSize: Point size <= 0 (-1)` 警告的问题。
- 修复 `CalibrationDialog` 在 PySide6 下 `QComboBox.currentData()` 对 `str, Enum` 拆箱导致视图态与场景模板切换失效的问题。
- 新增 `ConfigLoader(use_user_override=...)` 隔离开关，确保自动化单元测试不受本机已保存的 `configs/user_override.json` 干扰。

## [0.1.0] - 2026-11-21

### 新增 (Added)
- **项目工程与配置系统 (`arkcontroller.config`)**
  - 初始化 `pyproject.toml` 工程元数据、`arkcontroller` CLI 入口及最小化依赖集（`PySide6`、`mss`、`opencv-python-headless`、`numpy`）。
  - 实现强类型配置模型 `NormalizedPoint`、`NormalizedRect`、`PerspectiveQuad`、`NavNode`、`LinearZone`、`LayoutProfile`、`KeymapConfig` 与字段边界校验。
  - 提供卫戍协议标准 `16:9` 全流程布局配置 `configs/stronghold_layout_16_9.json` 及默认手柄手感与快捷键配置 `configs/default_bindings.json`。
  - 实现 `ConfigLoader`，支持默认配置加载、用户校准覆盖文件 `configs/user_override.json` 深度合并、原子化写入及运行时热重载。
- **Windows 平台权限与多端视口追踪 (`arkcontroller.platform`)**
  - 实现 Per-Monitor DPI V2 感知初始化（`SetProcessDpiAwarenessContext(-4)`）、管理员完整性检测与 UAC 自动提权重启（`uac.py`）。
  - 实现 `WindowTracker` 与 `ViewportRect`，优先匹配官方 PC 端 `Arknights.exe`（`UnityWndClass`），兼容 MuMu 模拟器（`nemuwin`）与雷电模拟器（`TheRender`）渲染子窗口，自动裁剪 `16:10` / `21:9` 黑边并提供归一化坐标与物理像素双向高精度转换。
- **XInput 手柄驱动、手势引擎与键鼠合成 (`arkcontroller.input`)**
  - 纯 `ctypes` 封装 `XInput1_4.dll`（自动回退 `XInput9_1_0.dll` / `xinput1_3.dll`），支持 125Hz 全局后台轮询、径向缩放死区（Radial Scaled Deadzone）与异步定时关停双马达震动反馈。
  - 实现 `GestureEngine` 高层手势解析器，完整支持单击、防误触长按进度反馈、方向键连发、组合键（`LT`/`RT`/`Back` 修饰层及单键误触抑制）、修饰键按住/松开状态通知以及三次方双速曲线亚像素自由光标积分器。
  - 纯 `ctypes` 封装 64 位对齐（`sizeof(INPUT) == 40`）Win32 `SendInput` 合成器 `VirtualHID`，支持多显示器虚拟桌面 `[0, 65535]` 绝对鼠标定位、平滑插值拖拽、四向滑动（`directional_flick`）、水平/垂直滚轮与基于 `MapVirtualKeyW` 的硬件扫描码键盘输入。
- **非侵入式屏幕采样与双层视觉识别 (`arkcontroller.vision`)**
  - 实现 `ScreenCapture`，通过独立守护线程与线程局部 `mss` 实例以 5Hz 异步采样游戏 `16:9` 视口并统一缩放至 `1280x720` BGR，支持 Win32 `PrintWindow` 遮挡回退通道。
  - 实现双层 `SceneDetector`（覆盖 `ACTIVITY_HUB`、`PRE_MATCH_SETUP`、`SQUAD_EDIT`、`BATTLE_PREP`、`MODAL_CHOICE`、`BATTLE_COMBAT`、`SETTLEMENT`、`UNKNOWN_FREE_CURSOR` 八大阶段），结合局部 ROI `TM_CCOEFF_NORMED` 模板匹配、无模板启发式安全回退与连续帧防抖滤波。
  - 实现 `VisionCalibrator`，支持裁剪保存场景标志性 ROI 模板（`.png` + `.json` 元数据）与商店展开状态辅助模板。
- **透视网格、两段式部署状态机与全流程调度中枢 (`arkcontroller.control`)**
  - 实现 `LinearZoneMapper`、`NodeGraphMapper`（含方向锥智能最近邻搜索）与基于 `cv2.getPerspectiveTransform` 的 `PerspectiveBoardMapper` 4 点单应性透视棋盘映射器。
  - 实现 `DeployFSM`，完整支持干员/装备跨区抓取、整备区互换、装备拖拽穿戴、干员棋盘落位后四向定朝向滑动、按 `B` 安全撤回、按 `X` 区分场上/整备快速出售及按 `Y` 拖拽赠予联机队友。
  - 实现 `ControlDispatcher`，覆盖局外主界面/准备/编队/技能模组切换与局内商店运营、极速连买、三选一按住 `LT` 隐藏看棋盘、`LT` 信息与联机速查层、长按 `Start` 防误触开战、后台失焦自动挂起及实时 HUD 状态快照输出。
- **PySide6 透明穿透 HUD、可视化校准、系统托盘与主入口 (`arkcontroller.overlay` & `arkcontroller.main`)**
  - 实现 PRTS 工业科技风 `HUDWindow`，通过 Qt 透明穿透标记与 Win32 `SetWindowLongW`（`WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOPMOST`）双重加固，实时渲染发光多边形焦点框、抓取源动态虚线轨迹、四向定朝向十字轮盘、长按开战进度环与彩色 Xbox 键位提示栏。
  - 实现 `CalibrationDialog` 交互式校准面板，支持鼠标实时拖拽棋盘透视四角并预览 `5x9` 网格、调整商店/整备区/装备栏槽位首尾锚点以及一键截取热重载场景模板。
  - 实现内置矢量 PRTS 图标的 `ControllerTray` 系统托盘与全链路 CLI 主入口 `arkcontroller.main`（支持 `--config-dir`、`--no-overlay`、`--no-elevate`、`--calibrate`、`--dry-run`）。
- **测试与技术文档**
  - 新增 6 个核心单元测试套件覆盖视口映射、手势状态机、`SendInput` 构造、场景识别、透视单应性与部署状态机。
  - 编写完整简体中文 `README.md`、`CHANGELOG.md`、`docs/architecture.md` 与 `docs/gamepad_manual.md`。
