# ArkController 系统架构与核心算法说明

本文档详细阐述 **ArkController** 的模块分层、多线程与事件循环拓扑、16:9 视口与 4 点单应性透视变换算法、零注入安全边界机制，以及核心状态机设计。

---

## 1. 整体分层架构

`ArkController` 采用严格的外部非侵入式分层架构，由 **配置与平台层**、**输入与键鼠合成层**、**非侵入视觉识别层**、**核心操控调度层** 与 **PySide6 悬浮交互层** 五大子系统组成。

```mermaid
flowchart TD
    subgraph Hardware["硬件与外部进程"]
        GP["Xbox 手柄 (XInput1_4.dll / 震动反馈)"]
        GAME["明日方舟进程 (Arknights.exe 或模拟器窗口)"]
    end

    subgraph PlatformConfig["配置与平台层"]
        CFG["ConfigLoader (默认 JSON + user_override.json 合并)"]
        UAC["UAC & DPI V2 (完整性提权 / Per-Monitor V2)"]
        WT["WindowTracker (HWND 追踪与 16:9 黑边裁剪)"]
    end

    subgraph InputLayer["手柄解析与虚拟键鼠层"]
        XI["XInputDriver (125Hz 轮询 / 径向缩放死区)"]
        GE["GestureEngine (Tap / Hold / Repeat / Chord / 光标积分)"]
        HID["VirtualHID (64位 SendInput 扫描码键盘与平滑鼠标)"]
    end

    subgraph VisionLayer["异步视觉感知层"]
        CAP["ScreenCapture (5Hz 守护线程 mss 自愈 + PrintWindow 过滤)"]
        SD["SceneDetector (5 项局内场景 / 确认弹窗 / 4-6 槽商店识别)"]
        CAL["VisionCalibrator (局内 ROI 模板裁剪保存器)"]
    end

    subgraph ControlLayer["核心调度与状态机层"]
        GM["GridMapper (线性槽位 / 对称羁绊 / 4点单应性棋盘)"]
        DFSM["DeployFSM (抓人 / 原地回环转向 / 穿装 / 策略拖拽)"]
        DISP["ControlDispatcher (局外自由鼠标 / 局内焦点路由)"]
    end

    subgraph OverlayLayer["PySide6 纯白极简悬浮与桌面主控层"]
        HUD["HUDWindow (H1/H2 极简穿透叠加层 / 矢量手柄图示 / WDA_EXCLUDEFROMCAPTURE)"]
        DASH["PRTSDashboardWindow (方案B v2 四分区主控台 / 16Hz 遥测仪 / 参数热保存)"]
        CALUI["CalibrationDialog (白底细边校准栏 / 透视网格与模板采样器)"]
        TRAY["ControllerTray (白底琥珀点矢量图标 / 浅色高对比托盘菜单)"]
    end

    GP --> XI
    XI --> GE
    GAME -. "客户区物理坐标与前台状态" .-> WT
    GAME -. "5Hz 非侵入画面采样" .-> CAP
    CFG --> GE
    CFG --> DISP
    UAC --> WT
    WT --> CAP
    WT --> DISP
    WT --> HUD
    CAP --> SD
    SD --> DISP
    GE --> DISP
    DISP <--> GM
    DISP <--> DFSM
    DISP --> HID
    DFSM --> HID
    HID == "管理员级 SendInput 键鼠输入" ==> GAME
    DISP --> HUD
    CALUI --> CAL
    CAL --> SD
    CALUI --> CFG
    TRAY --> DISP
    TRAY --> HUD
    TRAY --> DASH
    DASH --> CFG
    DASH --> DISP
    DASH --> CALUI
```

### 多频时钟协同调度（`main.py`）

为避免截屏、OpenCV 识别与桌面遥测刷新阻塞手柄操作手感，主进程通过解耦时钟协同运行：

| 调度单元 | 运行载体 | 周期 / 频率 | 核心职责 |
| :--- | :--- | :--- | :--- |
| `ScreenCapture` 采图线程 | 独立后台守护线程 | `~200ms` (5Hz) | 线程独占 `mss` 截取 16:9 视口下采样至 `1280x720` BGR，支持 UAC 切换后惰性重建自愈并拦截全黑伪帧 |
| `_vision_timer` | Qt 主线程 `QTimer` | `200ms` (5Hz) | 读取最新帧并执行 `SceneDetector.detect()` 更新稳定场景 |
| `_window_timer` | Qt 主线程 `QTimer` | `250ms` (4Hz) | 追踪游戏窗口句柄、位置尺寸及前后台切换，刷新托盘与主控台状态 |
| `_input_timer` | Qt 主线程 `QTimer` | `8ms` (~125Hz) | 轮询 XInput、驱动 `GestureEngine` 与 `ControlDispatcher`、刷新 HUD |
| `PRTSDashboardWindow` 遥测定时器 | Qt 主线程 `QTimer` | `60ms` (~16Hz) | 仅在主控台窗口可见时运行，刷新双摇杆靶盘、扳机条与按键阵列；关闭最小化至托盘后自动停止，后台零开销 |

---

## 2. 16:9 视口归一化与 DPI V2 坐标解算

### 2.1 高分屏 Per-Monitor DPI V2 一致性

在 Windows 125% / 150% / 200% 缩放环境下，如果未开启 DPI V2 感知，Win32 窗口坐标与 `SendInput` 绝对鼠标坐标将出现错位偏移。系统在创建 `QApplication` 之前调用 `SetProcessDpiAwarenessContext(-4)`：

- `WindowTracker` 获取的客户区坐标 `(client_left, client_top, client_width, client_height)` 全部为**真实物理像素**；
- `VirtualHID` 查询虚拟桌面 `SM_XVIRTUALSCREEN` / `SM_YVIRTUALSCREEN` / `SM_CXVIRTUALSCREEN` / `SM_CYVIRTUALSCREEN` 物理边界，并将目标物理像素 `(x, y)` 线性映射至 `[0, 65535]` 发送给 `SendInput(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK)`；
- `HUDWindow` 通过 `align_widget_to_physical_rect` 结合 `devicePixelRatio` 与 Win32 `SetWindowPos` 将透明窗口严丝合缝贴合至游戏 16:9 物理视口。

### 2.2 异形屏黑边裁剪与锚点模式

设游戏客户区宽高为 $(W_c, H_c)$，目标标准比例 $R = 16 / 9$：

- **宽屏模式**（$W_c / H_c > 16 / 9$，例如 `2560x1080`、`3440x1440`）：
  $$W_a = \text{round}(H_c \cdot 16 / 9), \quad H_a = H_c, \quad L_a = L_c + \lfloor(W_c - W_a) / 2\rfloor, \quad T_a = T_c$$
- **高屏模式**（$W_c / H_c < 16 / 9$，例如 `2560x1600` 16:10 笔记本）：
  $$W_a = W_c, \quad H_a = \text{round}(W_c \cdot 9 / 16), \quad L_a = L_c, \quad T_a = T_c + \lfloor(H_c - H_a) / 2\rfloor$$

对于归一化点 $(u, v) \in [0, 1]^2$：
- `center_16_9`（默认）：映射到 16:9 核心视口 $(L_a + u \cdot W_a,\; T_a + v \cdot H_a)$；
- `left_edge` / `right_edge` / `top_edge` / `bottom_edge` / `client`：支持在 21:9 或 16:10 屏幕上将贴边 UI 按钮精准锚定到真实客户区边缘。

---

## 3. 4 点单应性透视棋盘变换算法

《明日方舟》卫戍协议战场采用倾斜 3D 透视视角，棋盘在 2D 屏幕上呈现**上窄下宽的等腰/微斜梯形**，若使用简单均匀网格划分，远端行与边角列会产生明显点击偏差。

`PerspectiveBoardMapper` 采用计算机视觉中的 **2D 射影单应性变换（Homography）**：

1. **建立对应点集**：将理想单位棋盘正方形四角
   $$P_{\text{unit}} = \{(0, 0), (1, 0), (1, 1), (0, 1)\}$$
   与校准后的游戏棋盘四角归一化坐标
   $$P_{\text{quad}} = \{p_{\text{TL}}, p_{\text{TR}}, p_{\text{BR}}, p_{\text{BL}}\}$$
   通过 `cv2.getPerspectiveTransform(src, dst)` 求解唯一可逆的 $3 \times 3$ 单应性矩阵 $H$ 及逆矩阵 $H^{-1}$。
2. **正向格心与格框投影**：对于 $R \times C$ 棋盘（实机标准默认 $4 \times 7$）上的任意格 $(r, c)$：
   - 格心在单位平面为 $(s, t) = \left(\frac{c + 0.5}{C},\; \frac{r + 0.5}{R}\right)$；
   - 经齐次投影：
     $$\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = H \begin{bmatrix} s \\ t \\ 1 \end{bmatrix}, \quad (u, v) = \left(\frac{x'}{w'},\; \frac{y'}{w'}\right)$$
   - 同理对四个角 $\left(\frac{c}{C}, \frac{r}{R}\right)$ 等投影即可获得与 3D 透视完全贴合的四边形多边形 `cell_polygon(r, c)`，直接供 HUD 渲染透视高亮框。
3. **逆向光标拾取**：给定任意屏幕归一化坐标 $(u, v)$，经 $H^{-1}$ 反投影回单位正方形 $(s, t)$，即可 $O(1)$ 定位所在格 `(row, col)` 或在自由光标切回吸附模式时无缝定位最近地块。
4. **多视图态镜头缩放补偿与动态商店左延**：
   - `LayoutProfile` 支持 `LayoutViewState`（`normal` / `shop_open` / `unit_selected`）视图覆写与 `ViewportTransform` 仿射缩放平移补偿（经 OpenCV SIFT 实机配准，`shop_open` 下同步缩放棋盘、整备区与装备栏，中位误差 `1.1px`）；
   - `resolve_shop_zone` 与 `resolve_shop_upgrade_point` 支持商店升至二本（5 槽）与四本（6 槽）时以最右卡槽为锚点按固定步距向左动态延伸，并同步左移升级灰卡点击坐标。

---

## 4. 零注入安全边界与穿透设计

| 安全与稳定维度 | 实现机制 |
| :--- | :--- |
| **零内存/零注入** | 严禁调用 `OpenProcess` 读写内存、DLL 注入或 Hook 游戏进程；仅使用公开 Win32 输入与窗口查询 API |
| **UIPI 权限合规** | 通过标准 UAC 提权与官方 PC 客户端（`Arknights.exe`）对齐完整性级别，避免任何绕过安全检查的行为 |
| **64 位 SendInput 对齐** | `VirtualHID` 在 `ctypes.Union` 中完整定义 `MOUSEINPUT`、`KEYBDINPUT`、`HARDWAREINPUT`，确保 64 位 Windows 下 `ctypes.sizeof(INPUT) == 40`，并使用 `MapVirtualKeyW` 发送硬件扫描码（`KEYEVENTF_SCANCODE`） |
| **HUD 绝对不抢焦点 & 防截屏污染** | `HUDWindow` 同时启用 Qt `WindowTransparentForInput` + `WA_ShowWithoutActivating`、Win32 `SetWindowLongW(GWL_EXSTYLE, WS_EX_LAYERED \| WS_EX_TRANSPARENT \| WS_EX_NOACTIVATE \| WS_EX_TOPMOST)` 以及 `SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)`，确保所有点击 100% 穿透至游戏窗口，且后台 `mss` / DWM 截屏自动忽略透明 HUD 图层，杜绝青色选框干扰视觉识别 |
| **失焦自动挂起** | `ControlDispatcher` 在每帧核对 `viewport is not None and viewport.is_foreground`，游戏切出前台或暂停时立刻隐藏 HUD 并停止所有键鼠模拟 |

---

## 5. 异步采图自愈、双层场景识别与防抖流水线

### 5.1 后台截屏 UAC 自愈与黑帧防护（`ScreenCapture`）

Windows UAC 提权切回安全桌面瞬间可能导致后台线程首次初始化 `mss.mss()` 失败或已有 GDI 句柄失效，而 Unity DX11 渲染窗口下 `PrintWindow` 常返回全黑帧。`ScreenCapture` 为此建立双重防护：
- **惰性自愈**：后台 `_capture_loop` 每轮检测到 `sct is None` 时自动重试 `mss.mss()`；运行中若截屏抛异常，立即释放并置空失效 `sct`，由下一周期自动重建全新实例；
- **全黑伪帧拒绝**：当 `PrintWindow` 回退返回全黑帧（`max() == 0`）且 `mss` 也不可用时安全返回 `None`，绝不向识别器喂入假黑屏画面。

### 5.2 精简五场景双层判别器（`SceneDetector`）

自 `v0.2.0` 起，系统彻底移除易受版本改版影响的局外菜单场景与静态节点配置，`GameScene` 严格精简为 5 项局内核心状态：**`BATTLE_PREP`**、**`MODAL_CHOICE`**、**`MODAL_CONFIRM`**、**`BATTLE_COMBAT`**、**`UNKNOWN_FREE_CURSOR`**（所有局外与设置页均归入 `UNKNOWN_FREE_CURSOR`，自动由左摇杆平滑驱动自由模拟鼠标）。

`SceneDetector` 采用**“模板优先 + 零模板强特征启发式 + 连续帧防抖”**三级流水线：

1. **局部 ROI 模板匹配层**：扫描 `assets/templates/` 中的 `<scene>.png` 与配套 `.json`（含 `roi_norm`、`threshold`、`search_padding`），在限定局部窗口内运行 `cv2.matchTemplate(..., cv2.TM_CCOEFF_NORMED)`（默认阈值 `0.82`），并过滤零方差纯色图像避免伪高分。
2. **零模板 HSV 强特征与几何启发式层**：
   - **局外 100% 防误判双锚点门禁**：`BATTLE_PREP` / `BATTLE_COMBAT` 启发式分支强制校验 `_has_stronghold_in_game_chrome()`（左上深红退出门 + 顶部中央深黑状态栏双锚点同时命中），并将右上准备按钮色相收窄至真实粉红区间 `(H >= 158) | (H <= 4)`，同时要求所有弹窗分支满足 `modal_chrome`（顶部青色标题带或右侧方形收起按钮），确保主界面、活动横幅、战前编队、干员列表与基建画面 100% 保持在 `UNKNOWN_FREE_CURSOR`；
   - **回合结束 & `ESC` 退出确认弹窗（`MODAL_CONFIRM`）**：优先于右上角准备按钮检测，采用黑底分位数约束（`dark_ratio >= 0.82 & p80 < 30`）、左下纯灰取消带与避开右下机械剪影的暗红确认带联合判定，并通过中央上方亮像素占比输出 `confirm_has_dont_ask` 自动区分三项回合确认窗与双项 `ESC` 暂时离开窗；
   - **自适应决策弹窗（`MODAL_CHOICE`：`1x2` / `1x3` / `2x3`）**：基于 Sobel 垂直卡片边框梯度与白色「确认选择」底条检测，按 `6 选一（支持官方 PC 16:9 与旧宽屏双布局，单槽兼容未选暗卡与 1~5 张已被队友选走的均匀浅灰卡） -> 3 选一 -> 2 选一` 顺序零模板区分并实时追踪 `selected_choice_index`；二选一两翼外围检测采用按列垂直均值再取列峰（`mean(axis=0).max()`）免疫压暗雪山纹理背景；
   - **休整运营、0 资金商店与左下 `2x3` 表情面板（`BATTLE_PREP`）**：支持右上角粉红准备按钮或联机 `LOADING...` 顶部「休息一下」紫框 + 放大镜双路径判定；商店展开与槽数检测彻底脱离 `refresh_yellow` 资金颜色依赖，升级为 `(freeze_blue >= 0.45) or (refresh_yellow >= 0.25 and freeze_blue >= 0.25)`，在 0 资金刷新变灰后仍稳判 `shop_open=True` 与 `4/5/6` 槽；`±12px` 峰值搜索兼容紫红 `[G]` 升级卡与已售空 `NO INFO` 空槽；非商店/非详情态下同步检测左下 `2x3` 表情面板（`emote_panel_open`）、顶部居中对称羁绊数量（`0~9`）及多人左侧队友头像。
3. **连续帧防抖与不打断保护**：连续 `debounce_frames = 2` 帧一致后切换稳定场景；干员/装备/策略抓取与定朝向期间不强行切层。即便用户通过 `Back` 键处于**手动锁层**，只要同为 `BATTLE_PREP`、`MODAL_CHOICE` 或 `MODAL_CONFIRM`，`ControlDispatcher` 依然实时同步 `shop_open`、`shop_slot_count`、`unit_selected`、`emote_panel_open`、`covenant_count`、`has_teammates`、`choice_count`、`selected_choice_index` 与 `confirm_has_dont_ask`；当商店收起时自动将空闲 `FocusZone.SHOP` 退回 `FocusZone.BENCH` 防止残留选框。

---

## 6. 核心交互状态机详解

### 6.1 干员/装备/策略抓取、原地回环转向与两段式定朝向（`DeployFSM`）

```mermaid
stateDiagram-v2
    [*] --> IDLE: 休整期浏览槽位 (商店 / 整备区 / 棋盘 / 装备栏 / 策略头像)

    IDLE --> HOLDING_UNIT: 在整备区或棋盘干员格按 A 锁定抓取源
    IDLE --> HOLDING_EQUIP: 在装备栏槽位按 A 锁定装备源
    IDLE --> HOLDING_STRATEGY: 在右侧策略头像按 A 锁定策略源

    HOLDING_UNIT --> IDLE: 按 B 取消抓取
    HOLDING_UNIT --> IDLE: 移至另一整备格按 A (平滑拖拽互换)
    HOLDING_UNIT --> IDLE: 按 X 快速出售 (原生快捷键 X 或拖至右下售出区)
    HOLDING_UNIT --> IDLE: 按 Y 拖拽赠予联机队友头像
    HOLDING_UNIT --> AIMING_DIRECTION: 移至棋盘新格按 A (平滑拖拽落位) 或棋盘同格再按 A (smooth_drag_loopback 原地回环拖拽)

    AIMING_DIRECTION --> IDLE: 推摇杆或十字键四向滑动定朝向并微震动
    AIMING_DIRECTION --> IDLE: 按 B 左键点击菱形左上角取消按钮撤回

    HOLDING_EQUIP --> IDLE: 移至装备他槽(挪动) 或干员格(穿戴) 按 A
    HOLDING_EQUIP --> IDLE: 按 B 取消或按 X 出售装备

    HOLDING_STRATEGY --> IDLE: 原地再按 A 点击释放 或移至棋盘/整备干员/商店卡槽按 A 平滑拖拽释放
    HOLDING_STRATEGY --> IDLE: 按 B 无点击安全取消
```

### 6.2 组合键抑制与防误触机制（`GestureEngine`）

- **径向缩放死区（Radial Scaled Deadzone）**：对摇杆二维矢量模长 $r = \sqrt{x^2 + y^2}$ 应用平滑死区重映射，消除十字轴吸附伪影与漂移；
- **三次方双速光标积分与物理鼠标无损同步**：$\Delta p = \text{sign}(v) \cdot |v|^{\gamma} \cdot V_{\max} \cdot \Delta t$（默认 $\gamma = 2.2, V_{\max} = 1400\text{ px/s}$），配合亚像素余量累加器实现极低速 1 像素级精调与满推高速跨屏；自由鼠标模式下通过 `VirtualHID.get_cursor_pos()` 与 `>= 2px` 变化门槛同步窗口内真实物理鼠标位置（启动/重回前台/手移物理鼠标立即跟手，同时不抹零 `< 0.5px/tick` 慢速微推摇杆的亚像素累加，副屏越界自动免疫）；
- **组合键消耗标记**：当 `LT`、`RT`、`Back` 与另一按键组成 Chord（如 `LT + X`、`Back + Start`）并触发后，本次按下周期自动屏蔽修饰键与子键松开时的单独 Tap/Hold 误触发；
- **长按防误触进度反馈**：休整期按住 `Start` 时实时向 HUD 推送 `ButtonHoldProgress(0.0 ~ 1.0)` 渲染进度环，达到 `ready_hold_ms`（默认 `350ms`）才触发准备开战，中途松开立即归零取消。

---

## 7. 目录结构速览

```text
arkcontroller/
├── configs/
│   ├── default_bindings.json        # 手柄手感、防误触阈值、原生快捷键与按键映射预设
│   ├── stronghold_layout_16_9.json  # 卫戍协议局内 16:9 归一化锚点、策略/确认窗与 4x7 透视棋盘
│   └── user_override.json           # (运行校准后自动生成) 用户个性化覆盖配置
├── assets/templates/                # 用户采集的局部 ROI 场景识别模板 (.png + .json)
├── src/arkcontroller/
│   ├── config/                      # 强类型模型 (models.py) 与配置加载器 (loader.py)
│   ├── platform/                    # UAC / DPI V2 (uac.py) 与窗口视口追踪 (window_tracker.py)
│   ├── input/                       # XInput 驱动、手势引擎与 Win32 SendInput 合成器
│   ├── vision/                      # 异步截屏、双层场景识别与模板校准器
│   ├── control/                     # 透视网格映射、两段式部署状态机与调度中枢
│   ├── overlay/                     # 纯白极简穿透 HUD (含通用手柄矢量图示)、方案B v2 桌面主控台、校准向导与托盘
│   └── main.py                      # CLI 入口（含 --minimized）与多频事件循环装配
├── tests/                           # 7 个核心模块单元测试套件（含离屏 UI 测试）
└── docs/                            # 架构设计与手柄操作手册
```
