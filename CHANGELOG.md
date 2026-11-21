# 更新日志 (Changelog)

本项目严格遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 变更日志规范，并采用 [语义化版本 2.0.0](https://semver.org/lang/zh-CN/)。

## [0.2.2] - 2026-11-21

### 新增与修复 (Added & Fixed)
- **商店防误触「连按两次 `A` 购买」+「`RT + A` 查看商店干员详情」(`arkcontroller.control`)**：
  - 在 `FocusZone.SHOP` 空闲态下，第 1 次按 **`A`** 仅本地进入 2.0 秒预选确认态（不发送鼠标单击，避免诱发角色详情抢走第 2 次按键），HUD 状态栏提示 `商店 [i/N] · 再按A购买`；有效期内同槽第 2 次按 **`A`** 执行双击完成购买；预选态按 **`B`** 仅取消预选而不收起商店，换槽/切区/商店收起/超时自动清预选。
  - 在 `FocusZone.SHOP` 按 **`RT + A`** 单击展开该商店干员详情（屏蔽误触发 `X` 出售与 `Y` 撤回并在 HUD 隐藏这两项提示），按 **`B`** 点击右侧安全空白区退出详情并精准回到原商店卡槽。
- **左下「交流」按钮与 `2x3` 表情面板全交互 + 滚轮翻页 (`arkcontroller.config` & `arkcontroller.vision` & `arkcontroller.control`)**：
  - 移除官方 PC 客户端无效的 `"emote_wheel": "z"` 快捷键，实机校准左下「交流」按钮 `(0.0684, 0.9181, left_edge)`、面板滚轮中心 `emote_panel_center (0.1290, 0.7400, left_edge)` 与 `emote_top_zone` / `emote_bottom_zone`（`2x3` 六枚表情槽）。
  - 新增 `SceneDetectionResult.emote_panel_open` 自动识别与 `FocusZone.EMOTE_PANEL`：通过 `LT + D-Pad 下`（或在队友栏底部继续按下）呼出时，若商店展开则先自动收起商店再点击「交流」按钮；面板内支持方向键 `2x3` 选表情（指针跟随）、推右摇杆 **`RS`** 自动移至面板中心滚动翻页、`LB`/`RB` 快捷翻页、**`A`** 发送、**`B`** 关闭面板回归原区。
- **多人六选一（`2x3`）PC 16:9 校准、含已被选走浅灰卡识别与手动锁层同步 (`arkcontroller.config` & `arkcontroller.vision` & `arkcontroller.control`)**：
  - 实机校准官方 PC 16:9 六选一三列对齐布局 `choice_6_top_zone (v=0.2931)` / `choice_6_bottom_zone (v=0.7083)` 与 `choice_toggle_eye_6 (0.968, 0.500, right_edge)`；`SceneDetector` 同时兼容新 PC 16:9 与旧宽屏双布局，单槽接受「未选暗底卡」或「已被队友选走均匀浅灰卡」（支持 1~5 张灰卡，全 6 灰安全拒识）；同步修复手动锁层 `MODAL_CHOICE` 下实时同步 `choice_count` 与 `selected_choice_index`。
- **修复 0 资金刷新变灰商店识别错乱、联机 `LOADING...` 休整期识别与收起防残框 (`arkcontroller.vision` & `arkcontroller.control`)**：
  - `_detect_shop_open_heuristic` 与 `_detect_shop_slot_count_heuristic` 彻底脱离 `refresh_yellow >= 0.25` 资金颜色依赖，升级为 `(freeze_blue >= 0.45) or (refresh_yellow >= 0.25 and freeze_blue >= 0.25)`，0 资金刷新变灰后仍精准保持 `shop_open=True` 与三区缩放补偿；
  - 联机等待阶段无右上粉红准备按钮时，通过局内双锚点 + 顶部「休息一下」紫框与左右放大镜精准识别 `BATTLE_PREP`（`shop_open=False`），并统一 `UNKNOWN_FREE_CURSOR` 默认 `shop_open=False`；商店收起时若焦点仍停留在空闲 `FocusZone.SHOP`，自动退回 `FocusZone.BENCH` 防止残留商店选框。
- **局内 `ESC` 暂时离开确认弹窗抗剪影识别与无「本局不再提示」自动精简 (`arkcontroller.vision` & `arkcontroller.control`)**：
  - 加固 `_detect_modal_confirm_heuristic`（采用黑底分位数约束并避开右下红栏右端机械剪影），杜绝未压缩实机帧漏检；新增 `confirm_has_dont_ask` 自动区分回合结束确认窗（3 项）与 `ESC` 退出确认窗（2 项，无「本局不再提示」时自动隐藏 `X` 提示并限定仅在取消/确认间切换）。
- **透明 HUD 启用 `WDA_EXCLUDEFROMCAPTURE` 防截屏污染 (`arkcontroller.overlay`)**：
  - 在 `apply_win32_click_through()` 中调用 `SetWindowDisplayAffinity(hwnd, 0x00000011)`（异常安全降级），玩家肉眼正常可见透明 HUD，而后台 `mss` 截屏自动穿透忽略 HUD 选框与准星，从根源消除 HUD 绘制物干扰视觉识别。

## [0.2.1] - 2026-11-21

### 新增与修复 (Added & Fixed)
- **决策弹窗新增「二选一（`1x2`）」全链路适配 (`arkcontroller.config` & `arkcontroller.vision` & `arkcontroller.control`)**：
  - 实机像素测绘并在 `configs/stronghold_layout_16_9.json` 与 `LayoutProfile` 新增 `choice_2_zone`（`start_u=0.3809, end_u=0.6477, v=0.5000, slot_w=0.2188, slot_h=0.3306`），`resolve_choice_card_center/size/confirm/polygon/toggle` 在 `choice_count == 2` 时精准适配二选一卡片、底部白色「确认选择」横条及右侧「收起」按钮（并兼容旧 JSON 缺省回退）。
  - `SceneDetector._detect_real_modal_choice` 新增 `(C) 1x2` 二选一识别与底部白条选中索引感知；外围无第 3 卡判据采用按列垂直均值再取列峰（`mean(axis=0).max()`），免疫两翼压暗雪山/砖块斜边纹理干扰；同步强制二/三/六选一均需命中卫戍弹窗顶部青色标题/收起按钮特征。
  - `ControlDispatcher` 更新决策弹窗焦点文案与导航，精准输出 `二选一 [1/2]` / `二选一 [2/2] · 待确认`，完整支持左右/`LB`/`RB` 循环切换、初始白条吸附、两段式 `A` / `RT+A` 确认、`B` 退回与按住 `LT` 收起看场。
- **启动默认自由鼠标与局外界面 100% 防误判加固 (`arkcontroller.vision`)**：
  - 新增 `_has_stronghold_in_game_chrome()` 局内双锚点门禁（左上深红退出门 + 顶部中央深黑状态栏），并将右上角准备按钮检测色相收窄至真实粉红区间 `(H >= 158) | (H <= 4)`，同时为亮卡弹窗回退补齐 `modal_chrome` 约束，彻底杜绝主界面、活动横幅、战前编队、干员列表与基建等局外画面误判切入局内吸附模式。
- **自由鼠标物理光标无损同步与 HUD 顶部胶囊去重 (`arkcontroller.input` & `arkcontroller.control` & `arkcontroller.overlay`)**：
  - `VirtualHID` 新增 `get_cursor_pos()`（Win32 `GetCursorPos` 8 字节有符号 `POINT`，支持多屏负坐标，失败安全降级返回 `None`）。
  - `ControlDispatcher` 在启动、窗口重新激活及自由鼠标模式下基于 `>= 2px` 变化门槛无损同步游戏客户区内真实物理光标位置（防止 `< 0.5px/tick` 微推左摇杆亚像素累加被整数回读抹零，副屏越界自动免疫；局内按 `LS` 切自由光标立即将真实指针落至当前吸附槽位）。
  - `HUDWindow._format_top_status_text()` 清理同义重复段，`UNKNOWN_FREE_CURSOR` 场景清爽显示 `PRTS // 自由鼠标`，局内按 `LS` 切自由光标去重显示 `PRTS // 休整运营 · 自由光标`。

## [0.2.0] - 2026-11-21

### 新增、变更与修复 (Added, Changed & Fixed)
- **棋盘干员原地设置转向（回环拖拽状态机）(`arkcontroller.input.virtual_hid` & `arkcontroller.control`)**：
  - `VirtualHID` 新增 `smooth_drag_loopback(origin, offset, ...)` 往返平滑拖拽原语：在原格悬停按下左键 → 平滑拖离跨越 Unity `pixelDragThreshold` → 平滑拖回原格松开，稳定呼出游戏四向定朝向盘。
  - `DeployFSM.place_on_board()` 在源格与目标格为同一棋盘格时不再直接取消，而是执行自动方向回环拖拽（最右列向左偏防越界）并转入 `AIMING_DIRECTION` 态，支持推摇杆/十字键原地重置干员朝向或按 **`B`** 取消。
- **回合结束未用完资金 / 策略确认弹窗全自动识别与交互 (`arkcontroller.vision` & `arkcontroller.config` & `arkcontroller.control` & `arkcontroller.overlay`)**：
  - 新增 `GameScene.MODAL_CONFIRM` 与 `FocusZone.CONFIRM_DIALOG`，零模板精准识别全宽暗底 + 左下灰「取消」栏 + 右下暗红「确认」栏，且优先级高于右上角准备按钮检测。
  - 实机标定 `confirm_no (0.2496, 0.6729)`、`confirm_yes (0.7496, 0.6729)`、`confirm_dont_ask (0.4992, 0.5597)` 三处锚点；进入弹窗默认聚焦「确认」，支持方向键 / `LB`/`RB` 在三项间切换、**`A`** 确认（点击「本局不再提示」后自动回移聚焦「确认」便于连按开战）、**`B`** 直点取消、**`START`/`RT+A`** 直点确认、**`X`** 快捷勾选「本局不再提示」。
- **官方原生 `G` 键升本防误触 (`arkcontroller.config` & `arkcontroller.control`)**：
  - 在 `configs/default_bindings.json` 与 `KeymapConfig` 中补齐官方原生快捷键 `"shop_upgrade": "g"`（完整涵盖 `A/S/D/G/C/W/X/Q/Z/Esc`）。
  - 局内按 **`Y`** 升本在原生快捷键模式下直接发送 `G` 键，彻底避免因商店栏位或界面微偏导致的鼠标坐标误买商品卡（纯鼠标模拟器模式下仍保留同步左移坐标点击）。
- **彻底清理游戏外菜单配置，局外统一纯模拟自由鼠标 (`arkcontroller.config` & `arkcontroller.vision` & `arkcontroller.control` & `arkcontroller.overlay`)**：
  - 从 `configs/stronghold_layout_16_9.json`、`LayoutProfile`、`ConfigLoader` 及 `default_bindings.json` 中彻底删除 4 套局外菜单节点表（`activity_hub_nodes`、`pre_match_nodes`、`squad_edit_nodes`、`settlement_nodes`）、5 个局外按钮与 `"menu"` 键位段（`LayoutProfile.from_dict` 向后兼容静默忽略旧 `user_override.json` 残留字段）。
  - 精简 `GameScene` 枚举为 5 项局内核心状态（`BATTLE_PREP`、`MODAL_CHOICE`、`MODAL_CONFIRM`、`BATTLE_COMBAT`、`UNKNOWN_FREE_CURSOR`），移除局外菜单路由；`ControlDispatcher` 默认初始化为 `UNKNOWN_FREE_CURSOR`（HUD 显示「自由鼠标」），局外与设置界面左摇杆始终直接驱动模拟鼠标光标（不受局内 `left_stick_mode="dpad"` 限制），搭配 `A` 单击、`RT+A` 双击、`B` 返回 `Esc`、`RS` 滚轮滚动。
- **右侧策略头像拖拽与点击使用 (`arkcontroller.config` & `arkcontroller.control` & `arkcontroller.overlay`)**：
  - 实机标定固定 `right_edge` 锚点 `strategy_avatar (0.9605, 0.3243)` 与 `strategy_inspect (0.9047, 0.3236)`（不受 `shop_open` 缩放矩阵污染）。
  - 新增 `FocusZone.STRATEGY` 与 `DeployState.HOLDING_STRATEGY`（HUD 渲染琥珀色 `策略源` 虚线框与轨迹）：从棋盘最右列按右跨入策略头像区，支持 **`RT+A`** 一键直接点击使用、**`X`** 查看策略说明、**`A`** 锁定策略源后再按 **`A`** 原地点击使用，或移至棋盘/整备干员与商店卡槽（未展开商店自动先展开）按 **`A`** 平滑拖拽释放、**`B`** 安全取消。
- **修复商店 4/5/6 槽位识别失灵与 UAC 后台截屏自愈 (`arkcontroller.vision` & `arkcontroller.control`)**：
  - 修复 UAC 提权切回安全桌面瞬间 `ScreenCapture._capture_loop` 后台 `mss.mss()` 初始化异常后永久为 `None` 导致视觉识别失明的严重缺陷：改为每轮惰性重建自愈并在采集异常时立即销毁旧实例；同时拦截 Unity DX11 `PrintWindow` 返回的全黑伪帧。
  - 加固 `SceneDetector._detect_shop_slot_count_heuristic`：垂直边框搜索窗放宽至 `±12px` 取局部峰，升级卡检测兼容灰态与自带蓝紫底部发光渐层的紫红高亮 `[G]` 升级卡，商品卡检测对左专属非重叠区与右主区施加双段约束并兼容已售空 `NO INFO` 暗蓝槽，彻底消除亮地图背景下 4 误判 5 / 5 误判 6 问题。
  - 修复 `ControlDispatcher._sync_scene` 在 `手动锁层` 下屏蔽局内视觉同步的问题：当手动锁定 `BATTLE_PREP` 且视觉同为 `BATTLE_PREP` 时，继续实时同步 `shop_open`、`shop_slot_count`、`unit_selected`、`covenant_count` 与 `has_teammates`。

## [0.1.3] - 2026-11-21

### 新增与修复 (Added & Fixed)
- **三选一（单人 `1x3`）与六选一（多人 `2x3`）自动识别及底部白条两段确认 (`arkcontroller.vision` & `arkcontroller.control` & `arkcontroller.config`)**：
  - 修复实机暗底亮灰框决策卡片（如「悬赏决策」）无法被旧亮面卡片启发式识别的问题：基于 Sobel 垂直边框梯度、暗色卡面与青色标题/右侧「收起」按钮特征，零模板自动识别单人 `1x3` 三选一与多人联机 `2x3` 六选一网格，并实时检测哪张卡底部已弹出白色「`☑ 确认选择`」横条（`selected_choice_index`）。
  - 支持多人 `2x3` 二维网格导航（左右同行循环、上下跨排切换、`LB/RB` 全局步进）以及两段式选牌确认：首次按 **`A`** 点击卡片中心弹出底部白色确认条并将 HUD 选框下移聚焦至白条，再次按 **`A`** 精准点击底部白色确认条完成锁定；按 **`RT + A`** 自动依次点击卡片中心与底部白条一键确认；同步校准三选一 `(0.968, 0.500)` 与六选一 `(0.922, 0.500)` 右侧「收起」看场按钮，并在按住 **`LT`** 期间锁定弹窗上下文防止误切层。
- **装备区槽位内部平滑挪动/换位 (`arkcontroller.control.deploy_fsm` & `dispatcher`)**：
  - 新增 `DeployFSM.place_on_equip()`：抓取装备（`HOLDING_EQUIP`）后在 `EQUIPMENT` 槽位间左右移动（或从棋盘/整备区退回装备栏）按 **`A`** 即可平滑拖拽调整装备顺序，同槽位二次按 **`A`** 自动取消选中不发冗余拖拽，体验与整备区干员换位完全一致。
- **修复退出干员详情安全点击区误买商店干员 (`configs/stronghold_layout_16_9.json` & `dispatcher`)**：
  - 将 `safe_blank` 从原底部 `(0.650, 0.880)` 迁移至右侧策略头像正下方安全空白带 `(0.955, 0.445, right_edge)`，彻底杜绝商店展开时按 **`B`** 退出详情误点购买第 2/3 张商店卡的问题。
- **顶部左/右放大镜、居中对称羁绊栏与多人左侧队友头像栏导航 (`arkcontroller.vision` & `arkcontroller.control`)**：
  - 新增 `FocusZone.TOP_INFO` 与 `SymmetricCovenantZone`：局内休整期与作战期从棋盘顶行继续向上即可跨入顶部信息栏，按左右依次聚焦 **左侧放大镜（查看本局信息） ↔ 正下方居中对称分布的已激活/未激活羁绊图标（自动检测奇偶槽数 `0~9`） ↔ 右侧放大镜（查看本轮出怪，`W`）**；支持按 **`A`** 查看详情、按 **`B`** 关闭本局信息右上角 `[×]` `(0.940, 0.061)` / 退出本轮出怪 / 关闭羁绊浮窗且期间不误触发 `Esc` 退局弹窗。
  - 实机校准多人联机左侧 4 枚纵向队友头像 `teammate_1..4` `(u≈0.041, v=0.268..0.583)` 与左下「交流」按钮 `(0.0723, 0.9174)`；自动检测联机队友头像并开放 `FocusZone.TEAMMATES`（从棋盘最左列继续按左跨入），支持上下选队友按 **`A`** 点击查看战场（抓取干员/装备态下按 **`A`** 直接拖拽赠予该队友）。

## [0.1.2] - 2026-11-21

### 变更与修复 (Changed & Fixed)
- **动态商店栏位向左延伸与升级按钮同步左移 (`arkcontroller.config` & `arkcontroller.vision` & `arkcontroller.control`)**：
  - 支持卫戍协议商店升至二本（5 槽）与四本（6 槽）时卡片尺寸、步距与最右锚点保持不变、向左侧等距延伸：`LayoutProfile.resolve_shop_zone(..., slot_count=...)` 自动扩展槽位，`LayoutProfile.resolve_shop_upgrade_point(...)` 同步左移升级商店灰卡坐标，避免升本后按 `Y` 误点新增商品卡。
  - `SceneDetector` 新增零模板实机商店栏位数检测（`SceneDetectionResult.shop_slot_count`），实时识别 4 / 5 / 6 槽（含 6 本满级无升级卡边框兜底）；`ControlDispatcher` 在槽位扩展时自动顺移焦点索引咬住原卡片不乱跳，退局后自动重置为 4 槽。
- **实机官方 PC 端快捷键校准 (`arkcontroller.config` & `arkcontroller.control`)**：
  - 经管理员权限实机全链路验证，对齐官方 PC 端卫戍协议真实原生快捷键：`A`（展开/收起商店）、`S`（冻结商店）、`D`（刷新商店）、`C`（准备就绪）、`W`（敌方路线预览）、`Q`（撤回场上干员）、`X`（场上/整备通用出售）、`Esc`（返回/暂停菜单）。
  - 休整期按 `X`（刷新）、`Y`（升本）、`RS`（冻结）时若商店未展开且启用了 `auto_toggle_shop`，自动先按 `A` 展开商店；同时更新底部商店按 `DPAD_UP` 向上精准切入紧贴正上方的缩放整备区，垂直跨区链路严格吻合 `棋盘 ↔ 装备栏 ↔ 整备区 ↔ 商店`。
- **实机 `16:9` 透视布局与 SIFT 缩放矩阵测绘 (`configs/stronghold_layout_16_9.json`)**：
  - 实机测绘更新卫戍协议标准棋盘为 `4 行 x 7 列` 透视梯形、底部 10 格整备区、棋盘下方横向 5 格装备栏、展开后底部 4 槽等距商店卡与全部按钮坐标（含顶部左/右放大镜、升级灰卡、右侧刷/冻与安全空白点击区 `safe_blank`）。
  - 经 OpenCV SIFT 特征配准（151 内点，中位误差 `1.1px`）写入 `shop_open` 缩放矩阵（`scale_x=0.82772, scale_y=0.79099, offset_u=-0.02889, offset_v=-0.09946`），同时补偿棋盘、整备区与装备栏，彻底消除商店展开导致的框选错位。
- **选取角色详情特写态专属导航与无右键安全取消 (`arkcontroller.control`)**：
  - 实机核实选取角色会触发相机特写居中且官方 PC 端不响应鼠标右键退出：`UNIT_SELECTED` 态下自动切换为左侧固定四栏（`特质 / 技能 / 特性 / 天赋`）焦点导航，支持左右/LB/RB 切换标签、`X` 出售、`Y` 撤回、`B` 左键点击安全空白区 `(0.650, 0.880)` 退出并恢复标准视角。
  - 修复 `DeployFSM.cancel()` 在定朝向态（`AIMING_DIRECTION`）的取消动作：由无效右键改为左键点击定朝向菱形左上角红色 `× 点击取消`。
- **Unity 拖拽起手与过渡动画防竞态优化 (`arkcontroller.input.virtual_hid` & `arkcontroller.control`)**：
  - `VirtualHID.smooth_drag()` 新增起点悬停（`pre_hover_ms=35`）与终点稳固停顿（`post_hold_ms=35`），按下左键后零停顿立刻插值起拖，彻底消除拖拽起手失败或误判单击诱发特写镜头缩放的问题。
  - 新增 `VirtualHID.wait_ms()`（复用可注入 `sleep_fn`，单测零阻塞），在 `LT+RB` 一键撤回、原生快捷键出售及商店自动展开紧接操作间等待 `220ms` 过渡动画就绪，解决 0ms 连续输入被客户端吞键问题。
- **零模板实机视觉强特征识别 (`arkcontroller.vision.scene_detector`)**：
  - 新增官方 PC 端右上角粉红「准备就绪」按钮、展开商店右侧「黄色刷新 + 蓝色冻结/青绿解冻」双态色块、左侧角色详情青蓝血条（并排除拖拽售出金黄框）的零模板 HSV 强特征判别，实机全态测试 100% 准确。

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
