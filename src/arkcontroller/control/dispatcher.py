"""《明日方舟》卫戍协议全流程操控调度中枢（ControlDispatcher）。

负责整合窗口前景安全守卫、自动/手动场景层路由、槽位吸附导航与自由虚拟光标无缝切换、
两段式部署状态机联动，以及向透明悬浮 HUD 输出实时只读状态快照（HUDStateSnapshot）。
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

from arkcontroller.config.models import LayoutViewState
from arkcontroller.control.deploy_fsm import DeployFSM, DeployState, RumbleCallback
from arkcontroller.control.grid_mapper import (
    Direction,
    LinearZoneMapper,
    NodeGraphMapper,
    PerspectiveBoardMapper,
)
from arkcontroller.input.gesture_engine import (
    ButtonHoldProgress,
    ButtonHoldTriggered,
    ButtonRepeat,
    ButtonTap,
    ChordTriggered,
    ModifierStateChanged,
    StickDirection,
    StickMotion,
)
from arkcontroller.input.xinput_driver import GamepadButton
from arkcontroller.vision.scene_detector import GameScene, SceneDetectionResult

if TYPE_CHECKING:
    from arkcontroller.config.models import KeymapConfig, LayoutProfile
    from arkcontroller.input.virtual_hid import VirtualHID
    from arkcontroller.platform.window_tracker import ViewportRect

GestureEvent = (
    ButtonTap
    | ButtonHoldProgress
    | ButtonHoldTriggered
    | ButtonRepeat
    | ChordTriggered
    | ModifierStateChanged
    | StickMotion
    | StickDirection
)


class FocusZone(str, Enum):
    """当前焦点吸附所属功能分区枚举。"""

    MENU_NODES = "menu_nodes"
    SHOP = "shop"
    BENCH = "bench"
    BOARD = "board"
    EQUIPMENT = "equipment"
    CHOICE_CARDS = "choice_cards"
    FREE_CURSOR = "free_cursor"


@dataclass(frozen=True)
class PromptItem:
    """底部 HUD 键位提示栏单个提示条目。"""

    button: str
    label: str


@dataclass(frozen=True)
class HUDStateSnapshot:
    """供透明 HUD 悬浮层实时渲染的不可变状态快照。"""

    visible: bool
    paused: bool
    calibration_requested: bool
    scene: GameScene
    scene_locked_manual: bool
    focus_zone: FocusZone
    focus_label: str
    focus_polygon_norm: list[tuple[float, float]] | None
    holding_source_norm: tuple[float, float] | None
    cursor_norm: tuple[float, float]
    free_cursor_active: bool
    deploy_state: DeployState
    aim_origin_norm: tuple[float, float] | None
    aim_preview_dir: Literal["up", "down", "left", "right"] | None
    ready_hold_progress: float
    prompts: list[PromptItem]
    toast_message: str | None = None
    view_state: str = "normal"
    shop_open: bool = False
    left_stick_mode: str = "dpad"


_DPAD_TO_DIR: dict[GamepadButton, Direction] = {
    GamepadButton.DPAD_UP: "up",
    GamepadButton.DPAD_DOWN: "down",
    GamepadButton.DPAD_LEFT: "left",
    GamepadButton.DPAD_RIGHT: "right",
}

_DIR_TO_DPAD: dict[Direction, GamepadButton] = {
    "up": GamepadButton.DPAD_UP,
    "down": GamepadButton.DPAD_DOWN,
    "left": GamepadButton.DPAD_LEFT,
    "right": GamepadButton.DPAD_RIGHT,
}

_BATTLE_PREP_ZONE_ORDER: tuple[FocusZone, ...] = (
    FocusZone.SHOP,
    FocusZone.BENCH,
    FocusZone.BOARD,
    FocusZone.EQUIPMENT,
)

# Back 单击手动切层序列：None 表示恢复自动识别
_MANUAL_LAYER_CYCLE: tuple[GameScene | None, ...] = (
    None,
    GameScene.ACTIVITY_HUB,
    GameScene.SQUAD_EDIT,
    GameScene.BATTLE_PREP,
    GameScene.MODAL_CHOICE,
    GameScene.UNKNOWN_FREE_CURSOR,
)

_MANUAL_SCENE_TITLES: dict[GameScene, str] = {
    GameScene.ACTIVITY_HUB: "菜单主界面层",
    GameScene.PRE_MATCH_SETUP: "模式准备层",
    GameScene.SQUAD_EDIT: "战前编队层",
    GameScene.BATTLE_PREP: "局内运营层",
    GameScene.MODAL_CHOICE: "弹窗三选一层",
    GameScene.BATTLE_COMBAT: "作战观战层",
    GameScene.SETTLEMENT: "结算层",
    GameScene.UNKNOWN_FREE_CURSOR: "纯鼠标自由层",
}

_MENU_SCENES: frozenset[GameScene] = frozenset(
    {
        GameScene.ACTIVITY_HUB,
        GameScene.PRE_MATCH_SETUP,
        GameScene.SQUAD_EDIT,
        GameScene.SETTLEMENT,
    }
)

_SQUAD_SKILL_BUTTON_KEYS: tuple[tuple[str, str, str], ...] = (
    ("squad_skill_1", "skill_1", "技能 1"),
    ("squad_skill_2", "skill_2", "技能 2"),
    ("squad_skill_3", "skill_3", "技能 3"),
    ("squad_module", "module_slot", "干员模组"),
)

_UNIT_DETAIL_TABS: tuple[tuple[str, str, float, float], ...] = (
    ("unit_tab_trait", "特质", 0.045, 0.515),
    ("unit_tab_skill", "技能", 0.135, 0.515),
    ("unit_tab_feature", "特性", 0.215, 0.515),
    ("unit_tab_talent", "天赋", 0.295, 0.515),
)

_DIRECT_TEAMMATE_BUTTONS: dict[GamepadButton, int] = {
    GamepadButton.A: 1,
    GamepadButton.B: 2,
    GamepadButton.X: 3,
    GamepadButton.Y: 4,
}

_AIM_PREVIEW_THRESHOLD: float = 0.25
_AIM_COMMIT_THRESHOLD: float = 0.65
_SCROLL_STEP_PX: float = 24.0
_SCROLL_WHEEL_DELTA: int = 120
_TOAST_TTL_SECONDS: float = 1.8
_STATE_SYNC_LOCK_SECONDS: float = 0.50


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _stick_to_cardinal(x: float, y: float) -> Direction:
    """将摇杆向量 (x 右正, y 上正) 量化为四向朝向。"""
    if abs(x) >= abs(y):
        return "right" if x >= 0.0 else "left"
    return "up" if y >= 0.0 else "down"


class ControlDispatcher:
    """卫戍协议手柄全流程操控调度中枢。"""

    def __init__(
        self,
        keymap: KeymapConfig,
        layout: LayoutProfile,
        hid: VirtualHID,
        rumble_cb: RumbleCallback | None = None,
    ) -> None:
        self.keymap = keymap
        self.layout = layout
        self.hid = hid
        self._rumble_cb = rumble_cb

        self.deploy_fsm = DeployFSM(keymap, layout, hid, rumble_cb=rumble_cb)

        # 界面镜头/缩放视图态（normal / shop_open / unit_selected）
        self._view_state: LayoutViewState = LayoutViewState.NORMAL
        self._shop_open: bool = False
        self._unit_selected: bool = False
        self._shop_override_until: float = 0.0
        self._unit_override_until: float = 0.0

        # 构建多视图态区域映射器缓存
        self._shop_mappers: dict[LayoutViewState, LinearZoneMapper] = {}
        self._bench_mappers: dict[LayoutViewState, LinearZoneMapper] = {}
        self._equip_mappers: dict[LayoutViewState, LinearZoneMapper] = {}
        self._board_mappers: dict[LayoutViewState, PerspectiveBoardMapper] = {}
        self._shop_mapper: LinearZoneMapper
        self._bench_mapper: LinearZoneMapper
        self._equip_mapper: LinearZoneMapper
        self._choice_mapper: LinearZoneMapper
        self._board_mapper: PerspectiveBoardMapper
        self._menu_mappers: dict[GameScene, NodeGraphMapper] = {}
        self._rebuild_mappers()

        # 运行态开关
        self._paused: bool = False
        self._calibration_requested: bool = False

        # 场景与手动锁层状态
        self._current_scene: GameScene = GameScene.ACTIVITY_HUB
        self._manual_cycle_idx: int = 0  # 0 表示自动识别（未手动锁层）
        self._choice_count: int = self._choice_mapper.count

        # 焦点区与各区索引
        self._focus_zone: FocusZone = FocusZone.MENU_NODES
        self._menu_node_ids: dict[GameScene, str] = {}
        self._init_menu_node_ids()
        self._shop_idx: int = 0
        self._bench_idx: int = 0
        self._equip_idx: int = 0
        self._choice_idx: int = min(1, max(0, self._choice_count - 1))
        self._board_row: int = self._board_mapper.rows // 2
        self._board_col: int = self._board_mapper.cols // 2

        # 自由光标与滚轮积分器
        self._free_cursor_active: bool = False
        self._cursor_norm: tuple[float, float] = (0.5, 0.5)
        self._scroll_v_accum: float = 0.0
        self._scroll_h_accum: float = 0.0

        # 编队技能/模组循环索引、角色详情标签索引、联机队友视角与三选一小眼睛状态
        self._squad_slot_idx: int = 0
        self._unit_tab_idx: int = 0
        self._teammate_cycle_pos: int = 0  # 0=未切队友，首次循环落到 1 号位
        self._selected_teammate: int = 1
        self._choice_eye_hidden: bool = False
        self._ready_hold_progress: float = 0.0

        # 顶部短暂提示信息
        self._toast_message: str | None = None
        self._toast_expire_at: float = 0.0

    # ------------------------------------------------------------------
    # 公共控制与配置管理接口
    # ------------------------------------------------------------------

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def scene_locked_manual(self) -> bool:
        return self._manual_cycle_idx != 0

    @property
    def current_scene(self) -> GameScene:
        return self._current_scene

    @property
    def selected_teammate(self) -> int:
        return self._selected_teammate

    @property
    def view_state(self) -> LayoutViewState:
        return self._view_state

    @property
    def shop_open(self) -> bool:
        return self._shop_open

    @property
    def unit_selected(self) -> bool:
        return self._unit_selected

    def update_configs(self, keymap: KeymapConfig, layout: LayoutProfile) -> None:
        """热重载键位配置与 16:9 布局配置，并安全裁剪全部槽位索引。"""
        self.keymap = keymap
        self.layout = layout
        self.deploy_fsm.update_configs(keymap, layout)
        self._rebuild_mappers()

        self._shop_idx = self._shop_mapper.clamp_index(self._shop_idx)
        self._bench_idx = self._bench_mapper.clamp_index(self._bench_idx)
        self._equip_idx = self._equip_mapper.clamp_index(self._equip_idx)
        self._choice_count = max(1, min(self._choice_count, max(self._choice_mapper.count, self._choice_count)))
        self._choice_idx = max(0, min(self._choice_count - 1, self._choice_idx))
        self._board_row = max(0, min(self._board_mapper.rows - 1, self._board_row))
        self._board_col = max(0, min(self._board_mapper.cols - 1, self._board_col))

        for scene, mapper in self._menu_mappers.items():
            cur_id = self._menu_node_ids.get(scene)
            if cur_id is None or mapper.get_node(cur_id) is None:
                default_id = mapper.default_node_id()
                if default_id is not None:
                    self._menu_node_ids[scene] = default_id
                else:
                    self._menu_node_ids.pop(scene, None)

    def acknowledge_calibration_request(self) -> None:
        """由外部 UI 确认已响应校准面板呼出请求，清除挂起标志。"""
        self._calibration_requested = False

    def toggle_pause(self) -> bool:
        """切换手柄映射暂停状态并返回切换后的布尔值。"""
        self._paused = not self._paused
        if self._paused:
            self.deploy_fsm.reset()
            self._ready_hold_progress = 0.0
            self._set_toast("手柄映射已暂停（LT+Back 恢复）")
        else:
            self._set_toast("手柄映射已恢复")
        self._emit_rumble(0.20, 0.25, 45)
        return self._paused

    # ------------------------------------------------------------------
    # 主调度入口
    # ------------------------------------------------------------------

    def process_tick(
        self,
        viewport: ViewportRect | None,
        scene_result: SceneDetectionResult | None,
        events: Sequence[GestureEvent],
    ) -> HUDStateSnapshot:
        """处理单个主循环帧：窗口守卫 -> 场景同步 -> 手势路由 -> 生成 HUD 快照。"""
        self._expire_toast_if_needed()

        # 最高优先级安全护栏：游戏窗口未找到或处于后台时，严禁向 HID 发送任何输入
        if viewport is None or not viewport.is_foreground:
            self._ready_hold_progress = 0.0
            if self._choice_eye_hidden:
                self._choice_eye_hidden = False
            return self._build_snapshot(visible=False)

        # 同步自动识别场景（手动锁层或部署拖拽态下不强行打断）
        self._sync_scene(scene_result)

        # 逐条处理本帧手势事件
        for event in events:
            self._handle_event(event, viewport)

        return self._build_snapshot(visible=True)

    # ------------------------------------------------------------------
    # 内部初始化与工具方法
    # ------------------------------------------------------------------

    def _rebuild_mappers(self) -> None:
        self._shop_mappers = {}
        self._bench_mappers = {}
        self._equip_mappers = {}
        self._board_mappers = {}
        for vs in LayoutViewState:
            self._shop_mappers[vs] = LinearZoneMapper(self.layout.resolve_shop_zone(vs))
            self._bench_mappers[vs] = LinearZoneMapper(self.layout.resolve_bench_zone(vs))
            self._equip_mappers[vs] = LinearZoneMapper(self.layout.resolve_equip_zone(vs))
            self._board_mappers[vs] = PerspectiveBoardMapper(
                self.layout.resolve_board_quad(vs)
            )
        self._choice_mapper = LinearZoneMapper(self.layout.choice_zone)
        self._menu_mappers = {
            GameScene.ACTIVITY_HUB: NodeGraphMapper(self.layout.activity_hub_nodes),
            GameScene.PRE_MATCH_SETUP: NodeGraphMapper(self.layout.pre_match_nodes),
            GameScene.SQUAD_EDIT: NodeGraphMapper(self.layout.squad_edit_nodes),
            GameScene.SETTLEMENT: NodeGraphMapper(self.layout.settlement_nodes),
        }
        self._sync_active_mappers()

    def _sync_active_mappers(self) -> None:
        """将当前生效的槽位与棋盘映射器指向对应缩放视图态缓存。"""
        vs = self._view_state
        self._shop_mapper = self._shop_mappers[vs]
        self._bench_mapper = self._bench_mappers[vs]
        self._equip_mapper = self._equip_mappers[vs]
        self._board_mapper = self._board_mappers[vs]

    def _compute_effective_view_state(self) -> LayoutViewState:
        """依据当前场景、选取角色状态与商店展开状态推导镜头缩放视图态。"""
        if self._current_scene not in (GameScene.BATTLE_PREP, GameScene.BATTLE_COMBAT):
            return LayoutViewState.NORMAL
        if self._unit_selected:
            return LayoutViewState.UNIT_SELECTED
        if self._current_scene == GameScene.BATTLE_PREP and self._shop_open:
            return LayoutViewState.SHOP_OPEN
        return LayoutViewState.NORMAL

    def _refresh_view_state(self) -> None:
        new_state = self._compute_effective_view_state()
        if new_state != self._view_state:
            self._view_state = new_state
            self._sync_active_mappers()

    def _init_menu_node_ids(self) -> None:
        self._menu_node_ids.clear()
        for scene, mapper in self._menu_mappers.items():
            default_id = mapper.default_node_id()
            if default_id is not None:
                self._menu_node_ids[scene] = default_id

    def _emit_rumble(self, left: float, right: float, duration_ms: int) -> None:
        if self._rumble_cb is not None and self.keymap.vibration_enabled:
            self._rumble_cb(left, right, duration_ms)

    def _set_toast(self, message: str) -> None:
        self._toast_message = message
        self._toast_expire_at = time.monotonic() + _TOAST_TTL_SECONDS

    def _expire_toast_if_needed(self) -> None:
        if self._toast_message is not None and time.monotonic() >= self._toast_expire_at:
            self._toast_message = None

    def _get_shortcut(self, name: str, default: str | None = None) -> str | None:
        shortcuts = getattr(self.keymap, "keyboard_shortcuts", None)
        if isinstance(shortcuts, dict) and name in shortcuts:
            return shortcuts[name]
        return default

    def _click_norm(
        self,
        u: float,
        v: float,
        viewport: ViewportRect,
        *,
        anchor: str = "center_16_9",
        button: str = "left",
    ) -> None:
        sx, sy = viewport.norm_to_screen(u, v, anchor=anchor)
        self.hid.click(sx, sy, button=button)

    def _click_button(self, button_key: str, viewport: ViewportRect) -> bool:
        pt = self.layout.buttons.get(button_key)
        if pt is None:
            return False
        self._click_norm(pt.u, pt.v, viewport, anchor=pt.anchor, button="left")
        return True

    def _click_safe_blank(self, viewport: ViewportRect) -> None:
        """左键点击安全空白区以退出角色详情特写视角（官方 PC 端不识别右键取消）。"""
        if not self._click_button("safe_blank", viewport):
            self._click_norm(0.650, 0.880, viewport, button="left")

    def _unit_detail_tab_center(self, idx: int) -> tuple[float, float]:
        btn_key, _, default_u, default_v = _UNIT_DETAIL_TABS[idx % len(_UNIT_DETAIL_TABS)]
        pt = self.layout.buttons.get(btn_key)
        if pt is not None:
            return (pt.u, pt.v)
        return (default_u, default_v)

    def _unit_detail_tab_polygon(self, idx: int) -> list[tuple[float, float]]:
        cu, cv = self._unit_detail_tab_center(idx)
        hw, hh = 0.038, 0.022
        return [
            (cu - hw, cv - hh),
            (cu + hw, cv - hh),
            (cu + hw, cv + hh),
            (cu - hw, cv + hh),
        ]

    def _trigger_action(
        self,
        button_key: str,
        shortcut_name: str | None,
        default_key: str | None,
        viewport: ViewportRect,
    ) -> bool:
        """优先根据 use_native_keyboard_shortcuts 发送扫描码快捷键，否则点击对应按钮坐标。"""
        if self.keymap.use_native_keyboard_shortcuts and shortcut_name is not None:
            key = self._get_shortcut(shortcut_name, default_key)
            if key:
                self.hid.tap_key(key)
                return True
        return self._click_button(button_key, viewport)

    # ------------------------------------------------------------------
    # 场景切换与焦点同步
    # ------------------------------------------------------------------

    def _sync_scene(self, scene_result: SceneDetectionResult | None) -> None:
        if scene_result is None or self.scene_locked_manual:
            return

        now = time.monotonic()
        if now >= self._shop_override_until:
            self._shop_open = bool(scene_result.shop_open)
        if now >= self._unit_override_until:
            self._unit_selected = bool(getattr(scene_result, "unit_selected", False))

        new_choice_count = max(1, int(scene_result.choice_count))
        if new_choice_count != self._choice_count:
            self._choice_count = new_choice_count
            self._choice_idx = max(0, min(self._choice_count - 1, self._choice_idx))

        if scene_result.scene == self._current_scene:
            self._refresh_view_state()
            return

        # 若玩家正在拖拽或定朝向，不因短暂视觉波动强行中断部署流程
        if self.deploy_fsm.is_active and scene_result.scene != GameScene.MODAL_CHOICE:
            self._refresh_view_state()
            return

        self._apply_scene_transition(scene_result.scene)

    def _apply_scene_transition(self, target_scene: GameScene) -> None:
        if self._current_scene == GameScene.MODAL_CHOICE and target_scene != GameScene.MODAL_CHOICE:
            self._choice_eye_hidden = False

        if target_scene != GameScene.BATTLE_PREP and self.deploy_fsm.is_active:
            self.deploy_fsm.reset()

        if target_scene not in (GameScene.BATTLE_PREP, GameScene.BATTLE_COMBAT):
            self._unit_selected = False
            self._unit_override_until = 0.0
        if target_scene != GameScene.BATTLE_PREP:
            self._shop_override_until = 0.0

        self._current_scene = target_scene
        self._ready_hold_progress = 0.0

        if target_scene in _MENU_SCENES:
            self._focus_zone = FocusZone.MENU_NODES
            self._free_cursor_active = False
            mapper = self._menu_mappers[target_scene]
            cur_id = self._menu_node_ids.get(target_scene)
            if cur_id is None or mapper.get_node(cur_id) is None:
                default_id = mapper.default_node_id()
                if default_id is not None:
                    self._menu_node_ids[target_scene] = default_id
        elif target_scene == GameScene.BATTLE_PREP:
            self._focus_zone = FocusZone.SHOP if self._shop_open else FocusZone.BENCH
            self._free_cursor_active = False
        elif target_scene == GameScene.BATTLE_COMBAT:
            self._focus_zone = FocusZone.BOARD
            self._free_cursor_active = False
        elif target_scene == GameScene.MODAL_CHOICE:
            self._focus_zone = FocusZone.CHOICE_CARDS
            self._free_cursor_active = False
            self._choice_idx = min(self._choice_count // 2, self._choice_count - 1)
        else:
            self._focus_zone = FocusZone.FREE_CURSOR
            self._free_cursor_active = True

        self._refresh_view_state()

    def _set_shop_open(
        self,
        target_open: bool,
        viewport: ViewportRect,
        *,
        force: bool = False,
        announce: bool = True,
    ) -> bool:
        """展开或收起商店面板，并同步更新视图缩放态与视觉防抖保护窗口。"""
        if not force and self._shop_open == target_open:
            return False

        toggled = self._trigger_action("shop_toggle", "shop_toggle", "a", viewport)
        self._shop_open = bool(target_open)
        self._shop_override_until = time.monotonic() + _STATE_SYNC_LOCK_SECONDS
        if self._shop_open:
            self._unit_selected = False
            self._unit_override_until = time.monotonic() + _STATE_SYNC_LOCK_SECONDS
        self._refresh_view_state()
        if announce:
            self._set_toast("已展开商店（缩放补偿已同步）" if self._shop_open else "已收起商店（恢复标准视角）")
        return toggled

    def _set_unit_selected(
        self,
        selected: bool,
        *,
        lock: bool = True,
    ) -> None:
        """同步角色详情面板开关状态，并更新对应缩放透视映射器。"""
        self._unit_selected = bool(selected)
        if lock:
            self._unit_override_until = time.monotonic() + _STATE_SYNC_LOCK_SECONDS
        self._refresh_view_state()

    def _enter_battle_prep_zone(
        self, target_zone: FocusZone, viewport: ViewportRect
    ) -> None:
        """切换休整期焦点分区，并根据 auto_toggle_shop 自动联动商店展开/收起。"""
        self._focus_zone = target_zone
        if not getattr(self.keymap, "auto_toggle_shop", True):
            return
        if target_zone == FocusZone.SHOP and not self._shop_open:
            self._set_shop_open(True, viewport, announce=True)
        elif target_zone == FocusZone.BOARD and self._shop_open:
            self._set_shop_open(False, viewport, announce=True)

    def _cycle_manual_layer(self) -> None:
        """Back 单击：在自动识别与各功能操作层之间循环切换。"""
        self._manual_cycle_idx = (self._manual_cycle_idx + 1) % len(_MANUAL_LAYER_CYCLE)
        locked_scene = _MANUAL_LAYER_CYCLE[self._manual_cycle_idx]
        self._emit_rumble(0.18, 0.22, 40)

        if locked_scene is None:
            self._set_toast("已切回自动场景识别")
            return

        self._apply_scene_transition(locked_scene)
        title = _MANUAL_SCENE_TITLES.get(locked_scene, locked_scene.value)
        self._set_toast(f"手动锁定：{title}")

    def _current_snap_center(self) -> tuple[float, float]:
        """获取当前吸附焦点的归一化中心坐标。"""
        if (
            self._unit_selected
            and self._current_scene in (GameScene.BATTLE_PREP, GameScene.BATTLE_COMBAT)
            and not self.deploy_fsm.is_active
        ):
            return self._unit_detail_tab_center(self._unit_tab_idx)
        if self._focus_zone == FocusZone.MENU_NODES:
            mapper = self._menu_mappers.get(self._current_scene)
            node_id = self._menu_node_ids.get(self._current_scene)
            if mapper is not None and node_id is not None and mapper.get_node(node_id) is not None:
                return mapper.node_center(node_id)
            return self._cursor_norm
        if self._focus_zone == FocusZone.SHOP:
            return self._shop_mapper.slot_center(self._shop_idx)
        if self._focus_zone == FocusZone.BENCH:
            return self._bench_mapper.slot_center(self._bench_idx)
        if self._focus_zone == FocusZone.BOARD:
            return self._board_mapper.cell_center(self._board_row, self._board_col)
        if self._focus_zone == FocusZone.EQUIPMENT:
            return self._equip_mapper.slot_center(self._equip_idx)
        if self._focus_zone == FocusZone.CHOICE_CARDS:
            return self._choice_mapper.slot_center(
                self._choice_idx, count=self._choice_count
            )
        return self._cursor_norm

    def _current_snap_polygon(self) -> list[tuple[float, float]] | None:
        """获取当前吸附焦点的 4 顶点归一化多边形。"""
        if self._free_cursor_active or self._focus_zone == FocusZone.FREE_CURSOR:
            return None
        if self._focus_zone == FocusZone.CHOICE_CARDS and self._choice_eye_hidden:
            return None
        if (
            self._unit_selected
            and self._current_scene in (GameScene.BATTLE_PREP, GameScene.BATTLE_COMBAT)
            and not self.deploy_fsm.is_active
        ):
            return self._unit_detail_tab_polygon(self._unit_tab_idx)
        if self._focus_zone == FocusZone.MENU_NODES:
            mapper = self._menu_mappers.get(self._current_scene)
            node_id = self._menu_node_ids.get(self._current_scene)
            if mapper is not None and node_id is not None and mapper.get_node(node_id) is not None:
                return mapper.node_polygon(node_id)
            return None
        if self._focus_zone == FocusZone.SHOP:
            return self._shop_mapper.slot_polygon(self._shop_idx)
        if self._focus_zone == FocusZone.BENCH:
            return self._bench_mapper.slot_polygon(self._bench_idx)
        if self._focus_zone == FocusZone.BOARD:
            return self._board_mapper.cell_polygon(self._board_row, self._board_col)
        if self._focus_zone == FocusZone.EQUIPMENT:
            return self._equip_mapper.slot_polygon(self._equip_idx)
        if self._focus_zone == FocusZone.CHOICE_CARDS:
            return self._choice_mapper.slot_polygon(
                self._choice_idx, count=self._choice_count
            )
        return None

    # ------------------------------------------------------------------
    # 事件总路由
    # ------------------------------------------------------------------

    def _handle_event(self, event: GestureEvent, viewport: ViewportRect) -> None:
        # 1. 全局最高优先级组合键（即使处于暂停态依然响应）
        if isinstance(event, ChordTriggered):
            pair = {event.modifier, event.button}
            if pair == {GamepadButton.BACK, GamepadButton.START}:
                self._calibration_requested = True
                self._set_toast("已请求打开可视化校准向导")
                self._emit_rumble(0.25, 0.35, 50)
                return
            if pair == {GamepadButton.LT, GamepadButton.BACK}:
                self.toggle_pause()
                return

        # 暂停态拦截所有其他模拟操作
        if self._paused:
            return

        # 2. 全局 Back 单击手动切层；left_stick_mode="dpad" 下按 LS 一键切换自由光标
        if isinstance(event, ButtonTap):
            if event.button == GamepadButton.BACK:
                self._cycle_manual_layer()
                return
            if (
                event.button == GamepadButton.LS
                and getattr(self.keymap, "left_stick_mode", "dpad") == "dpad"
                and self.deploy_fsm.state != DeployState.AIMING_DIRECTION
                and self._current_scene != GameScene.UNKNOWN_FREE_CURSOR
            ):
                self._toggle_free_cursor()
                return

        # 3. 修饰键按下/松开状态变化（用于三选一按住 LT 查看战场、松开恢复）
        if isinstance(event, ModifierStateChanged):
            self._handle_modifier_changed(event, viewport)
            return

        # 4. 左摇杆离散方向事件（代替十字键进行槽位与节点吸附导航）
        if isinstance(event, StickDirection):
            self._handle_stick_direction(event, viewport)
            return

        # 5. 摇杆连续运动（定朝向 / 左摇杆自由光标 / 右摇杆滚轮）
        if isinstance(event, StickMotion):
            self._handle_stick_motion(event, viewport)
            return

        # 5. Start 长按进度与触发（局内防误触准备开战）
        if isinstance(event, ButtonHoldProgress):
            if event.button == GamepadButton.START and self._current_scene == GameScene.BATTLE_PREP:
                self._ready_hold_progress = _clamp01(event.progress)
            return

        if isinstance(event, ButtonHoldTriggered):
            self._handle_hold_triggered(event, viewport)
            return

        # 6. 其余组合键
        if isinstance(event, ChordTriggered):
            self._handle_chord(event, viewport)
            return

        # 7. 单击与连发事件路由
        if isinstance(event, (ButtonTap, ButtonRepeat)):
            self._handle_button_action(
                event.button,
                viewport,
                is_repeat=isinstance(event, ButtonRepeat),
            )

    # ------------------------------------------------------------------
    # 修饰键与摇杆事件处理
    # ------------------------------------------------------------------

    def _handle_modifier_changed(
        self, event: ModifierStateChanged, viewport: ViewportRect
    ) -> None:
        if event.modifier != GamepadButton.LT:
            return
        if self._current_scene != GameScene.MODAL_CHOICE:
            return

        if event.active and not self._choice_eye_hidden:
            if self._click_button("choice_toggle_eye", viewport):
                self._choice_eye_hidden = True
                self._set_toast("查看战场中（松开 LT 恢复弹窗）")
        elif not event.active and self._choice_eye_hidden:
            if self._click_button("choice_toggle_eye", viewport):
                self._choice_eye_hidden = False
                self._set_toast("已恢复三选一弹窗")

    def _toggle_free_cursor(self) -> None:
        """LS 按下：一键切换槽位吸附导航与自由虚拟光标模式。"""
        if not self._free_cursor_active:
            self._cursor_norm = self._current_snap_center()
            self._free_cursor_active = True
            self._set_toast("已切至自由光标（再按 LS 或十字键恢复吸附）")
        else:
            self._free_cursor_active = False
            self._set_toast("已恢复槽位吸附导航（左摇杆 / 十字键）")
        self._emit_rumble(0.16, 0.22, 35)

    def _handle_stick_direction(
        self, event: StickDirection, viewport: ViewportRect
    ) -> None:
        """左摇杆离散四向事件直接等效于十字键，驱动全场景吸附导航。"""
        if event.stick != "left":
            return
        if getattr(self.keymap, "left_stick_mode", "dpad") != "dpad":
            return
        if self.deploy_fsm.state == DeployState.AIMING_DIRECTION:
            return
        if self._free_cursor_active or self._current_scene == GameScene.UNKNOWN_FREE_CURSOR:
            return

        dpad_btn = _DIR_TO_DPAD[event.direction]
        self._handle_button_action(dpad_btn, viewport, is_repeat=event.is_repeat)

    def _handle_stick_motion(
        self, event: StickMotion, viewport: ViewportRect
    ) -> None:
        # 定朝向态下，左/右摇杆均直接驱动四向定朝向轮盘
        if self.deploy_fsm.state == DeployState.AIMING_DIRECTION:
            mag = math.hypot(event.x, event.y)
            if mag < _AIM_PREVIEW_THRESHOLD:
                return
            direction = _stick_to_cardinal(event.x, event.y)
            self.deploy_fsm.set_preview_direction(direction)
            if mag >= _AIM_COMMIT_THRESHOLD:
                self.deploy_fsm.confirm_direction(direction, viewport)
                self._set_toast(f"定朝向完成：{self._dir_label(direction)}")
            return

        if event.stick == "left":
            # 左摇杆代替十字键模式下，仅在显式开启自由光标或未知自由层时移动鼠标光标
            if (
                getattr(self.keymap, "left_stick_mode", "dpad") == "dpad"
                and not self._free_cursor_active
                and self._current_scene != GameScene.UNKNOWN_FREE_CURSOR
            ):
                return
            if (
                abs(event.dx_px) <= 1e-6
                and abs(event.dy_px) <= 1e-6
                and math.hypot(event.x, event.y) <= 1e-3
            ):
                return
            if not self._free_cursor_active:
                self._cursor_norm = self._current_snap_center()
                self._free_cursor_active = True

            du = event.dx_px / max(1, viewport.active_width)
            dv = event.dy_px / max(1, viewport.active_height)
            self._cursor_norm = (
                _clamp01(self._cursor_norm[0] + du),
                _clamp01(self._cursor_norm[1] + dv),
            )
            sx, sy = viewport.norm_to_screen(*self._cursor_norm)
            self.hid.move_cursor(sx, sy)
            return

        # 右摇杆模拟垂直/水平滚轮滚动（严格使用 GestureEngine 亚像素积分后的像素增量）
        scroll_scale = getattr(self.keymap, "scroll_speed", 1.0)
        raw_v = -event.dy_px
        raw_h = event.dx_px

        # 反向推杆时清零异号滚轮余量，消除换向迟滞
        if raw_v * self._scroll_v_accum < 0.0:
            self._scroll_v_accum = 0.0
        if raw_h * self._scroll_h_accum < 0.0:
            self._scroll_h_accum = 0.0

        self._scroll_v_accum += raw_v * scroll_scale
        self._scroll_h_accum += raw_h * scroll_scale

        if abs(self._scroll_v_accum) >= _SCROLL_STEP_PX:
            ticks = int(self._scroll_v_accum / _SCROLL_STEP_PX)
            self._scroll_v_accum -= ticks * _SCROLL_STEP_PX
            self.hid.scroll(ticks * _SCROLL_WHEEL_DELTA, horizontal=False)

        if abs(self._scroll_h_accum) >= _SCROLL_STEP_PX:
            ticks = int(self._scroll_h_accum / _SCROLL_STEP_PX)
            self._scroll_h_accum -= ticks * _SCROLL_STEP_PX
            self.hid.scroll(ticks * _SCROLL_WHEEL_DELTA, horizontal=True)

    # ------------------------------------------------------------------
    # 长按与组合键处理
    # ------------------------------------------------------------------

    def _handle_hold_triggered(
        self, event: ButtonHoldTriggered, viewport: ViewportRect
    ) -> None:
        if event.button != GamepadButton.START:
            return

        if self._current_scene == GameScene.BATTLE_PREP:
            self._ready_hold_progress = 0.0
            self._trigger_battle_ready(viewport)
        elif self._current_scene in _MENU_SCENES:
            self._click_button("start_action", viewport)
            self._set_toast("执行确认 / 开始")

    def _trigger_battle_ready(self, viewport: ViewportRect) -> None:
        if self.deploy_fsm.is_active:
            self.deploy_fsm.cancel(viewport)
        self._trigger_action("battle_ready", "battle_ready", "c", viewport)
        self._emit_rumble(0.50, 0.50, 80)
        self._set_toast("准备就绪 · 开始战斗")

    def _handle_chord(self, event: ChordTriggered, viewport: ViewportRect) -> None:
        # RT + A：商店极速连买 / 三选一双击拿牌 / 自由光标双击
        if event.modifier == GamepadButton.RT and event.button == GamepadButton.A:
            self._handle_rt_a(viewport)
            return

        # RT + Start：局内快捷准备就绪
        if {event.modifier, event.button} == {GamepadButton.RT, GamepadButton.START}:
            if self._current_scene == GameScene.BATTLE_PREP:
                self._ready_hold_progress = 0.0
                self._trigger_battle_ready(viewport)
            elif self._current_scene in _MENU_SCENES:
                self._click_button("start_action", viewport)
            return

        # LT 信息速查与联机组合键（局内休整期与作战期全程可用）
        if event.modifier == GamepadButton.LT and self._current_scene in (
            GameScene.BATTLE_PREP,
            GameScene.BATTLE_COMBAT,
        ):
            self._handle_in_battle_lt_chord(event.button, viewport)

    def _handle_rt_a(self, viewport: ViewportRect) -> None:
        if self._free_cursor_active or self._current_scene == GameScene.UNKNOWN_FREE_CURSOR:
            sx, sy = viewport.norm_to_screen(*self._cursor_norm)
            self.hid.double_click(sx, sy)
            self._emit_rumble(0.22, 0.30, 40)
            return

        if (
            self._current_scene == GameScene.BATTLE_PREP
            and self._focus_zone == FocusZone.SHOP
            and self.deploy_fsm.state == DeployState.IDLE
        ):
            if getattr(self.keymap, "auto_toggle_shop", True) and not self._shop_open:
                self._set_shop_open(True, viewport, announce=False)
                self._wait_ui_transition(220)
            u, v = self._shop_mapper.slot_center(self._shop_idx)
            sx, sy = viewport.norm_to_screen(u, v)
            self.hid.double_click(sx, sy)
            self._emit_rumble(0.25, 0.35, 45)
            self._set_toast("极速购买商店卡牌")
            return

        if (
            self._current_scene == GameScene.BATTLE_PREP
            and self._focus_zone in (FocusZone.BENCH, FocusZone.BOARD)
            and self.deploy_fsm.state == DeployState.IDLE
        ):
            # 在整备区/棋盘按 RT+A 主动单击查看角色详情，并自动切换至选中角色特写详情态
            u, v = self._current_snap_center()
            self._click_norm(u, v, viewport)
            self._unit_tab_idx = 0
            self._set_unit_selected(True)
            self._emit_rumble(0.18, 0.25, 35)
            self._set_toast("查看角色详情（左右切标签 · X 出售 · Y 撤回 · B 退出）")
            return

        if self._current_scene == GameScene.MODAL_CHOICE and not self._choice_eye_hidden:
            u, v = self._choice_mapper.slot_center(
                self._choice_idx, count=self._choice_count
            )
            sx, sy = viewport.norm_to_screen(u, v)
            self.hid.double_click(sx, sy)
            self._emit_rumble(0.30, 0.40, 50)
            self._set_toast(f"极速确认第 {self._choice_idx + 1} 张策略卡")

    def _handle_in_battle_lt_chord(
        self, button: GamepadButton, viewport: ViewportRect
    ) -> None:
        if button == GamepadButton.LB and self._current_scene == GameScene.BATTLE_PREP:
            target_open = not self._shop_open
            self._set_shop_open(target_open, viewport, force=True, announce=True)
            if target_open and not self.deploy_fsm.is_active:
                self._focus_zone = FocusZone.SHOP
            elif not target_open and self._focus_zone == FocusZone.SHOP:
                self._focus_zone = FocusZone.BENCH
            self._emit_rumble(0.18, 0.24, 40)
            return

        if button == GamepadButton.DPAD_LEFT:
            self._trigger_action("covenant_panel", "covenant_panel", None, viewport)
            self._set_toast("切换盟约详情面板")
            return

        if button == GamepadButton.DPAD_UP:
            self._trigger_action("enemy_preview", "enemy_preview", "w", viewport)
            self._set_toast("切换敌方波次与路线预览")
            return

        if button == GamepadButton.DPAD_DOWN:
            self._trigger_action("emote_wheel", "emote_wheel", "z", viewport)
            self._set_toast("呼出联机互动表情轮盘")
            return

        if button == GamepadButton.DPAD_RIGHT:
            self._teammate_cycle_pos = (self._teammate_cycle_pos % 4) + 1
            self._switch_teammate_view(self._teammate_cycle_pos, viewport)
            return

        if button in _DIRECT_TEAMMATE_BUTTONS:
            target_mate = _DIRECT_TEAMMATE_BUTTONS[button]
            self._teammate_cycle_pos = target_mate
            self._switch_teammate_view(target_mate, viewport)
            return

        if button == GamepadButton.RB:
            if not getattr(self.keymap, "allow_retreat_chord", True):
                self._set_toast("已在配置中禁用一键撤退")
                return
            self._trigger_retreat(viewport)
            return

        if button == GamepadButton.START:
            self._click_button("pause_menu", viewport)
            self._set_toast("打开局内暂停菜单")

    def _switch_teammate_view(self, index: int, viewport: ViewportRect) -> None:
        idx = max(1, min(4, index))
        self._selected_teammate = idx
        if not self._click_button(f"teammate_{idx}", viewport):
            self.hid.tap_key(str(idx))
        self._emit_rumble(0.18, 0.25, 35)
        self._set_toast(f"切换队友 {idx} 号位视角")

    def _wait_ui_transition(self, duration_ms: int = 220) -> None:
        """等待游戏端面板/特写过渡动画挂载完毕（单测 MockVirtualHID 无阻塞）。"""
        wait_fn = getattr(self.hid, "wait_ms", None)
        if callable(wait_fn):
            wait_fn(duration_ms)

    def _trigger_retreat(self, viewport: ViewportRect) -> None:
        retreat_key = self._get_shortcut("retreat_unit", "q") or "q"
        if self._unit_selected:
            self.hid.tap_key(retreat_key)
            self._set_unit_selected(False)
        elif (
            self.deploy_fsm.state == DeployState.HOLDING_UNIT
            and self.deploy_fsm.holding_source is not None
            and self.deploy_fsm.holding_source.zone == "board"
        ):
            sx, sy = viewport.norm_to_screen(*self.deploy_fsm.holding_source.norm_pos)
            self.hid.click(sx, sy, button="left")
            self._wait_ui_transition(220)
            self.hid.tap_key(retreat_key)
            self.deploy_fsm.reset()
        elif self._focus_zone == FocusZone.BOARD and not self._free_cursor_active:
            u, v = self._board_mapper.cell_center(self._board_row, self._board_col)
            self._click_norm(u, v, viewport)
            self._wait_ui_transition(220)
            self.hid.tap_key(retreat_key)
            self._set_unit_selected(False)
        else:
            self.hid.tap_key(retreat_key)
        self._emit_rumble(0.35, 0.40, 55)
        self._set_toast(f"撤退场上干员 ({retreat_key.upper()})")

    # ------------------------------------------------------------------
    # 常规按键与导航路由
    # ------------------------------------------------------------------

    def _handle_button_action(
        self,
        button: GamepadButton,
        viewport: ViewportRect,
        *,
        is_repeat: bool = False,
    ) -> None:
        # 十字键或 LB/RB 随时将自由光标无缝切回吸附槽位模式
        if (
            self._free_cursor_active
            and self._current_scene != GameScene.UNKNOWN_FREE_CURSOR
            and (button in _DPAD_TO_DIR or button in (GamepadButton.LB, GamepadButton.RB))
        ):
            self._free_cursor_active = False

        # 纯自由光标模式处理
        if self._free_cursor_active or self._current_scene == GameScene.UNKNOWN_FREE_CURSOR:
            self._handle_free_cursor_button(button, viewport, is_repeat=is_repeat)
            return

        if self._current_scene in _MENU_SCENES:
            self._handle_menu_button(button, viewport, is_repeat=is_repeat)
            return

        if self._current_scene == GameScene.MODAL_CHOICE:
            self._handle_modal_choice_button(button, viewport, is_repeat=is_repeat)
            return

        if self._current_scene == GameScene.BATTLE_COMBAT:
            self._handle_battle_combat_button(button, viewport, is_repeat=is_repeat)
            return

        if self._current_scene == GameScene.BATTLE_PREP:
            self._handle_battle_prep_button(button, viewport, is_repeat=is_repeat)

    def _handle_free_cursor_button(
        self,
        button: GamepadButton,
        viewport: ViewportRect,
        *,
        is_repeat: bool,
    ) -> None:
        if is_repeat:
            return
        if button == GamepadButton.A:
            self._click_norm(self._cursor_norm[0], self._cursor_norm[1], viewport)
        elif button == GamepadButton.B:
            self._trigger_action("back", "back", "esc", viewport)
        elif button == GamepadButton.START:
            if self._current_scene == GameScene.BATTLE_PREP and self.keymap.ready_hold_ms == 0:
                self._trigger_battle_ready(viewport)
            elif self._current_scene in _MENU_SCENES:
                self._click_button("start_action", viewport)
        elif self._current_scene == GameScene.BATTLE_PREP:
            if button == GamepadButton.X:
                self._trigger_action("shop_refresh", "shop_refresh", "d", viewport)
                self._set_toast("刷新商店")
            elif button == GamepadButton.Y:
                self._trigger_action("shop_upgrade", "shop_upgrade", None, viewport)
                self._set_toast("升级商店等级")
            elif button == GamepadButton.RS:
                self._trigger_action("shop_freeze", "shop_freeze", "s", viewport)
                self._set_toast("切换商店冻结")
            elif button == GamepadButton.LS:
                self._trigger_action("emote_wheel", "emote_wheel", "z", viewport)
                self._set_toast("呼出联机互动表情轮盘")

    # ------------------------------------------------------------------
    # 局外菜单、准备、编队与结算页处理
    # ------------------------------------------------------------------

    def _handle_menu_button(
        self,
        button: GamepadButton,
        viewport: ViewportRect,
        *,
        is_repeat: bool,
    ) -> None:
        scene = self._current_scene
        mapper = self._menu_mappers[scene]
        cur_id = self._menu_node_ids.get(scene) or mapper.default_node_id()

        if button in _DPAD_TO_DIR:
            if cur_id is not None:
                next_id = mapper.navigate(cur_id, _DPAD_TO_DIR[button])
                self._menu_node_ids[scene] = next_id
            return

        if button in (GamepadButton.LB, GamepadButton.RB):
            delta = -1 if button == GamepadButton.LB else 1
            if scene == GameScene.SQUAD_EDIT:
                self._cycle_squad_skill_or_module(delta, viewport)
            elif cur_id is not None:
                self._menu_node_ids[scene] = mapper.cycle(cur_id, delta)
            return

        if is_repeat:
            return

        if button == GamepadButton.A:
            if cur_id is not None and mapper.get_node(cur_id) is not None:
                u, v = mapper.node_center(cur_id)
                self._click_norm(u, v, viewport)
            else:
                self._click_button("start_action", viewport)
            return

        if button == GamepadButton.B:
            self._trigger_action("back", "back", "esc", viewport)
            return

        if button == GamepadButton.START:
            if scene == GameScene.SETTLEMENT:
                target = mapper.find_by_keywords(("continue", "claim", "确认", "继续"))
                if target is not None:
                    self._menu_node_ids[scene] = target
                    u, v = mapper.node_center(target)
                    self._click_norm(u, v, viewport)
                    return
            self._click_button("start_action", viewport)
            return

        if button == GamepadButton.X:
            if scene == GameScene.ACTIVITY_HUB:
                self._focus_and_click_node(
                    scene, ("squad", "formation", "edit", "编队", "甄选"), viewport
                )
            elif scene == GameScene.PRE_MATCH_SETUP:
                self._focus_and_click_node(
                    scene, ("squad", "edit", "编队"), viewport
                )
            elif scene == GameScene.SQUAD_EDIT:
                self._focus_and_click_node(
                    scene, ("filter", "sort", "筛选", "排序"), viewport
                )
            return

        if button == GamepadButton.Y:
            if scene == GameScene.ACTIVITY_HUB:
                self._focus_and_click_node(
                    scene, ("reward", "mission", "奖励", "任务"), viewport
                )
            elif scene == GameScene.PRE_MATCH_SETUP:
                self._focus_and_click_node(
                    scene, ("room", "ready", "coop", "联机", "准备"), viewport
                )
            elif scene == GameScene.SQUAD_EDIT:
                self._focus_and_click_node(
                    scene, ("friend", "support", "助战", "好友"), viewport
                )

    def _focus_and_click_node(
        self,
        scene: GameScene,
        keywords: tuple[str, ...],
        viewport: ViewportRect,
    ) -> bool:
        mapper = self._menu_mappers[scene]
        target_id = mapper.find_by_keywords(keywords)
        if target_id is None:
            return False
        self._menu_node_ids[scene] = target_id
        u, v = mapper.node_center(target_id)
        self._click_norm(u, v, viewport)
        node = mapper.get_node(target_id)
        if node is not None:
            self._set_toast(f"快捷跳转：{node.label}")
        return True

    def _cycle_squad_skill_or_module(self, delta: int, viewport: ViewportRect) -> None:
        self._squad_slot_idx = (self._squad_slot_idx + delta) % len(
            _SQUAD_SKILL_BUTTON_KEYS
        )
        btn_key, node_id, label = _SQUAD_SKILL_BUTTON_KEYS[self._squad_slot_idx]
        mapper = self._menu_mappers[GameScene.SQUAD_EDIT]
        if mapper.get_node(node_id) is not None:
            self._menu_node_ids[GameScene.SQUAD_EDIT] = node_id
        self._click_button(btn_key, viewport)
        self._set_toast(f"切换至{label}")

    # ------------------------------------------------------------------
    # 三选一弹窗处理
    # ------------------------------------------------------------------

    def _handle_modal_choice_button(
        self,
        button: GamepadButton,
        viewport: ViewportRect,
        *,
        is_repeat: bool,
    ) -> None:
        # 查看战场状态下忽略选卡输入，防止误点背景
        if self._choice_eye_hidden:
            return

        if button in (GamepadButton.DPAD_LEFT, GamepadButton.LB):
            self._choice_idx, _ = self._choice_mapper.step(
                self._choice_idx, -1, wrap=True, count=self._choice_count
            )
            return

        if button in (GamepadButton.DPAD_RIGHT, GamepadButton.RB):
            self._choice_idx, _ = self._choice_mapper.step(
                self._choice_idx, 1, wrap=True, count=self._choice_count
            )
            return

        if is_repeat:
            return

        if button == GamepadButton.A:
            u, v = self._choice_mapper.slot_center(
                self._choice_idx, count=self._choice_count
            )
            self._click_norm(u, v, viewport)
            self._emit_rumble(0.22, 0.30, 40)

    # ------------------------------------------------------------------
    # 局内作战期处理
    # ------------------------------------------------------------------

    def _handle_unit_selected_button(
        self,
        button: GamepadButton,
        viewport: ViewportRect,
        *,
        is_repeat: bool,
    ) -> None:
        """选取角色特写详情态专属操控：切换左侧特质/技能/特性/天赋，X 出售，Y 撤退，B 点空白退出。"""
        if button in (GamepadButton.DPAD_LEFT, GamepadButton.LB):
            self._unit_tab_idx = (self._unit_tab_idx - 1) % len(_UNIT_DETAIL_TABS)
            u, v = self._unit_detail_tab_center(self._unit_tab_idx)
            self._click_norm(u, v, viewport)
            return

        if button in (GamepadButton.DPAD_RIGHT, GamepadButton.RB):
            self._unit_tab_idx = (self._unit_tab_idx + 1) % len(_UNIT_DETAIL_TABS)
            u, v = self._unit_detail_tab_center(self._unit_tab_idx)
            self._click_norm(u, v, viewport)
            return

        if is_repeat:
            return

        if button == GamepadButton.A:
            u, v = self._unit_detail_tab_center(self._unit_tab_idx)
            self._click_norm(u, v, viewport)
            return

        if button == GamepadButton.B:
            self._click_safe_blank(viewport)
            self._set_unit_selected(False)
            self._set_toast("已退出角色详情（恢复标准视角）")
            return

        if button == GamepadButton.X and self._current_scene == GameScene.BATTLE_PREP:
            sell_key = self._get_shortcut("sell_board", "x") or "x"
            self.hid.tap_key(sell_key)
            self._set_unit_selected(False)
            self._emit_rumble(0.30, 0.40, 55)
            self._set_toast(f"已出售选中角色 ({sell_key.upper()})")
            return

        if button == GamepadButton.Y:
            retreat_key = self._get_shortcut("retreat_unit", "q") or "q"
            self.hid.tap_key(retreat_key)
            self._set_unit_selected(False)
            self._emit_rumble(0.30, 0.35, 50)
            self._set_toast(f"已撤回角色 ({retreat_key.upper()})")

    def _handle_battle_combat_button(
        self,
        button: GamepadButton,
        viewport: ViewportRect,
        *,
        is_repeat: bool,
    ) -> None:
        if self._unit_selected:
            self._handle_unit_selected_button(button, viewport, is_repeat=is_repeat)
            return

        if button in _DPAD_TO_DIR:
            self._focus_zone = FocusZone.BOARD
            self._board_row, self._board_col, _ = self._board_mapper.move(
                self._board_row, self._board_col, _DPAD_TO_DIR[button], wrap_cols=False
            )
            return

        if is_repeat:
            return

        if button == GamepadButton.A:
            u, v = self._board_mapper.cell_center(self._board_row, self._board_col)
            self._click_norm(u, v, viewport)
            self._unit_tab_idx = 0
            self._set_unit_selected(True)
        elif button == GamepadButton.B:
            self._click_safe_blank(viewport)
            if self._unit_selected:
                self._set_unit_selected(False)
        elif button == GamepadButton.LS:
            self._trigger_action("emote_wheel", "emote_wheel", "z", viewport)
            self._set_toast("呼出联机互动表情轮盘")

    # ------------------------------------------------------------------
    # 局内休整期处理（商店 / 整备 / 棋盘 / 装备 + 部署状态机）
    # ------------------------------------------------------------------

    def _handle_battle_prep_button(
        self,
        button: GamepadButton,
        viewport: ViewportRect,
        *,
        is_repeat: bool,
    ) -> None:
        fsm_state = self.deploy_fsm.state

        # 1. 定朝向态优先拦截输入
        if fsm_state == DeployState.AIMING_DIRECTION:
            if button in _DPAD_TO_DIR and not is_repeat:
                direction = _DPAD_TO_DIR[button]
                self.deploy_fsm.set_preview_direction(direction)
                self.deploy_fsm.confirm_direction(direction, viewport)
                self._set_toast(f"定朝向完成：{self._dir_label(direction)}")
            elif button == GamepadButton.A and not is_repeat:
                if self.deploy_fsm.aim_preview_dir is not None:
                    direction = self.deploy_fsm.aim_preview_dir
                    self.deploy_fsm.confirm_direction(direction, viewport)
                    self._set_toast(f"定朝向完成：{self._dir_label(direction)}")
            elif button == GamepadButton.B and not is_repeat:
                self.deploy_fsm.cancel(viewport)
                self._set_toast("已取消干员落位")
            return

        # 1b. 选取角色详情特写态优先拦截（避免特写放大时棋盘框选错位）
        if fsm_state == DeployState.IDLE and self._unit_selected:
            self._handle_unit_selected_button(button, viewport, is_repeat=is_repeat)
            return

        # 2. LB / RB 分区切换
        if button in (GamepadButton.LB, GamepadButton.RB):
            delta = -1 if button == GamepadButton.LB else 1
            self._switch_battle_prep_zone(delta, viewport)
            return

        # 3. 十字键区内外导航
        if button in _DPAD_TO_DIR:
            self._navigate_battle_prep(_DPAD_TO_DIR[button], viewport)
            return

        if is_repeat:
            return

        # 4. 抓取持有态功能按键
        if fsm_state in (DeployState.HOLDING_UNIT, DeployState.HOLDING_EQUIP):
            self._handle_holding_button(button, viewport)
            return

        # 5. 休整期常规空闲浏览态按键
        if button == GamepadButton.A:
            self._handle_idle_prep_a(viewport)
        elif button == GamepadButton.B:
            if self._focus_zone == FocusZone.SHOP and self._shop_open:
                self._set_shop_open(False, viewport, announce=True)
                self._focus_zone = FocusZone.BENCH
            else:
                self._click_safe_blank(viewport)
        elif button == GamepadButton.X:
            if not self._shop_open and getattr(self.keymap, "auto_toggle_shop", True):
                self._set_shop_open(True, viewport, announce=False)
                self._wait_ui_transition(220)
            self._trigger_action("shop_refresh", "shop_refresh", "d", viewport)
            self._emit_rumble(0.18, 0.22, 35)
            self._set_toast("刷新商店 (D)")
        elif button == GamepadButton.Y:
            if not self._shop_open and getattr(self.keymap, "auto_toggle_shop", True):
                self._set_shop_open(True, viewport, announce=False)
                self._wait_ui_transition(220)
            self._trigger_action("shop_upgrade", "shop_upgrade", None, viewport)
            self._emit_rumble(0.25, 0.30, 45)
            self._set_toast("升级商店等级")
        elif button == GamepadButton.RS:
            if not self._shop_open and getattr(self.keymap, "auto_toggle_shop", True):
                self._set_shop_open(True, viewport, announce=False)
                self._wait_ui_transition(220)
            self._trigger_action("shop_freeze", "shop_freeze", "s", viewport)
            self._emit_rumble(0.18, 0.22, 35)
            self._set_toast("切换商店冻结 (S)")
        elif button == GamepadButton.LS:
            self._trigger_action("emote_wheel", "emote_wheel", "z", viewport)
            self._set_toast("呼出联机互动表情轮盘")
        elif button == GamepadButton.START and self.keymap.ready_hold_ms == 0:
            self._trigger_battle_ready(viewport)

    def _switch_battle_prep_zone(
        self, delta: int, viewport: ViewportRect
    ) -> None:
        fsm_state = self.deploy_fsm.state
        if fsm_state in (DeployState.HOLDING_UNIT, DeployState.HOLDING_EQUIP):
            # 抓取状态下在整备区与棋盘之间快速跳区
            target = (
                FocusZone.BOARD
                if self._focus_zone != FocusZone.BOARD
                else FocusZone.BENCH
            )
            self._enter_battle_prep_zone(target, viewport)
            return

        if self._focus_zone not in _BATTLE_PREP_ZONE_ORDER:
            self._enter_battle_prep_zone(FocusZone.BENCH, viewport)
            return
        idx = _BATTLE_PREP_ZONE_ORDER.index(self._focus_zone)
        next_zone = _BATTLE_PREP_ZONE_ORDER[
            (idx + delta) % len(_BATTLE_PREP_ZONE_ORDER)
        ]
        self._enter_battle_prep_zone(next_zone, viewport)

    def _navigate_battle_prep(
        self, direction: Direction, viewport: ViewportRect
    ) -> None:
        is_holding = self.deploy_fsm.state in (
            DeployState.HOLDING_UNIT,
            DeployState.HOLDING_EQUIP,
        )
        shop_above_board = (
            self._shop_mapper.zone.start_v < self._board_mapper.quad.top_left[1]
        )
        equip_vertical = self._equip_mapper.is_vertical
        equip_on_right = self._equip_mapper.zone.start_u >= 0.5

        if self._focus_zone == FocusZone.SHOP:
            new_idx, exit_dir = self._shop_mapper.move(
                self._shop_idx, direction, wrap=True
            )
            self._shop_idx = new_idx
            if exit_dir == "down" and shop_above_board:
                cu, cv = self._shop_mapper.slot_center(self._shop_idx)
                self._enter_battle_prep_zone(FocusZone.BOARD, viewport)
                _, col = self._board_mapper.nearest_cell(cu, cv)
                self._board_row = 0
                self._board_col = col
            elif exit_dir == "up" and not shop_above_board:
                cu, cv = self._shop_mapper.slot_center(self._shop_idx)
                self._enter_battle_prep_zone(FocusZone.BENCH, viewport)
                self._bench_idx = self._bench_mapper.nearest_slot(cu, cv)
            return

        if self._focus_zone == FocusZone.BENCH:
            new_idx, exit_dir = self._bench_mapper.move(
                self._bench_idx, direction, wrap=True
            )
            self._bench_idx = new_idx
            if exit_dir == "up":
                cu, cv = self._bench_mapper.slot_center(self._bench_idx)
                if not equip_vertical and not is_holding:
                    self._enter_battle_prep_zone(FocusZone.EQUIPMENT, viewport)
                    self._equip_idx = self._equip_mapper.nearest_slot(cu, cv)
                else:
                    self._enter_battle_prep_zone(FocusZone.BOARD, viewport)
                    _, col = self._board_mapper.nearest_cell(cu, cv)
                    self._board_row = self._board_mapper.rows - 1
                    self._board_col = col
            elif exit_dir == "down" and not is_holding:
                cu, cv = self._bench_mapper.slot_center(self._bench_idx)
                self._enter_battle_prep_zone(FocusZone.SHOP, viewport)
                self._shop_idx = self._shop_mapper.nearest_slot(cu, cv)
            return

        if self._focus_zone == FocusZone.BOARD:
            new_r, new_c, exit_dir = self._board_mapper.move(
                self._board_row, self._board_col, direction, wrap_cols=False
            )
            self._board_row = new_r
            self._board_col = new_c
            if exit_dir == "down":
                cu, cv = self._board_mapper.cell_center(self._board_row, self._board_col)
                if not equip_vertical and not is_holding:
                    self._enter_battle_prep_zone(FocusZone.EQUIPMENT, viewport)
                    self._equip_idx = self._equip_mapper.nearest_slot(cu, cv)
                else:
                    self._enter_battle_prep_zone(FocusZone.BENCH, viewport)
                    self._bench_idx = self._bench_mapper.nearest_slot(cu, cv)
            elif exit_dir == "up" and shop_above_board and not is_holding:
                cu, cv = self._board_mapper.cell_center(self._board_row, self._board_col)
                self._enter_battle_prep_zone(FocusZone.SHOP, viewport)
                self._shop_idx = self._shop_mapper.nearest_slot(cu, cv)
            elif exit_dir == "right" and equip_vertical and equip_on_right and not is_holding:
                cu, cv = self._board_mapper.cell_center(self._board_row, self._board_col)
                self._enter_battle_prep_zone(FocusZone.EQUIPMENT, viewport)
                self._equip_idx = self._equip_mapper.nearest_slot(cu, cv)
            elif exit_dir == "left" and equip_vertical and not equip_on_right and not is_holding:
                cu, cv = self._board_mapper.cell_center(self._board_row, self._board_col)
                self._enter_battle_prep_zone(FocusZone.EQUIPMENT, viewport)
                self._equip_idx = self._equip_mapper.nearest_slot(cu, cv)
            return

        if self._focus_zone == FocusZone.EQUIPMENT:
            new_idx, exit_dir = self._equip_mapper.move(
                self._equip_idx, direction, wrap=True
            )
            self._equip_idx = new_idx
            if not equip_vertical:
                if exit_dir == "up":
                    cu, cv = self._equip_mapper.slot_center(self._equip_idx)
                    self._enter_battle_prep_zone(FocusZone.BOARD, viewport)
                    _, col = self._board_mapper.nearest_cell(cu, cv)
                    self._board_row = self._board_mapper.rows - 1
                    self._board_col = col
                elif exit_dir == "down":
                    cu, cv = self._equip_mapper.slot_center(self._equip_idx)
                    self._enter_battle_prep_zone(FocusZone.BENCH, viewport)
                    self._bench_idx = self._bench_mapper.nearest_slot(cu, cv)
            elif (equip_on_right and exit_dir == "left") or (
                not equip_on_right and exit_dir == "right"
            ):
                cu, cv = self._equip_mapper.slot_center(self._equip_idx)
                self._enter_battle_prep_zone(FocusZone.BOARD, viewport)
                row, _ = self._board_mapper.nearest_cell(cu, cv)
                self._board_row = row
                self._board_col = self._board_mapper.cols - 1 if equip_on_right else 0

    def _prepare_clean_view_for_drag(self, viewport: ViewportRect) -> None:
        """在抓取干员/装备前确保退出角色详情特写态并按需收起商店，恢复标准未缩放坐标系。"""
        if self._unit_selected:
            self._click_safe_blank(viewport)
            self._set_unit_selected(False)
        if self._shop_open and getattr(self.keymap, "auto_toggle_shop", True):
            self._set_shop_open(False, viewport, announce=False)

    def _handle_idle_prep_a(self, viewport: ViewportRect) -> None:
        if self._focus_zone == FocusZone.SHOP:
            if getattr(self.keymap, "auto_toggle_shop", True) and not self._shop_open:
                self._set_shop_open(True, viewport, announce=False)
                self._wait_ui_transition(220)
            u, v = self._shop_mapper.slot_center(self._shop_idx)
            self._click_norm(u, v, viewport)
            return

        if self._focus_zone == FocusZone.BENCH:
            self._prepare_clean_view_for_drag(viewport)
            pos = self._bench_mapper.slot_center(self._bench_idx)
            self.deploy_fsm.pick_unit(
                "bench",
                pos,
                viewport,
                slot_index=self._bench_idx,
                click_on_pick=False,
            )
            self._set_toast(f"已抓取整备区 [{self._bench_idx + 1}] 干员")
            return

        if self._focus_zone == FocusZone.BOARD:
            self._prepare_clean_view_for_drag(viewport)
            pos = self._board_mapper.cell_center(self._board_row, self._board_col)
            self.deploy_fsm.pick_unit(
                "board",
                pos,
                viewport,
                board_cell=(self._board_row, self._board_col),
                click_on_pick=False,
            )
            self._set_toast(
                f"已抓取场上 [{self._board_row + 1},{self._board_col + 1}] 干员"
            )
            return

        if self._focus_zone == FocusZone.EQUIPMENT:
            self._prepare_clean_view_for_drag(viewport)
            pos = self._equip_mapper.slot_center(self._equip_idx)
            self.deploy_fsm.pick_equip(
                pos,
                viewport,
                slot_index=self._equip_idx,
                click_on_pick=False,
            )
            self._set_toast(f"已抓取装备 [{self._equip_idx + 1}]")

    def _handle_holding_button(
        self, button: GamepadButton, viewport: ViewportRect
    ) -> None:
        fsm_state = self.deploy_fsm.state

        if button == GamepadButton.B:
            self.deploy_fsm.cancel(viewport)
            self._set_toast("已取消抓取")
            return

        if button == GamepadButton.X:
            if self.deploy_fsm.sell_holding(viewport):
                self._set_toast("已出售目标")
            return

        if button == GamepadButton.Y:
            if self.deploy_fsm.gift_holding(self._selected_teammate, viewport):
                self._set_toast(f"已赠予联机队友 {self._selected_teammate}")
            return

        if button == GamepadButton.A:
            if fsm_state == DeployState.HOLDING_UNIT:
                if self._focus_zone == FocusZone.BENCH:
                    pos = self._bench_mapper.slot_center(self._bench_idx)
                    self.deploy_fsm.place_on_bench(
                        pos, viewport, slot_index=self._bench_idx
                    )
                    self._set_toast(f"已移至整备区 [{self._bench_idx + 1}]")
                elif self._focus_zone == FocusZone.BOARD:
                    pos = self._board_mapper.cell_center(
                        self._board_row, self._board_col
                    )
                    self.deploy_fsm.place_on_board(
                        (self._board_row, self._board_col), pos, viewport
                    )
                    if self.deploy_fsm.state == DeployState.AIMING_DIRECTION:
                        self._set_toast("请推摇杆或十字键确定干员朝向")
            elif fsm_state == DeployState.HOLDING_EQUIP:
                if self._focus_zone == FocusZone.BOARD:
                    pos = self._board_mapper.cell_center(
                        self._board_row, self._board_col
                    )
                    self.deploy_fsm.equip_target(pos, viewport)
                    self._set_toast("已为场上干员穿戴装备")
                elif self._focus_zone == FocusZone.BENCH:
                    pos = self._bench_mapper.slot_center(self._bench_idx)
                    self.deploy_fsm.equip_target(pos, viewport)
                    self._set_toast("已为整备区干员穿戴装备")

    # ------------------------------------------------------------------
    # HUD 状态快照构造
    # ------------------------------------------------------------------

    @staticmethod
    def _dir_label(direction: Direction) -> str:
        return {"up": "上", "down": "下", "left": "左", "right": "右"}[direction]

    def _build_focus_label(self) -> str:
        if self._free_cursor_active or self._focus_zone == FocusZone.FREE_CURSOR:
            return "自由光标"
        if (
            self._unit_selected
            and self._current_scene in (GameScene.BATTLE_PREP, GameScene.BATTLE_COMBAT)
            and not self.deploy_fsm.is_active
        ):
            _, tab_label, _, _ = _UNIT_DETAIL_TABS[self._unit_tab_idx % len(_UNIT_DETAIL_TABS)]
            return f"角色详情 · {tab_label}"
        if self._focus_zone == FocusZone.MENU_NODES:
            mapper = self._menu_mappers.get(self._current_scene)
            node_id = self._menu_node_ids.get(self._current_scene)
            if mapper is not None and node_id is not None:
                node = mapper.get_node(node_id)
                if node is not None:
                    return node.label
            return "菜单导航"
        if self._focus_zone == FocusZone.SHOP:
            return f"商店 [{self._shop_idx + 1}/{self._shop_mapper.count}]"
        if self._focus_zone == FocusZone.BENCH:
            return f"整备区 [{self._bench_idx + 1}/{self._bench_mapper.count}]"
        if self._focus_zone == FocusZone.BOARD:
            return f"棋盘 [{self._board_row + 1},{self._board_col + 1}]"
        if self._focus_zone == FocusZone.EQUIPMENT:
            return f"装备栏 [{self._equip_idx + 1}/{self._equip_mapper.count}]"
        if self._focus_zone == FocusZone.CHOICE_CARDS:
            if self._choice_eye_hidden:
                return "查看战场"
            return f"三选一 [{self._choice_idx + 1}/{self._choice_count}]"
        return ""

    def _build_prompts(self) -> list[PromptItem]:
        if self._paused:
            return [
                PromptItem("LT+Back", "恢复操控"),
                PromptItem("Back+Start", "校准"),
            ]

        if self._free_cursor_active or self._current_scene == GameScene.UNKNOWN_FREE_CURSOR:
            items = [
                PromptItem("LS", "移动光标"),
                PromptItem("A", "点击"),
                PromptItem("RT+A", "双击"),
                PromptItem("B", "返回"),
                PromptItem("RS", "滚轮"),
            ]
            if self._current_scene != GameScene.UNKNOWN_FREE_CURSOR:
                recover_key = (
                    "LS/十字键"
                    if getattr(self.keymap, "left_stick_mode", "dpad") == "dpad"
                    else "D-Pad"
                )
                items.append(PromptItem(recover_key, "恢复吸附"))
            return items

        nav_key = (
            "LS/D-Pad"
            if getattr(self.keymap, "left_stick_mode", "dpad") == "dpad"
            else "D-Pad"
        )
        scene = self._current_scene
        if scene == GameScene.ACTIVITY_HUB:
            return [
                PromptItem(nav_key, "切换节点"),
                PromptItem("A", "确认"),
                PromptItem("X", "编队"),
                PromptItem("Y", "奖励"),
                PromptItem("Start", "开始模拟"),
                PromptItem("B", "返回"),
            ]
        if scene == GameScene.PRE_MATCH_SETUP:
            return [
                PromptItem(nav_key, "选分队"),
                PromptItem("LB/RB", "切换"),
                PromptItem("A", "确认"),
                PromptItem("Start", "进入对局"),
                PromptItem("B", "返回"),
            ]
        if scene == GameScene.SQUAD_EDIT:
            return [
                PromptItem(nav_key, "选干员"),
                PromptItem("RS", "滚轮浏览"),
                PromptItem("LB/RB", "技能/模组"),
                PromptItem("Y", "好友助战"),
                PromptItem("Start", "确认出击"),
                PromptItem("B", "返回"),
            ]
        if scene == GameScene.SETTLEMENT:
            return [
                PromptItem(nav_key, "切换项"),
                PromptItem("A", "确认"),
                PromptItem("Start", "继续"),
                PromptItem("B", "返回"),
            ]
        if scene == GameScene.MODAL_CHOICE:
            if self._choice_eye_hidden:
                return [PromptItem("松开 LT", "恢复三选一弹窗")]
            return [
                PromptItem(nav_key, "选卡"),
                PromptItem("A", "选中"),
                PromptItem("RT+A", "极速拿牌"),
                PromptItem("按住 LT", "查看战场"),
            ]
        # 选取角色特写详情态专属底部提示条
        if (
            self._unit_selected
            and scene in (GameScene.BATTLE_PREP, GameScene.BATTLE_COMBAT)
            and not self.deploy_fsm.is_active
        ):
            items = [
                PromptItem(nav_key, "切换特质/技能/特性/天赋"),
                PromptItem("B", "退出详情"),
            ]
            if scene == GameScene.BATTLE_PREP:
                items.insert(1, PromptItem("X", "出售(X)"))
                items.insert(2, PromptItem("Y", "撤回(Q)"))
            else:
                items.insert(1, PromptItem("Y", "撤退(Q)"))
            return items

        if scene == GameScene.BATTLE_COMBAT:
            return [
                PromptItem(nav_key, "浏览棋盘"),
                PromptItem("A", "查看详情"),
                PromptItem("LT+十字键", "盟约/敌情/表情/队友"),
                PromptItem("LT+RB", "撤退"),
                PromptItem("LT+Start", "暂停"),
            ]

        # BATTLE_PREP
        fsm_state = self.deploy_fsm.state
        if fsm_state == DeployState.AIMING_DIRECTION:
            return [
                PromptItem("摇杆/十字键", "确认朝向"),
                PromptItem("A", "确定预选"),
                PromptItem("B", "取消落位"),
            ]
        if fsm_state == DeployState.HOLDING_UNIT:
            return [
                PromptItem(nav_key, "选目标格"),
                PromptItem("LB/RB", "整备↔棋盘"),
                PromptItem("A", "部署/换位"),
                PromptItem("X", "出售"),
                PromptItem("Y", f"赠队友{self._selected_teammate}"),
                PromptItem("B", "取消"),
            ]
        if fsm_state == DeployState.HOLDING_EQUIP:
            return [
                PromptItem(nav_key, "选目标干员"),
                PromptItem("LB/RB", "棋盘↔整备"),
                PromptItem("A", "穿戴装备"),
                PromptItem("X", "出售"),
                PromptItem("Y", f"赠队友{self._selected_teammate}"),
                PromptItem("B", "取消"),
            ]

        # BATTLE_PREP + IDLE
        if self._focus_zone == FocusZone.SHOP:
            return [
                PromptItem("A", "选/买"),
                PromptItem("RT+A", "极速连买"),
                PromptItem("X", "刷新"),
                PromptItem("Y", "升本"),
                PromptItem("RS", "冻结"),
                PromptItem("B", "收起商店"),
                PromptItem("长按Start", "开战"),
            ]
        if self._focus_zone == FocusZone.BENCH:
            return [
                PromptItem("A", "抓取干员"),
                PromptItem("RT+A", "查详情"),
                PromptItem("LT+LB", "开/关商店"),
                PromptItem("X", "刷新"),
                PromptItem("LB/RB", "切区"),
                PromptItem("长按Start", "开战"),
            ]
        if self._focus_zone == FocusZone.BOARD:
            return [
                PromptItem("A", "抓取/换位"),
                PromptItem("RT+A", "查详情"),
                PromptItem("LT+LB", "开/关商店"),
                PromptItem("LT+RB", "撤退"),
                PromptItem("LB/RB", "切区"),
                PromptItem("长按Start", "开战"),
            ]
        return [
            PromptItem("A", "抓取装备"),
            PromptItem("LT+LB", "开/关商店"),
            PromptItem("X", "刷新"),
            PromptItem("LB/RB", "切区"),
            PromptItem("长按Start", "开战"),
        ]

    def _build_snapshot(self, *, visible: bool) -> HUDStateSnapshot:
        effective_zone = (
            FocusZone.FREE_CURSOR if self._free_cursor_active else self._focus_zone
        )
        cursor_norm = (
            self._cursor_norm
            if self._free_cursor_active or effective_zone == FocusZone.FREE_CURSOR
            else self._current_snap_center()
        )
        return HUDStateSnapshot(
            visible=visible,
            paused=self._paused,
            calibration_requested=self._calibration_requested,
            scene=self._current_scene,
            scene_locked_manual=self.scene_locked_manual,
            focus_zone=effective_zone,
            focus_label=self._build_focus_label(),
            focus_polygon_norm=self._current_snap_polygon(),
            holding_source_norm=self.deploy_fsm.holding_source_norm,
            cursor_norm=cursor_norm,
            free_cursor_active=self._free_cursor_active,
            deploy_state=self.deploy_fsm.state,
            aim_origin_norm=self.deploy_fsm.aim_origin_norm,
            aim_preview_dir=self.deploy_fsm.aim_preview_dir,
            ready_hold_progress=self._ready_hold_progress,
            prompts=self._build_prompts(),
            toast_message=self._toast_message,
            view_state=self._view_state.value,
            shop_open=self._shop_open,
            left_stick_mode=getattr(self.keymap, "left_stick_mode", "dpad"),
        )
