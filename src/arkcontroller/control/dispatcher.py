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

    SHOP = "shop"
    BENCH = "bench"
    BOARD = "board"
    EQUIPMENT = "equipment"
    STRATEGY = "strategy"
    TOP_INFO = "top_info"
    TEAMMATES = "teammates"
    EMOTE_PANEL = "emote_panel"
    CHOICE_CARDS = "choice_cards"
    CONFIRM_DIALOG = "confirm_dialog"
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

_HOLDING_EQUIP_ZONE_ORDER: tuple[FocusZone, ...] = (
    FocusZone.EQUIPMENT,
    FocusZone.BOARD,
    FocusZone.BENCH,
)

_HOLDING_STRATEGY_ZONE_ORDER: tuple[FocusZone, ...] = (
    FocusZone.STRATEGY,
    FocusZone.BOARD,
    FocusZone.BENCH,
    FocusZone.SHOP,
)

# Back 单击手动切层序列：None 表示恢复自动识别
_MANUAL_LAYER_CYCLE: tuple[GameScene | None, ...] = (
    None,
    GameScene.BATTLE_PREP,
    GameScene.MODAL_CHOICE,
    GameScene.MODAL_CONFIRM,
    GameScene.UNKNOWN_FREE_CURSOR,
)

_MANUAL_SCENE_TITLES: dict[GameScene, str] = {
    GameScene.BATTLE_PREP: "局内运营层",
    GameScene.MODAL_CHOICE: "弹窗决策层",
    GameScene.MODAL_CONFIRM: "回合确认层",
    GameScene.BATTLE_COMBAT: "作战观战层",
    GameScene.UNKNOWN_FREE_CURSOR: "自由鼠标层",
}

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
_SHOP_PENDING_BUY_TTL_SECONDS: float = 2.0


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

        # 界面镜头/缩放视图态（normal / shop_open / unit_selected）与商店动态栏位数（1本4槽/2本5槽/4本6槽）
        self._view_state: LayoutViewState = LayoutViewState.NORMAL
        self._shop_open: bool = False
        self._shop_slot_count: int = self.layout.shop_zone.count
        self._shop_pending_buy_idx: int | None = None
        self._shop_pending_buy_until: float = 0.0
        self._unit_selected: bool = False
        self._unit_detail_from_shop: bool = False
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
        self._rebuild_mappers()

        # 运行态开关
        self._paused: bool = False
        self._calibration_requested: bool = False

        # 场景与手动锁层状态（游戏外与未识别界面默认采用自由模拟鼠标）
        self._current_scene: GameScene = GameScene.UNKNOWN_FREE_CURSOR
        self._manual_cycle_idx: int = 0  # 0 表示自动识别（未手动锁层）
        self._choice_count: int = self._choice_mapper.count

        # 焦点区与各区索引
        self._focus_zone: FocusZone = FocusZone.FREE_CURSOR
        self._shop_idx: int = 0
        self._bench_idx: int = 0
        self._equip_idx: int = 0
        self._choice_idx: int = min(1, max(0, self._choice_count - 1))
        self._selected_choice_idx: int | None = None
        self._selected_choice_override_until: float = 0.0
        self._confirm_idx: int = 1  # 0=取消, 1=确认(默认), 2=本局不再提示
        self._confirm_has_dont_ask: bool = True
        self._covenant_count: int = self.layout.covenant_zone.default_count
        self._top_info_idx: int = 0
        self._has_teammates: bool = False
        self._teammate_focus_idx: int = 0
        self._emote_panel_open: bool = False
        self._emote_idx: int = 0
        self._emote_override_until: float = 0.0
        self._zone_before_emote: FocusZone = FocusZone.BOARD
        self._info_overlay: Literal["match_info", "enemy_preview"] | None = None
        self._board_row: int = self._board_mapper.rows // 2
        self._board_col: int = self._board_mapper.cols // 2

        # 自由光标与滚轮积分器
        self._free_cursor_active: bool = True
        self._cursor_norm: tuple[float, float] = (0.5, 0.5)
        self._last_cmd_cursor_px: tuple[int, int] | None = None
        self._scroll_v_accum: float = 0.0
        self._scroll_h_accum: float = 0.0

        # 角色详情标签索引、联机队友视角与决策弹窗收起状态
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
    def shop_slot_count(self) -> int:
        return self._shop_slot_count

    @property
    def unit_selected(self) -> bool:
        return self._unit_selected

    @property
    def selected_choice_index(self) -> int | None:
        return self._selected_choice_idx

    @property
    def covenant_count(self) -> int:
        return self._covenant_count

    @property
    def has_teammates(self) -> bool:
        return self._has_teammates

    @property
    def shop_pending_buy_index(self) -> int | None:
        return self._shop_pending_buy_idx

    @property
    def unit_detail_from_shop(self) -> bool:
        return self._unit_detail_from_shop

    @property
    def emote_panel_open(self) -> bool:
        return self._emote_panel_open

    @property
    def emote_index(self) -> int:
        return self._emote_idx

    @property
    def confirm_has_dont_ask(self) -> bool:
        return self._confirm_has_dont_ask

    def set_covenant_count(self, count: int) -> None:
        """设置顶部栏正下方居中对称羁绊槽位数（0~9）并安全同步顶部焦点索引。"""
        self._apply_covenant_count(count)

    def set_has_teammates(self, enabled: bool) -> None:
        """设置当前对局是否启用左侧多人联机队友头像栏。"""
        self._has_teammates = bool(enabled)
        if not self._has_teammates and self._focus_zone == FocusZone.TEAMMATES:
            self._focus_zone = FocusZone.BOARD

    def set_shop_slot_count(self, count: int) -> None:
        """设置当前商店栏位数（支持 4/5/6 槽向左等距延伸）并同步重建商店映射器。"""
        self._apply_shop_slot_count(count)

    def update_configs(self, keymap: KeymapConfig, layout: LayoutProfile) -> None:
        """热重载键位配置与 16:9 布局配置，并安全裁剪全部槽位索引。"""
        self.keymap = keymap
        self.layout = layout
        self._shop_slot_count = layout.shop_zone.count
        self._covenant_count = max(0, min(9, self._covenant_count))
        self.deploy_fsm.update_configs(keymap, layout)
        self._rebuild_mappers()

        self._shop_idx = self._shop_mapper.clamp_index(self._shop_idx)
        self._clear_shop_pending_buy()
        self._bench_idx = self._bench_mapper.clamp_index(self._bench_idx)
        self._equip_idx = self._equip_mapper.clamp_index(self._equip_idx)
        self._choice_count = max(1, min(self._choice_count, max(self._choice_mapper.count, self._choice_count)))
        self._choice_idx = max(0, min(self._choice_count - 1, self._choice_idx))
        if self._selected_choice_idx is not None and self._selected_choice_idx >= self._choice_count:
            self._selected_choice_idx = None
        top_items = self._top_info_items()
        self._top_info_idx = max(0, min(len(top_items) - 1, self._top_info_idx))
        self._teammate_focus_idx = max(0, min(3, self._teammate_focus_idx))
        self._emote_idx = max(0, min(5, self._emote_idx))
        self._confirm_idx = max(0, min(2 if self._confirm_has_dont_ask else 1, self._confirm_idx))
        self._board_row = max(0, min(self._board_mapper.rows - 1, self._board_row))
        self._board_col = max(0, min(self._board_mapper.cols - 1, self._board_col))

    def acknowledge_calibration_request(self) -> None:
        """由外部 UI 确认已响应校准面板呼出请求，清除挂起标志。"""
        self._calibration_requested = False

    def toggle_pause(self) -> bool:
        """切换手柄映射暂停状态并返回切换后的布尔值。"""
        self._paused = not self._paused
        if self._paused:
            self.deploy_fsm.reset()
            self._clear_shop_pending_buy()
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
        self._expire_shop_pending_buy_if_needed()

        # 最高优先级安全护栏：游戏窗口未找到或处于后台时，严禁向 HID 发送任何输入
        if viewport is None or not viewport.is_foreground:
            self._ready_hold_progress = 0.0
            self._last_cmd_cursor_px = None
            self._clear_shop_pending_buy()
            if self._choice_eye_hidden:
                self._choice_eye_hidden = False
            return self._build_snapshot(visible=False)

        # 同步自动识别场景（手动锁层或部署拖拽态下不强行打断）
        self._sync_scene(scene_result)

        # 自由鼠标态下同步真实物理光标位置（仅在首次入前台或物理鼠标实际移动 >=2px 时同步，避免微推摇杆亚像素累加被抹零）
        if self._free_cursor_active or self._current_scene == GameScene.UNKNOWN_FREE_CURSOR:
            self._sync_physical_cursor(viewport)

        # 逐条处理本帧手势事件
        for event in events:
            self._handle_event(event, viewport)

        return self._build_snapshot(visible=True)

    def _sync_physical_cursor(self, viewport: ViewportRect) -> None:
        """无损同步系统物理光标位置到自由鼠标归一化坐标（保护摇杆浮点亚像素累加不被取整锁死）。"""
        get_pos = getattr(self.hid, "get_cursor_pos", None)
        if not callable(get_pos):
            return
        pos = get_pos()
        if pos is None or len(pos) != 2:
            return
        px, py = int(pos[0]), int(pos[1])
        if not (
            viewport.client_left <= px < viewport.client_left + viewport.client_width
            and viewport.client_top <= py < viewport.client_top + viewport.client_height
        ):
            return
        if (
            self._last_cmd_cursor_px is None
            or abs(px - self._last_cmd_cursor_px[0]) >= 2
            or abs(py - self._last_cmd_cursor_px[1]) >= 2
        ):
            nu, nv = viewport.screen_to_norm(px, py)
            self._cursor_norm = (_clamp01(nu), _clamp01(nv))
            self._last_cmd_cursor_px = (px, py)

    # ------------------------------------------------------------------
    # 内部初始化与工具方法
    # ------------------------------------------------------------------

    def _rebuild_mappers(self) -> None:
        self._shop_mappers = {}
        self._bench_mappers = {}
        self._equip_mappers = {}
        self._board_mappers = {}
        for vs in LayoutViewState:
            self._shop_mappers[vs] = LinearZoneMapper(
                self.layout.resolve_shop_zone(vs, slot_count=self._shop_slot_count)
            )
            self._bench_mappers[vs] = LinearZoneMapper(self.layout.resolve_bench_zone(vs))
            self._equip_mappers[vs] = LinearZoneMapper(self.layout.resolve_equip_zone(vs))
            self._board_mappers[vs] = PerspectiveBoardMapper(
                self.layout.resolve_board_quad(vs)
            )
        self._choice_mapper = LinearZoneMapper(self.layout.choice_zone)
        self._sync_active_mappers()

    def _apply_shop_slot_count(self, new_count: int) -> None:
        """更新商店栏位数（4/5/6 槽向左延伸），顺移当前焦点索引以咬住原卡片位置。"""
        clamped = max(1, min(8, int(new_count)))
        if clamped == self._shop_slot_count:
            return
        delta = clamped - self._shop_slot_count
        self._shop_slot_count = clamped
        for vs in LayoutViewState:
            self._shop_mappers[vs] = LinearZoneMapper(
                self.layout.resolve_shop_zone(vs, slot_count=self._shop_slot_count)
            )
        self._shop_mapper = self._shop_mappers[self._view_state]
        self._shop_idx = max(0, min(self._shop_slot_count - 1, self._shop_idx + delta))
        self._clear_shop_pending_buy()

    def _clear_shop_pending_buy(self) -> None:
        """清空商店连按两下 A 的预选购买状态。"""
        self._shop_pending_buy_idx = None
        self._shop_pending_buy_until = 0.0

    def _expire_shop_pending_buy_if_needed(self) -> None:
        """超时自动清除商店预选购买状态（默认有效期 2.0 秒）。"""
        if (
            self._shop_pending_buy_idx is not None
            and time.monotonic() >= self._shop_pending_buy_until
        ):
            self._clear_shop_pending_buy()

    def _is_shop_pending_buy_for(self, idx: int) -> bool:
        """判断当前是否处于指定商店槽位有效期内的待二次按 A 购买态。"""
        return (
            self._shop_pending_buy_idx == idx
            and time.monotonic() < self._shop_pending_buy_until
        )

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
        """左键点击右侧策略头像正下方安全空白区以退出角色详情/羁绊浮窗（避免误买商店卡）。"""
        if not self._click_button("safe_blank", viewport):
            self._click_norm(0.955, 0.445, viewport, anchor="right_edge", button="left")

    @staticmethod
    def _rect_poly(
        cu: float, cv: float, w: float, h: float
    ) -> list[tuple[float, float]]:
        hw, hh = w * 0.5, h * 0.5
        return [
            (_clamp01(cu - hw), _clamp01(cv - hh)),
            (_clamp01(cu + hw), _clamp01(cv - hh)),
            (_clamp01(cu + hw), _clamp01(cv + hh)),
            (_clamp01(cu - hw), _clamp01(cv + hh)),
        ]

    def _apply_covenant_count(self, new_count: int) -> None:
        """更新顶部栏下方居中对称羁绊数量，并在顶部栏吸附时保持右放大镜等相对焦点稳定。"""
        clamped = max(0, min(9, int(new_count)))
        if clamped == self._covenant_count:
            return
        old_total = self._covenant_count + 2
        was_right_mag = self._top_info_idx == old_total - 1
        self._covenant_count = clamped
        new_total = self._covenant_count + 2
        if was_right_mag:
            self._top_info_idx = new_total - 1
        else:
            self._top_info_idx = max(0, min(new_total - 1, self._top_info_idx))

    def _top_info_items(
        self,
    ) -> list[
        tuple[
            Literal["match_info", "covenant", "enemy_preview"],
            str,
            tuple[float, float],
            str,
            list[tuple[float, float]],
        ]
    ]:
        """构建顶部信息与居中对称羁绊栏的从左至右吸附序列：[左放大镜, 羁绊1..N, 右放大镜]。"""
        left_pt = self.layout.buttons.get("covenant_panel")
        lu, lv = (left_pt.u, left_pt.v) if left_pt is not None else (0.265, 0.047)
        l_anchor = left_pt.anchor if left_pt is not None else "center_16_9"

        right_pt = self.layout.buttons.get("enemy_preview")
        ru, rv = (right_pt.u, right_pt.v) if right_pt is not None else (0.738, 0.048)
        r_anchor = right_pt.anchor if right_pt is not None else "center_16_9"

        items: list[
            tuple[
                Literal["match_info", "covenant", "enemy_preview"],
                str,
                tuple[float, float],
                str,
                list[tuple[float, float]],
            ]
        ] = [
            (
                "match_info",
                "查看本局信息",
                (lu, lv),
                l_anchor,
                self._rect_poly(lu, lv, 0.056, 0.052),
            )
        ]

        cov_zone = self.layout.covenant_zone
        for k in range(self._covenant_count):
            cu, cv = cov_zone.slot_center(k, count=self._covenant_count)
            poly = cov_zone.slot_polygon(k, count=self._covenant_count)
            items.append(
                (
                    "covenant",
                    f"羁绊 [{k + 1}/{self._covenant_count}]",
                    (cu, cv),
                    "center_16_9",
                    poly,
                )
            )

        items.append(
            (
                "enemy_preview",
                "查看本轮出怪",
                (ru, rv),
                r_anchor,
                self._rect_poly(ru, rv, 0.056, 0.052),
            )
        )
        return items

    def _nearest_top_info_idx(self, u: float) -> int:
        items = self._top_info_items()
        best_idx = 0
        best_dist = float("inf")
        for idx, (_, _, (iu, _), _, _) in enumerate(items):
            dist = abs(iu - u)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    def _teammate_slot_geometry(
        self, idx: int
    ) -> tuple[tuple[float, float], str, list[tuple[float, float]]]:
        mate_num = max(1, min(4, int(idx) + 1))
        pt = self.layout.buttons.get(f"teammate_{mate_num}")
        if pt is not None:
            u, v, anchor = pt.u, pt.v, pt.anchor
        else:
            u, v, anchor = 0.041, 0.268 + 0.105 * (mate_num - 1), "left_edge"
        return ((u, v), anchor, self._rect_poly(u, v, 0.054, 0.092))

    def _nearest_teammate_idx(self, v: float) -> int:
        best_idx = 0
        best_dist = float("inf")
        for idx in range(4):
            (_, mv), _, _ = self._teammate_slot_geometry(idx)
            dist = abs(mv - v)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    def _toggle_match_info_panel(self, viewport: ViewportRect) -> None:
        """打开或关闭顶部左侧放大镜（查看本局信息：核心/附加盟约、敌方领袖与词条）。"""
        if self._info_overlay == "match_info":
            if not self._click_button("covenant_close", viewport):
                self._click_norm(0.940, 0.061, viewport, anchor="right_edge")
            self._info_overlay = None
            self._emit_rumble(0.16, 0.20, 30)
            self._set_toast("已关闭本局信息面板")
            return

        if self._info_overlay == "enemy_preview":
            self._trigger_action("enemy_preview", "enemy_preview", "w", viewport)
            self._wait_ui_transition(160)

        self._trigger_action("covenant_panel", "covenant_panel", None, viewport)
        self._info_overlay = "match_info"
        self._emit_rumble(0.18, 0.24, 35)
        self._set_toast("查看本局信息（按 B 关闭）")

    def _toggle_enemy_preview(self, viewport: ViewportRect) -> None:
        """打开或关闭顶部右侧放大镜（查看本轮出怪与路线预览，原生快捷键 W）。"""
        if self._info_overlay == "match_info":
            if not self._click_button("covenant_close", viewport):
                self._click_norm(0.940, 0.061, viewport, anchor="right_edge")
            self._wait_ui_transition(160)

        self._trigger_action("enemy_preview", "enemy_preview", "w", viewport)
        self._emit_rumble(0.18, 0.24, 35)
        if self._info_overlay == "enemy_preview":
            self._info_overlay = None
            self._set_toast("已退出本轮出怪预览")
        else:
            self._info_overlay = "enemy_preview"
            self._set_toast("查看本轮出怪（按 B 关闭）")

    def _close_info_overlay_if_open(self, viewport: ViewportRect) -> bool:
        """若本局信息或本轮出怪浮层处于开启态，则执行对应安全关闭操作并返回 True。"""
        if self._info_overlay == "match_info":
            if not self._click_button("covenant_close", viewport):
                self._click_norm(0.940, 0.061, viewport, anchor="right_edge")
            self._info_overlay = None
            self._set_toast("已关闭本局信息面板")
            return True
        if self._info_overlay == "enemy_preview":
            self._trigger_action("enemy_preview", "enemy_preview", "w", viewport)
            self._info_overlay = None
            self._set_toast("已退出本轮出怪预览")
            return True
        return False

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

    def _strategy_avatar_geometry(
        self,
    ) -> tuple[tuple[float, float], str, list[tuple[float, float]]]:
        """返回右侧策略头像的固定中心坐标、锚点与选框多边形（绝不受 shop_open 缩放影响）。"""
        pt = self.layout.buttons.get("strategy_avatar")
        if pt is not None:
            u, v, anchor = pt.u, pt.v, pt.anchor
        else:
            u, v, anchor = 0.9605, 0.3243, "right_edge"
        return ((u, v), anchor, self._rect_poly(u, v, 0.0656, 0.1167))

    def _strategy_inspect_point(self) -> tuple[float, float, str]:
        """返回策略头像左侧「🔍 查看」按钮的固定坐标与锚点。"""
        pt = self.layout.buttons.get("strategy_inspect")
        if pt is not None:
            return (pt.u, pt.v, pt.anchor)
        return (0.9047, 0.3236, "right_edge")

    def _confirm_dialog_item(
        self, idx: int
    ) -> tuple[str, str, tuple[float, float], str, list[tuple[float, float]]]:
        """返回回合确认弹窗第 idx 项（0=取消, 1=确认, 2=本局不再提示）的按钮键、标签、坐标、锚点与选框。"""
        clamped = max(0, min(2, int(idx)))
        if clamped == 0:
            pt = self.layout.buttons.get("confirm_no")
            u, v, anchor = (
                (pt.u, pt.v, pt.anchor)
                if pt is not None
                else (0.2496, 0.6729, "center_16_9")
            )
            poly = [(0.0, 0.623), (0.50, 0.623), (0.50, 0.723), (0.0, 0.723)]
            return ("confirm_no", "取消", (u, v), anchor, poly)
        if clamped == 1:
            pt = self.layout.buttons.get("confirm_yes")
            u, v, anchor = (
                (pt.u, pt.v, pt.anchor)
                if pt is not None
                else (0.7496, 0.6729, "center_16_9")
            )
            poly = [(0.50, 0.623), (1.0, 0.623), (1.0, 0.723), (0.50, 0.723)]
            return ("confirm_yes", "确认", (u, v), anchor, poly)
        pt = self.layout.buttons.get("confirm_dont_ask")
        u, v, anchor = (
            (pt.u, pt.v, pt.anchor)
            if pt is not None
            else (0.4992, 0.5597, "center_16_9")
        )
        poly = self._rect_poly(u, v, 0.16, 0.05)
        return ("confirm_dont_ask", "本局不再提示", (u, v), anchor, poly)

    def _click_confirm_item(self, idx: int, viewport: ViewportRect) -> None:
        """点击回合确认弹窗指定项并输出触觉与文案反馈。"""
        clamped = max(0, min(2, int(idx)))
        if not self._confirm_has_dont_ask and clamped >= 2:
            return
        _, _, (u, v), anchor, _ = self._confirm_dialog_item(clamped)
        self._click_norm(u, v, viewport, anchor=anchor, button="left")
        if clamped == 2:
            self._confirm_idx = 1
            self._emit_rumble(0.18, 0.24, 35)
            self._set_toast("已切换「本局不再提示」（焦点已回移至确认）")
        elif clamped == 0:
            self._emit_rumble(0.18, 0.22, 35)
            self._set_toast(
                "已取消进入下一回合" if self._confirm_has_dont_ask else "已取消确认弹窗"
            )
        else:
            self._emit_rumble(0.32, 0.45, 55)
            self._set_toast(
                "已确认进入下一回合" if self._confirm_has_dont_ask else "已确认弹窗操作"
            )

    def _apply_confirm_has_dont_ask(self, has_dont_ask: bool) -> None:
        """同步确认弹窗是否包含「本局不再提示」复选框，无该项时安全钳回确认项。"""
        self._confirm_has_dont_ask = bool(has_dont_ask)
        if not self._confirm_has_dont_ask and self._confirm_idx >= 2:
            self._confirm_idx = 1

    # ------------------------------------------------------------------
    # 左下「交流」按钮与 2x3 表情面板几何与交互辅助
    # ------------------------------------------------------------------

    def _emote_button_point(self) -> tuple[float, float, str]:
        """返回左下「交流」按钮的归一化坐标与锚点（绝不发送无效 Z 键）。"""
        pt = self.layout.buttons.get("emote_wheel")
        if pt is not None:
            return (pt.u, pt.v, pt.anchor)
        return (0.0684, 0.9181, "left_edge")

    def _emote_panel_center_point(self) -> tuple[float, float, str]:
        """返回表情面板 ScrollRect 中心点坐标与锚点（供 RS/LB/RB 滚轮翻页落点使用）。"""
        pt = self.layout.buttons.get("emote_panel_center")
        if pt is not None:
            return (pt.u, pt.v, pt.anchor)
        return (0.1290, 0.7400, "left_edge")

    def _emote_slot_center(self, idx: int) -> tuple[float, float]:
        """返回左下 2x3 表情网格第 idx (0..5) 槽位中心归一化坐标。"""
        clamped = max(0, min(5, int(idx)))
        resolver = getattr(self.layout, "resolve_emote_slot_center", None)
        if callable(resolver):
            return resolver(clamped)
        col = clamped % 3
        u = 0.0582 + (0.1996 - 0.0582) * (col / 2.0)
        v = 0.6771 if clamped < 3 else 0.8016
        return (u, v)

    def _emote_slot_polygon(self, idx: int) -> list[tuple[float, float]]:
        """返回左下 2x3 表情网格第 idx (0..5) 槽位四顶点归一化多边形。"""
        clamped = max(0, min(5, int(idx)))
        resolver = getattr(self.layout, "resolve_emote_slot_polygon", None)
        if callable(resolver):
            return resolver(clamped)
        cu, cv = self._emote_slot_center(clamped)
        return self._rect_poly(cu, cv, 0.0640, 0.1120)

    def _move_cursor_to_emote_slot(self, viewport: ViewportRect) -> None:
        """将系统物理光标移至当前聚焦的 2x3 表情槽位中心，确保 ScrollRect 悬停与视觉对齐。"""
        u, v = self._emote_slot_center(self._emote_idx)
        sx, sy = viewport.norm_to_screen(u, v, anchor="left_edge")
        self.hid.move_cursor(sx, sy)
        self._last_cmd_cursor_px = (sx, sy)

    def _move_cursor_to_emote_panel_center(self, viewport: ViewportRect) -> None:
        """将系统物理光标移至表情面板中心（保证游戏 ScrollRect 准确接收滚轮翻页）。"""
        u, v, anchor = self._emote_panel_center_point()
        sx, sy = viewport.norm_to_screen(u, v, anchor=anchor)
        self.hid.move_cursor(sx, sy)
        self._last_cmd_cursor_px = (sx, sy)

    def _remember_zone_before_emote(self) -> None:
        """记录进入表情面板前的合法功能区，避免关闭表情面板后落回已关闭商店残框。"""
        if self._focus_zone in (FocusZone.EMOTE_PANEL, FocusZone.FREE_CURSOR):
            self._zone_before_emote = (
                FocusZone.BENCH
                if self._current_scene == GameScene.BATTLE_PREP
                else FocusZone.BOARD
            )
            return
        if self._focus_zone == FocusZone.SHOP and not self._shop_open:
            self._zone_before_emote = FocusZone.BENCH
            return
        self._zone_before_emote = self._focus_zone

    def _restore_zone_after_emote(self) -> None:
        """关闭表情面板后将焦点安全恢复至原功能区。"""
        target = self._zone_before_emote
        if self._current_scene == GameScene.BATTLE_COMBAT and target in (
            FocusZone.SHOP,
            FocusZone.BENCH,
        ):
            target = FocusZone.BOARD
        elif target == FocusZone.SHOP and not self._shop_open:
            target = FocusZone.BENCH
        elif target in (FocusZone.EMOTE_PANEL, FocusZone.FREE_CURSOR):
            target = (
                FocusZone.BENCH
                if self._current_scene == GameScene.BATTLE_PREP
                else FocusZone.BOARD
            )
        self._focus_zone = target

    def _toggle_emote_panel(
        self,
        viewport: ViewportRect,
        *,
        target_open: bool | None = None,
    ) -> None:
        """打开或关闭左下「交流」表情面板（商店展开态先收起商店，绝不发送无效 Z 键）。"""
        want_open = (not self._emote_panel_open) if target_open is None else bool(target_open)
        self._clear_shop_pending_buy()

        if want_open:
            if self.deploy_fsm.is_active:
                self.deploy_fsm.cancel(viewport)
            if self._unit_selected:
                self._click_safe_blank(viewport)
                self._set_unit_selected(False)
            if self._current_scene == GameScene.BATTLE_PREP and self._shop_open:
                self._set_shop_open(False, viewport, announce=False)
                self._wait_ui_transition(220)
            self._remember_zone_before_emote()
            if self._free_cursor_active:
                self._free_cursor_active = False
            bu, bv, b_anchor = self._emote_button_point()
            self._click_norm(bu, bv, viewport, anchor=b_anchor, button="left")
            self._emote_panel_open = True
            self._emote_override_until = time.monotonic() + _STATE_SYNC_LOCK_SECONDS
            self._focus_zone = FocusZone.EMOTE_PANEL
            self._emote_idx = max(0, min(5, self._emote_idx))
            self._wait_ui_transition(120)
            self._move_cursor_to_emote_slot(viewport)
            self._emit_rumble(0.16, 0.22, 35)
            self._set_toast("已打开交流表情面板（十字键选表情 · RS/LB/RB 翻页 · A 发送 · B 关闭）")
            return

        bu, bv, b_anchor = self._emote_button_point()
        self._click_norm(bu, bv, viewport, anchor=b_anchor, button="left")
        self._emote_panel_open = False
        self._emote_override_until = time.monotonic() + _STATE_SYNC_LOCK_SECONDS
        self._restore_zone_after_emote()
        self._emit_rumble(0.14, 0.18, 30)
        self._set_toast("已关闭交流表情面板")

    def _handle_emote_panel_button(
        self,
        button: GamepadButton,
        viewport: ViewportRect,
        *,
        is_repeat: bool,
    ) -> None:
        """处理左下 2x3 交流表情面板内的方向导航、翻页、发送与关闭操作。"""
        if button == GamepadButton.DPAD_LEFT:
            if self._emote_idx < 3:
                self._emote_idx = (self._emote_idx - 1) % 3
            else:
                self._emote_idx = 3 + ((self._emote_idx - 3 - 1) % 3)
            self._move_cursor_to_emote_slot(viewport)
            return
        if button == GamepadButton.DPAD_RIGHT:
            if self._emote_idx < 3:
                self._emote_idx = (self._emote_idx + 1) % 3
            else:
                self._emote_idx = 3 + ((self._emote_idx - 3 + 1) % 3)
            self._move_cursor_to_emote_slot(viewport)
            return
        if button in (GamepadButton.DPAD_UP, GamepadButton.DPAD_DOWN):
            # 2x3 网格严格上下跨排映射：0<->3, 1<->4, 2<->5
            self._emote_idx = (self._emote_idx + 3) % 6
            self._move_cursor_to_emote_slot(viewport)
            return
        if button == GamepadButton.LB:
            self._move_cursor_to_emote_panel_center(viewport)
            self.hid.scroll(_SCROLL_WHEEL_DELTA * 2, horizontal=False)
            self._set_toast("表情面板向上翻页")
            return
        if button == GamepadButton.RB:
            self._move_cursor_to_emote_panel_center(viewport)
            self.hid.scroll(-_SCROLL_WHEEL_DELTA * 2, horizontal=False)
            self._set_toast("表情面板向下翻页")
            return

        if is_repeat:
            return

        if button == GamepadButton.A:
            u, v = self._emote_slot_center(self._emote_idx)
            sx, sy = viewport.norm_to_screen(u, v, anchor="left_edge")
            self.hid.click(sx, sy, button="left")
            self._last_cmd_cursor_px = (sx, sy)
            self._emit_rumble(0.22, 0.28, 40)
            self._set_toast(f"已发送交流表情 [{self._emote_idx + 1}/6]")
        elif button in (GamepadButton.B, GamepadButton.LS):
            self._toggle_emote_panel(viewport, target_open=False)
        elif (
            button == GamepadButton.START
            and self._current_scene == GameScene.BATTLE_PREP
            and self.keymap.ready_hold_ms == 0
        ):
            self._trigger_battle_ready(viewport)

    def _trigger_shop_upgrade(self, viewport: ViewportRect) -> bool:
        """触发商店升级：原生快捷键模式默认发送 G 键，纯鼠标模式点击随栏位数左移的升级卡。"""
        if self.keymap.use_native_keyboard_shortcuts:
            key = self._get_shortcut("shop_upgrade", "g")
            if key:
                self.hid.tap_key(key)
                return True
        pt = self.layout.resolve_shop_upgrade_point(
            self._view_state, slot_count=self._shop_slot_count
        )
        if pt is None:
            return False
        self._click_norm(pt.u, pt.v, viewport, anchor=pt.anchor, button="left")
        return True

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

    def _sync_emote_panel_open(
        self, scene_result: SceneDetectionResult, now: float
    ) -> None:
        """同步视觉检测到的左下表情面板开关状态，并自动切换/恢复 FocusZone.EMOTE_PANEL。"""
        if now < self._emote_override_until:
            return
        detected_emote = bool(getattr(scene_result, "emote_panel_open", False))
        if detected_emote == self._emote_panel_open:
            return
        self._emote_panel_open = detected_emote
        if detected_emote:
            if not self.deploy_fsm.is_active and not self._unit_selected:
                self._clear_shop_pending_buy()
                self._remember_zone_before_emote()
                self._focus_zone = FocusZone.EMOTE_PANEL
        elif self._focus_zone == FocusZone.EMOTE_PANEL:
            self._restore_zone_after_emote()

    def _sync_shop_closed_fallback(self) -> None:
        """当商店已收起时清理预选购买，并在空闲态将商店残框退回整备区。"""
        if self._shop_open:
            return
        self._clear_shop_pending_buy()
        if (
            self._current_scene == GameScene.BATTLE_PREP
            and self._focus_zone == FocusZone.SHOP
            and not self.deploy_fsm.is_active
        ):
            self._focus_zone = FocusZone.BENCH

    def _sync_choice_metadata(
        self, scene_result: SceneDetectionResult, now: float
    ) -> int | None:
        """同步选卡弹窗卡片总数（二/三/六选一）与底部白条预选索引。"""
        new_choice_count = max(1, int(scene_result.choice_count))
        if new_choice_count != self._choice_count:
            self._choice_count = new_choice_count
            self._choice_idx = max(0, min(self._choice_count - 1, self._choice_idx))
            if (
                self._selected_choice_idx is not None
                and self._selected_choice_idx >= self._choice_count
            ):
                self._selected_choice_idx = None

        detected_sel_choice = getattr(scene_result, "selected_choice_index", None)
        if (
            scene_result.scene == GameScene.MODAL_CHOICE
            and self._current_scene == GameScene.MODAL_CHOICE
            and now >= self._selected_choice_override_until
            and isinstance(detected_sel_choice, int)
            and 0 <= detected_sel_choice < self._choice_count
        ):
            self._selected_choice_idx = detected_sel_choice
        return detected_sel_choice if isinstance(detected_sel_choice, int) else None

    def _sync_scene(self, scene_result: SceneDetectionResult | None) -> None:
        if scene_result is None:
            return

        now = time.monotonic()

        # 手动锁层态下不自动切场景，但若视觉与锁层一致仍实时同步子状态
        if self.scene_locked_manual:
            if (
                self._current_scene == GameScene.BATTLE_PREP
                and scene_result.scene == GameScene.BATTLE_PREP
            ):
                if getattr(scene_result, "has_teammates", False):
                    self._has_teammates = True
                detected_cov = getattr(scene_result, "covenant_count", None)
                if isinstance(detected_cov, int) and detected_cov >= 0:
                    self._apply_covenant_count(detected_cov)
                if now >= self._shop_override_until:
                    self._shop_open = bool(scene_result.shop_open)
                if now >= self._unit_override_until:
                    self._unit_selected = bool(getattr(scene_result, "unit_selected", False))
                    if not self._unit_selected:
                        self._unit_detail_from_shop = False
                detected_slots = getattr(scene_result, "shop_slot_count", None)
                if (
                    scene_result.shop_open
                    and self._shop_open
                    and isinstance(detected_slots, int)
                    and detected_slots >= 1
                ):
                    self._apply_shop_slot_count(detected_slots)
                self._sync_shop_closed_fallback()
                self._sync_emote_panel_open(scene_result, now)
                self._refresh_view_state()
            elif (
                self._current_scene == GameScene.BATTLE_COMBAT
                and scene_result.scene == GameScene.BATTLE_COMBAT
            ):
                if getattr(scene_result, "has_teammates", False):
                    self._has_teammates = True
                detected_cov = getattr(scene_result, "covenant_count", None)
                if isinstance(detected_cov, int) and detected_cov >= 0:
                    self._apply_covenant_count(detected_cov)
                if now >= self._unit_override_until:
                    self._unit_selected = bool(getattr(scene_result, "unit_selected", False))
                    if not self._unit_selected:
                        self._unit_detail_from_shop = False
                self._sync_emote_panel_open(scene_result, now)
                self._refresh_view_state()
            elif (
                self._current_scene == GameScene.MODAL_CHOICE
                and scene_result.scene == GameScene.MODAL_CHOICE
            ):
                self._sync_choice_metadata(scene_result, now)
            elif (
                self._current_scene == GameScene.MODAL_CONFIRM
                and scene_result.scene == GameScene.MODAL_CONFIRM
            ):
                self._apply_confirm_has_dont_ask(
                    getattr(scene_result, "confirm_has_dont_ask", True)
                )
            return

        # 按住 LT 收起决策弹窗查看战场期间，锁住 MODAL_CHOICE 防止松开 LT 前丢失恢复上下文
        if self._current_scene == GameScene.MODAL_CHOICE and self._choice_eye_hidden:
            return

        # 打开本局信息/本轮出怪期间背景按钮隐藏可能使视觉暂报 UNKNOWN，保留局内上下文以便按 B 关闭
        if (
            self._info_overlay is not None
            and self._current_scene in (GameScene.BATTLE_PREP, GameScene.BATTLE_COMBAT)
            and scene_result.scene == GameScene.UNKNOWN_FREE_CURSOR
        ):
            return

        if getattr(scene_result, "has_teammates", False):
            self._has_teammates = True

        detected_cov = getattr(scene_result, "covenant_count", None)
        if (
            scene_result.scene in (GameScene.BATTLE_PREP, GameScene.BATTLE_COMBAT)
            and isinstance(detected_cov, int)
            and detected_cov >= 0
        ):
            self._apply_covenant_count(detected_cov)

        if now >= self._shop_override_until:
            self._shop_open = bool(scene_result.shop_open)
        if now >= self._unit_override_until:
            self._unit_selected = bool(getattr(scene_result, "unit_selected", False))
            if not self._unit_selected:
                self._unit_detail_from_shop = False

        detected_slots = getattr(scene_result, "shop_slot_count", None)
        if (
            scene_result.scene == GameScene.BATTLE_PREP
            and scene_result.shop_open
            and self._shop_open
            and isinstance(detected_slots, int)
            and detected_slots >= 1
        ):
            self._apply_shop_slot_count(detected_slots)

        detected_sel_choice = self._sync_choice_metadata(scene_result, now)
        detected_confirm_dont_ask = bool(
            getattr(scene_result, "confirm_has_dont_ask", True)
        )

        if scene_result.scene == self._current_scene:
            if self._current_scene == GameScene.BATTLE_PREP:
                self._sync_shop_closed_fallback()
            if self._current_scene in (GameScene.BATTLE_PREP, GameScene.BATTLE_COMBAT):
                self._sync_emote_panel_open(scene_result, now)
            elif self._current_scene == GameScene.MODAL_CONFIRM:
                self._apply_confirm_has_dont_ask(detected_confirm_dont_ask)
            self._refresh_view_state()
            return

        # 若玩家正在拖拽或定朝向，不因短暂视觉波动强行中断部署流程；但弹窗（选卡/回合确认）始终立即响应
        if self.deploy_fsm.is_active and scene_result.scene not in (
            GameScene.MODAL_CHOICE,
            GameScene.MODAL_CONFIRM,
        ):
            self._refresh_view_state()
            return

        self._apply_scene_transition(
            scene_result.scene,
            initial_selected_choice=detected_sel_choice,
            confirm_has_dont_ask=(
                detected_confirm_dont_ask
                if scene_result.scene == GameScene.MODAL_CONFIRM
                else None
            ),
        )
        if self._current_scene in (GameScene.BATTLE_PREP, GameScene.BATTLE_COMBAT):
            self._sync_emote_panel_open(scene_result, now)

    def _apply_scene_transition(
        self,
        target_scene: GameScene,
        *,
        initial_selected_choice: int | None = None,
        confirm_has_dont_ask: bool | None = None,
    ) -> None:
        self._clear_shop_pending_buy()
        if self._current_scene == GameScene.MODAL_CHOICE and target_scene != GameScene.MODAL_CHOICE:
            self._choice_eye_hidden = False
            self._selected_choice_idx = None
            self._selected_choice_override_until = 0.0

        if target_scene != GameScene.BATTLE_PREP and self.deploy_fsm.is_active:
            self.deploy_fsm.reset()

        if target_scene not in (GameScene.BATTLE_PREP, GameScene.BATTLE_COMBAT):
            self._unit_selected = False
            self._unit_detail_from_shop = False
            self._unit_override_until = 0.0
            self._emote_panel_open = False
            self._emote_override_until = 0.0
            self._info_overlay = None
        if target_scene != GameScene.BATTLE_PREP:
            self._shop_override_until = 0.0
            self._unit_detail_from_shop = False

        self._current_scene = target_scene
        self._ready_hold_progress = 0.0

        if target_scene == GameScene.BATTLE_PREP:
            self._focus_zone = FocusZone.SHOP if self._shop_open else FocusZone.BENCH
            self._free_cursor_active = False
        elif target_scene == GameScene.BATTLE_COMBAT:
            self._focus_zone = FocusZone.BOARD
            self._free_cursor_active = False
        elif target_scene == GameScene.MODAL_CHOICE:
            self._focus_zone = FocusZone.CHOICE_CARDS
            self._free_cursor_active = False
            if (
                isinstance(initial_selected_choice, int)
                and 0 <= initial_selected_choice < self._choice_count
            ):
                self._selected_choice_idx = initial_selected_choice
                self._choice_idx = initial_selected_choice
            else:
                self._selected_choice_idx = None
                self._choice_idx = 1 if self._choice_count >= 3 else 0
        elif target_scene == GameScene.MODAL_CONFIRM:
            self._focus_zone = FocusZone.CONFIRM_DIALOG
            self._free_cursor_active = False
            self._confirm_idx = 1
            self._apply_confirm_has_dont_ask(
                True if confirm_has_dont_ask is None else confirm_has_dont_ask
            )
        else:
            self._focus_zone = FocusZone.FREE_CURSOR
            self._free_cursor_active = True
            self._last_cmd_cursor_px = None

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

        self._clear_shop_pending_buy()
        toggled = self._trigger_action("shop_toggle", "shop_toggle", "a", viewport)
        self._shop_open = bool(target_open)
        now = time.monotonic()
        self._shop_override_until = now + _STATE_SYNC_LOCK_SECONDS
        if self._shop_open:
            self._unit_selected = False
            self._unit_detail_from_shop = False
            self._unit_override_until = now + _STATE_SYNC_LOCK_SECONDS
            self._emote_panel_open = False
            self._emote_override_until = now + _STATE_SYNC_LOCK_SECONDS
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
        if not self._unit_selected:
            self._unit_detail_from_shop = False
        if lock:
            self._unit_override_until = time.monotonic() + _STATE_SYNC_LOCK_SECONDS
        self._refresh_view_state()

    def _enter_battle_prep_zone(
        self, target_zone: FocusZone, viewport: ViewportRect
    ) -> None:
        """切换休整期焦点分区，并根据 auto_toggle_shop 自动联动商店展开/收起。"""
        if target_zone != FocusZone.SHOP:
            self._clear_shop_pending_buy()
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
        if self._focus_zone == FocusZone.SHOP:
            return self._shop_mapper.slot_center(self._shop_idx)
        if self._focus_zone == FocusZone.BENCH:
            return self._bench_mapper.slot_center(self._bench_idx)
        if self._focus_zone == FocusZone.BOARD:
            return self._board_mapper.cell_center(self._board_row, self._board_col)
        if self._focus_zone == FocusZone.EQUIPMENT:
            return self._equip_mapper.slot_center(self._equip_idx)
        if self._focus_zone == FocusZone.STRATEGY:
            uv, _, _ = self._strategy_avatar_geometry()
            return uv
        if self._focus_zone == FocusZone.TOP_INFO:
            items = self._top_info_items()
            idx = max(0, min(len(items) - 1, self._top_info_idx))
            return items[idx][2]
        if self._focus_zone == FocusZone.TEAMMATES:
            uv, _, _ = self._teammate_slot_geometry(self._teammate_focus_idx)
            return uv
        if self._focus_zone == FocusZone.EMOTE_PANEL:
            return self._emote_slot_center(self._emote_idx)
        if self._focus_zone == FocusZone.CHOICE_CARDS:
            if self._selected_choice_idx == self._choice_idx:
                return self.layout.resolve_choice_confirm_point(
                    self._choice_idx, self._choice_count
                )
            return self.layout.resolve_choice_card_center(
                self._choice_idx, self._choice_count
            )
        if self._focus_zone == FocusZone.CONFIRM_DIALOG:
            idx = (
                1
                if not self._confirm_has_dont_ask and self._confirm_idx >= 2
                else self._confirm_idx
            )
            _, _, uv, _, _ = self._confirm_dialog_item(idx)
            return uv
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
        if self._focus_zone == FocusZone.SHOP:
            return self._shop_mapper.slot_polygon(self._shop_idx)
        if self._focus_zone == FocusZone.BENCH:
            return self._bench_mapper.slot_polygon(self._bench_idx)
        if self._focus_zone == FocusZone.BOARD:
            return self._board_mapper.cell_polygon(self._board_row, self._board_col)
        if self._focus_zone == FocusZone.EQUIPMENT:
            return self._equip_mapper.slot_polygon(self._equip_idx)
        if self._focus_zone == FocusZone.STRATEGY:
            _, _, poly = self._strategy_avatar_geometry()
            return poly
        if self._focus_zone == FocusZone.TOP_INFO:
            items = self._top_info_items()
            idx = max(0, min(len(items) - 1, self._top_info_idx))
            return items[idx][4]
        if self._focus_zone == FocusZone.TEAMMATES:
            _, _, poly = self._teammate_slot_geometry(self._teammate_focus_idx)
            return poly
        if self._focus_zone == FocusZone.EMOTE_PANEL:
            return self._emote_slot_polygon(self._emote_idx)
        if self._focus_zone == FocusZone.CHOICE_CARDS:
            if self._selected_choice_idx == self._choice_idx:
                return self.layout.resolve_choice_confirm_polygon(
                    self._choice_idx, self._choice_count
                )
            return self.layout.resolve_choice_card_polygon(
                self._choice_idx, self._choice_count
            )
        if self._focus_zone == FocusZone.CONFIRM_DIALOG:
            idx = (
                1
                if not self._confirm_has_dont_ask and self._confirm_idx >= 2
                else self._confirm_idx
            )
            _, _, _, _, poly = self._confirm_dialog_item(idx)
            return poly
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
                self._toggle_free_cursor(viewport)
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

        toggle_pt = self.layout.resolve_choice_toggle_point(self._choice_count)
        if toggle_pt is None:
            return

        if event.active and not self._choice_eye_hidden:
            self._click_norm(toggle_pt.u, toggle_pt.v, viewport, anchor=toggle_pt.anchor)
            self._choice_eye_hidden = True
            self._set_toast("已收起弹窗查看战场（松开 LT 展开）")
        elif not event.active and self._choice_eye_hidden:
            self._click_norm(toggle_pt.u, toggle_pt.v, viewport, anchor=toggle_pt.anchor)
            self._choice_eye_hidden = False
            self._set_toast("已恢复决策弹窗")

    def _toggle_free_cursor(self, viewport: ViewportRect | None = None) -> None:
        """LS 按下：一键切换槽位吸附导航与自由虚拟光标模式。"""
        self._clear_shop_pending_buy()
        if not self._free_cursor_active:
            self._cursor_norm = self._current_snap_center()
            self._free_cursor_active = True
            if viewport is not None:
                sx, sy = viewport.norm_to_screen(*self._cursor_norm)
                self.hid.move_cursor(sx, sy)
                self._last_cmd_cursor_px = (sx, sy)
            self._set_toast("已切至自由光标（再按 LS 或十字键恢复吸附）")
        else:
            self._free_cursor_active = False
            self._last_cmd_cursor_px = None
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
            self._last_cmd_cursor_px = (sx, sy)
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
            if self._focus_zone == FocusZone.EMOTE_PANEL and not self._free_cursor_active:
                self._move_cursor_to_emote_panel_center(viewport)
            self.hid.scroll(ticks * _SCROLL_WHEEL_DELTA, horizontal=False)

        if abs(self._scroll_h_accum) >= _SCROLL_STEP_PX:
            ticks = int(self._scroll_h_accum / _SCROLL_STEP_PX)
            self._scroll_h_accum -= ticks * _SCROLL_STEP_PX
            if self._focus_zone == FocusZone.EMOTE_PANEL and not self._free_cursor_active:
                self._move_cursor_to_emote_panel_center(viewport)
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
        elif self._current_scene == GameScene.MODAL_CONFIRM:
            self._click_confirm_item(1, viewport)

    def _trigger_battle_ready(self, viewport: ViewportRect) -> None:
        if self.deploy_fsm.is_active:
            self.deploy_fsm.cancel(viewport)
        self._clear_shop_pending_buy()
        self._info_overlay = None
        self._unit_selected = False
        self._unit_detail_from_shop = False
        self._unit_override_until = 0.0
        self._emote_panel_open = False
        self._emote_override_until = 0.0
        self._trigger_action("battle_ready", "battle_ready", "c", viewport)
        self._emit_rumble(0.50, 0.50, 80)
        self._set_toast("准备就绪 · 开始战斗")

    def _handle_chord(self, event: ChordTriggered, viewport: ViewportRect) -> None:
        # RT + A：商店极速连买 / 三选一双击拿牌 / 回合确认直接确认 / 策略一键点击 / 自由光标双击
        if event.modifier == GamepadButton.RT and event.button == GamepadButton.A:
            self._handle_rt_a(viewport)
            return

        # RT + Start：局内快捷准备就绪 / 回合确认弹窗直接确认
        if {event.modifier, event.button} == {GamepadButton.RT, GamepadButton.START}:
            if self._current_scene == GameScene.BATTLE_PREP:
                self._ready_hold_progress = 0.0
                self._trigger_battle_ready(viewport)
            elif self._current_scene == GameScene.MODAL_CONFIRM:
                self._click_confirm_item(1, viewport)
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
            self._last_cmd_cursor_px = (sx, sy)
            self._emit_rumble(0.22, 0.30, 40)
            return

        if self._current_scene == GameScene.MODAL_CONFIRM:
            self._click_confirm_item(1, viewport)
            return

        if (
            self._current_scene in (GameScene.BATTLE_PREP, GameScene.BATTLE_COMBAT)
            and self._focus_zone == FocusZone.STRATEGY
            and self.deploy_fsm.state == DeployState.IDLE
        ):
            self.deploy_fsm.click_strategy_in_place(viewport)
            self._set_toast("已直接使用右侧策略")
            return

        if (
            self._current_scene == GameScene.BATTLE_PREP
            and self._focus_zone == FocusZone.SHOP
            and self.deploy_fsm.state == DeployState.IDLE
            and not self._unit_selected
        ):
            # 在商店按 RT+A 单击查看商店干员详情（屏蔽出售与撤回）
            self._clear_shop_pending_buy()
            if getattr(self.keymap, "auto_toggle_shop", True) and not self._shop_open:
                self._set_shop_open(True, viewport, announce=False)
                self._wait_ui_transition(220)
            u, v = self._shop_mapper.slot_center(self._shop_idx)
            self._click_norm(u, v, viewport)
            self._unit_tab_idx = 0
            self._unit_detail_from_shop = True
            self._set_unit_selected(True)
            self._emit_rumble(0.18, 0.25, 35)
            self._set_toast("查看商店干员详情（左右切标签 · B 退出）")
            return

        if (
            self._current_scene == GameScene.BATTLE_PREP
            and self._focus_zone in (FocusZone.BENCH, FocusZone.BOARD)
            and self.deploy_fsm.state == DeployState.IDLE
            and not self._unit_selected
        ):
            # 在整备区/棋盘按 RT+A 主动单击查看角色详情，并自动切换至选中角色特写详情态
            u, v = self._current_snap_center()
            self._click_norm(u, v, viewport)
            self._unit_tab_idx = 0
            self._unit_detail_from_shop = False
            self._set_unit_selected(True)
            self._emit_rumble(0.18, 0.25, 35)
            self._set_toast("查看角色详情（左右切标签 · X 出售 · Y 撤回 · B 退出）")
            return

        if self._current_scene == GameScene.MODAL_CHOICE and not self._choice_eye_hidden:
            if self._selected_choice_idx != self._choice_idx:
                cu, cv = self.layout.resolve_choice_card_center(
                    self._choice_idx, self._choice_count
                )
                self._click_norm(cu, cv, viewport)
                self._wait_ui_transition(160)
            confirm_u, confirm_v = self.layout.resolve_choice_confirm_point(
                self._choice_idx, self._choice_count
            )
            self._click_norm(confirm_u, confirm_v, viewport)
            self._selected_choice_idx = None
            self._selected_choice_override_until = (
                time.monotonic() + _STATE_SYNC_LOCK_SECONDS
            )
            self._emit_rumble(0.30, 0.40, 50)
            self._set_toast(f"已确认选择第 {self._choice_idx + 1} 张决策卡")

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
            self._toggle_match_info_panel(viewport)
            return

        if button == GamepadButton.DPAD_UP:
            self._toggle_enemy_preview(viewport)
            return

        if button == GamepadButton.DPAD_DOWN:
            self._toggle_emote_panel(viewport)
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
        self._has_teammates = True
        self._selected_teammate = idx
        self._teammate_focus_idx = idx - 1
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
            self._last_cmd_cursor_px = None

        # 纯自由光标模式处理（含局外未知界面）
        if self._free_cursor_active or self._current_scene == GameScene.UNKNOWN_FREE_CURSOR:
            self._handle_free_cursor_button(button, viewport, is_repeat=is_repeat)
            return

        if self._current_scene == GameScene.MODAL_CHOICE:
            self._handle_modal_choice_button(button, viewport, is_repeat=is_repeat)
            return

        if self._current_scene == GameScene.MODAL_CONFIRM:
            self._handle_modal_confirm_button(button, viewport, is_repeat=is_repeat)
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
            sx, sy = viewport.norm_to_screen(*self._cursor_norm)
            self.hid.click(sx, sy, button="left")
            self._last_cmd_cursor_px = (sx, sy)
        elif button == GamepadButton.B:
            if not self._close_info_overlay_if_open(viewport):
                self._trigger_action("back", "back", "esc", viewport)
        elif button == GamepadButton.START:
            if self._current_scene == GameScene.BATTLE_PREP and self.keymap.ready_hold_ms == 0:
                self._trigger_battle_ready(viewport)
        elif self._current_scene == GameScene.BATTLE_PREP:
            if button == GamepadButton.X:
                self._trigger_action("shop_refresh", "shop_refresh", "d", viewport)
                self._set_toast("刷新商店")
            elif button == GamepadButton.Y:
                self._trigger_shop_upgrade(viewport)
                self._set_toast("升级商店等级")
            elif button == GamepadButton.RS:
                self._trigger_action("shop_freeze", "shop_freeze", "s", viewport)
                self._set_toast("切换商店冻结")
            elif button == GamepadButton.LS:
                self._toggle_emote_panel(viewport)
        elif (
            self._current_scene == GameScene.BATTLE_COMBAT
            and button == GamepadButton.LS
        ):
            self._toggle_emote_panel(viewport)

    # ------------------------------------------------------------------
    # 回合结束资金/策略确认弹窗处理（GameScene.MODAL_CONFIRM）
    # ------------------------------------------------------------------

    def _handle_modal_confirm_button(
        self,
        button: GamepadButton,
        viewport: ViewportRect,
        *,
        is_repeat: bool,
    ) -> None:
        item_count = 3 if self._confirm_has_dont_ask else 2
        if button == GamepadButton.DPAD_LEFT:
            self._confirm_idx = 0
            return
        if button == GamepadButton.DPAD_RIGHT:
            self._confirm_idx = 1
            return
        if button == GamepadButton.DPAD_UP:
            if self._confirm_has_dont_ask:
                self._confirm_idx = 2
            return
        if button == GamepadButton.DPAD_DOWN:
            self._confirm_idx = 1
            return
        if button == GamepadButton.LB:
            self._confirm_idx = (self._confirm_idx - 1) % item_count
            return
        if button == GamepadButton.RB:
            self._confirm_idx = (self._confirm_idx + 1) % item_count
            return

        if is_repeat:
            return

        if button == GamepadButton.A:
            target_idx = (
                1
                if not self._confirm_has_dont_ask and self._confirm_idx >= 2
                else self._confirm_idx
            )
            self._click_confirm_item(target_idx, viewport)
        elif button == GamepadButton.B:
            self._click_confirm_item(0, viewport)
        elif button == GamepadButton.START:
            self._click_confirm_item(1, viewport)
        elif button == GamepadButton.X and self._confirm_has_dont_ask:
            self._click_confirm_item(2, viewport)

    # ------------------------------------------------------------------
    # 三选一 / 多人六选一决策弹窗处理
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

        total = max(1, self._choice_count)
        if total >= 5:
            # 多人 2x3 六选一网格导航：左右同行循环，上下跨排切换，LB/RB 全局顺序步进
            top_cols = max(1, (total + 1) // 2)
            if button == GamepadButton.DPAD_LEFT:
                if self._choice_idx < top_cols:
                    self._choice_idx = (self._choice_idx - 1) % top_cols
                else:
                    bot_cols = max(1, total - top_cols)
                    self._choice_idx = top_cols + ((self._choice_idx - top_cols - 1) % bot_cols)
                return
            if button == GamepadButton.DPAD_RIGHT:
                if self._choice_idx < top_cols:
                    self._choice_idx = (self._choice_idx + 1) % top_cols
                else:
                    bot_cols = max(1, total - top_cols)
                    self._choice_idx = top_cols + ((self._choice_idx - top_cols + 1) % bot_cols)
                return
            if button in (GamepadButton.DPAD_UP, GamepadButton.DPAD_DOWN):
                if self._choice_idx < top_cols:
                    self._choice_idx = min(total - 1, self._choice_idx + top_cols)
                else:
                    self._choice_idx = max(0, self._choice_idx - top_cols)
                return
            if button == GamepadButton.LB:
                self._choice_idx = (self._choice_idx - 1) % total
                return
            if button == GamepadButton.RB:
                self._choice_idx = (self._choice_idx + 1) % total
                return
        else:
            if button in (GamepadButton.DPAD_LEFT, GamepadButton.LB):
                self._choice_idx = (self._choice_idx - 1) % total
                return
            if button in (GamepadButton.DPAD_RIGHT, GamepadButton.RB):
                self._choice_idx = (self._choice_idx + 1) % total
                return

        if is_repeat:
            return

        if button == GamepadButton.A:
            if self._selected_choice_idx != self._choice_idx:
                cu, cv = self.layout.resolve_choice_card_center(
                    self._choice_idx, self._choice_count
                )
                self._click_norm(cu, cv, viewport)
                self._selected_choice_idx = self._choice_idx
                self._selected_choice_override_until = (
                    time.monotonic() + _STATE_SYNC_LOCK_SECONDS
                )
                self._emit_rumble(0.18, 0.24, 35)
                self._set_toast(
                    f"已选中第 {self._choice_idx + 1} 张卡（再按 A 点击底部白色确认条）"
                )
            else:
                confirm_u, confirm_v = self.layout.resolve_choice_confirm_point(
                    self._choice_idx, self._choice_count
                )
                self._click_norm(confirm_u, confirm_v, viewport)
                self._selected_choice_idx = None
                self._selected_choice_override_until = (
                    time.monotonic() + _STATE_SYNC_LOCK_SECONDS
                )
                self._emit_rumble(0.30, 0.42, 55)
                self._set_toast(f"已确认选择第 {self._choice_idx + 1} 张决策卡")
            return

        if button == GamepadButton.B and self._selected_choice_idx is not None:
            self._selected_choice_idx = None
            self._selected_choice_override_until = (
                time.monotonic() + _STATE_SYNC_LOCK_SECONDS
            )
            self._set_toast("已取消待确认聚焦，可重新选卡")

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
            from_shop = self._unit_detail_from_shop
            self._click_safe_blank(viewport)
            self._set_unit_selected(False)
            self._set_toast(
                "已退出商店干员详情"
                if from_shop
                else "已退出角色详情（恢复标准视角）"
            )
            return

        if (
            button == GamepadButton.X
            and self._current_scene == GameScene.BATTLE_PREP
            and not self._unit_detail_from_shop
        ):
            sell_key = self._get_shortcut("sell_board", "x") or "x"
            self.hid.tap_key(sell_key)
            self._set_unit_selected(False)
            self._emit_rumble(0.30, 0.40, 55)
            self._set_toast(f"已出售选中角色 ({sell_key.upper()})")
            return

        if button == GamepadButton.Y and not self._unit_detail_from_shop:
            retreat_key = self._get_shortcut("retreat_unit", "q") or "q"
            self.hid.tap_key(retreat_key)
            self._set_unit_selected(False)
            self._emit_rumble(0.30, 0.35, 50)
            self._set_toast(f"已撤回角色 ({retreat_key.upper()})")

    def _activate_top_info_item(self, viewport: ViewportRect) -> None:
        items = self._top_info_items()
        idx = max(0, min(len(items) - 1, self._top_info_idx))
        kind, label, (cu, cv), anchor, _ = items[idx]
        if kind == "match_info":
            self._toggle_match_info_panel(viewport)
            return
        if kind == "enemy_preview":
            self._toggle_enemy_preview(viewport)
            return
        self._close_info_overlay_if_open(viewport)
        self._click_norm(cu, cv, viewport, anchor=anchor, button="left")
        self._emit_rumble(0.16, 0.22, 35)
        self._set_toast(f"查看{label}详情（按 B 关闭浮窗）")

    def _navigate_top_info_or_teammates(
        self, direction: Direction, viewport: ViewportRect | None = None
    ) -> bool:
        """处理顶部信息/羁绊栏、左侧队友栏与右侧策略头像的区内外方向导航；若消费该事件则返回 True。"""
        if self._focus_zone == FocusZone.TOP_INFO:
            items = self._top_info_items()
            if direction == "left":
                self._top_info_idx = (self._top_info_idx - 1) % len(items)
            elif direction == "right":
                self._top_info_idx = (self._top_info_idx + 1) % len(items)
            elif direction == "down":
                cu = items[max(0, min(len(items) - 1, self._top_info_idx))][2][0]
                _, col = self._board_mapper.nearest_cell(
                    cu, self._board_mapper.quad.top_left[1]
                )
                self._board_row = 0
                self._board_col = col
                if self._current_scene == GameScene.BATTLE_PREP and viewport is not None:
                    self._enter_battle_prep_zone(FocusZone.BOARD, viewport)
                else:
                    self._focus_zone = FocusZone.BOARD
            return True

        if self._focus_zone == FocusZone.TEAMMATES:
            if direction == "up":
                self._teammate_focus_idx = (self._teammate_focus_idx - 1) % 4
                self._selected_teammate = self._teammate_focus_idx + 1
            elif direction == "down":
                if (
                    self._teammate_focus_idx >= 3
                    and viewport is not None
                    and not self.deploy_fsm.is_active
                ):
                    self._toggle_emote_panel(viewport, target_open=True)
                    return True
                self._teammate_focus_idx = (self._teammate_focus_idx + 1) % 4
                self._selected_teammate = self._teammate_focus_idx + 1
            elif direction == "right":
                (_, mv), _, _ = self._teammate_slot_geometry(self._teammate_focus_idx)
                row, _ = self._board_mapper.nearest_cell(
                    self._board_mapper.quad.top_left[0], mv
                )
                self._board_row = row
                self._board_col = 0
                if self._current_scene == GameScene.BATTLE_PREP and viewport is not None:
                    self._enter_battle_prep_zone(FocusZone.BOARD, viewport)
                else:
                    self._focus_zone = FocusZone.BOARD
            return True

        if self._focus_zone == FocusZone.STRATEGY:
            (su, sv), _, _ = self._strategy_avatar_geometry()
            if direction == "left":
                row, _ = self._board_mapper.nearest_cell(
                    self._board_mapper.quad.top_right[0], sv
                )
                self._board_row = row
                self._board_col = self._board_mapper.cols - 1
                if self._current_scene == GameScene.BATTLE_PREP and viewport is not None:
                    self._enter_battle_prep_zone(FocusZone.BOARD, viewport)
                else:
                    self._focus_zone = FocusZone.BOARD
            elif direction == "up":
                items = self._top_info_items()
                self._top_info_idx = len(items) - 1
                self._focus_zone = FocusZone.TOP_INFO
            elif direction == "down":
                self._equip_idx = self._equip_mapper.nearest_slot(su, sv)
                if self._current_scene == GameScene.BATTLE_PREP and viewport is not None:
                    self._enter_battle_prep_zone(FocusZone.EQUIPMENT, viewport)
                else:
                    self._focus_zone = FocusZone.EQUIPMENT
            return True

        return False

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

        if self._focus_zone == FocusZone.EMOTE_PANEL:
            self._handle_emote_panel_button(button, viewport, is_repeat=is_repeat)
            return

        if button in _DPAD_TO_DIR:
            direction = _DPAD_TO_DIR[button]
            if self._navigate_top_info_or_teammates(direction, viewport):
                return
            if self._focus_zone == FocusZone.EQUIPMENT:
                new_idx, exit_dir = self._equip_mapper.move(
                    self._equip_idx, direction, wrap=True
                )
                self._equip_idx = new_idx
                if exit_dir == "up":
                    cu, cv = self._equip_mapper.slot_center(self._equip_idx)
                    _, col = self._board_mapper.nearest_cell(cu, cv)
                    self._board_row = self._board_mapper.rows - 1
                    self._board_col = col
                    self._focus_zone = FocusZone.BOARD
                elif exit_dir == "right":
                    self._focus_zone = FocusZone.STRATEGY
                return
            self._focus_zone = FocusZone.BOARD
            self._board_row, self._board_col, exit_dir = self._board_mapper.move(
                self._board_row, self._board_col, direction, wrap_cols=False
            )
            if exit_dir == "up":
                cu, _ = self._board_mapper.cell_center(self._board_row, self._board_col)
                self._top_info_idx = self._nearest_top_info_idx(cu)
                self._focus_zone = FocusZone.TOP_INFO
            elif exit_dir == "right":
                self._focus_zone = FocusZone.STRATEGY
            elif exit_dir == "left" and self._has_teammates:
                _, cv = self._board_mapper.cell_center(self._board_row, self._board_col)
                self._teammate_focus_idx = self._nearest_teammate_idx(cv)
                self._selected_teammate = self._teammate_focus_idx + 1
                self._focus_zone = FocusZone.TEAMMATES
            return

        if is_repeat:
            return

        if button == GamepadButton.A:
            if self._focus_zone == FocusZone.TOP_INFO:
                self._activate_top_info_item(viewport)
            elif self._focus_zone == FocusZone.TEAMMATES:
                self._teammate_cycle_pos = self._teammate_focus_idx + 1
                self._switch_teammate_view(self._teammate_focus_idx + 1, viewport)
            elif self._focus_zone == FocusZone.STRATEGY:
                self.deploy_fsm.click_strategy_in_place(viewport)
                self._set_toast("已直接使用右侧策略")
            else:
                u, v = self._board_mapper.cell_center(self._board_row, self._board_col)
                self._click_norm(u, v, viewport)
                self._unit_tab_idx = 0
                self._unit_detail_from_shop = False
                self._set_unit_selected(True)
        elif button == GamepadButton.X and self._focus_zone == FocusZone.STRATEGY:
            iu, iv, i_anchor = self._strategy_inspect_point()
            self._click_norm(iu, iv, viewport, anchor=i_anchor, button="left")
            self._emit_rumble(0.16, 0.22, 35)
            self._set_toast("查看策略说明（按 B 关闭）")
        elif button == GamepadButton.B:
            if self._close_info_overlay_if_open(viewport):
                return
            if self._focus_zone == FocusZone.TEAMMATES:
                self._focus_zone = FocusZone.BOARD
                return
            self._click_safe_blank(viewport)
            if self._unit_selected:
                self._set_unit_selected(False)
        elif button == GamepadButton.LS:
            self._toggle_emote_panel(viewport)

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

        # 1c. 左下交流表情面板态优先拦截（2x3 网格导航 + RS/LB/RB 翻页 + A 发送 + B 关闭）
        if fsm_state == DeployState.IDLE and self._focus_zone == FocusZone.EMOTE_PANEL:
            self._handle_emote_panel_button(button, viewport, is_repeat=is_repeat)
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

        # 4. 抓取持有态功能按键（干员 / 装备 / 策略头像）
        if fsm_state in (
            DeployState.HOLDING_UNIT,
            DeployState.HOLDING_EQUIP,
            DeployState.HOLDING_STRATEGY,
        ):
            self._handle_holding_button(button, viewport)
            return

        # 5. 休整期常规空闲浏览态按键
        if button == GamepadButton.A:
            self._handle_idle_prep_a(viewport)
        elif button == GamepadButton.B:
            if self._close_info_overlay_if_open(viewport):
                return
            if self._focus_zone == FocusZone.SHOP and self._shop_pending_buy_idx is not None:
                self._clear_shop_pending_buy()
                self._set_toast("已取消商店购买预选")
                return
            if self._focus_zone == FocusZone.SHOP and self._shop_open:
                self._set_shop_open(False, viewport, announce=True)
                self._focus_zone = FocusZone.BENCH
            elif self._focus_zone == FocusZone.TEAMMATES:
                self._enter_battle_prep_zone(FocusZone.BOARD, viewport)
            else:
                self._click_safe_blank(viewport)
        elif button == GamepadButton.X:
            if self._focus_zone == FocusZone.STRATEGY:
                iu, iv, i_anchor = self._strategy_inspect_point()
                self._click_norm(iu, iv, viewport, anchor=i_anchor, button="left")
                self._emit_rumble(0.16, 0.22, 35)
                self._set_toast("查看策略说明（按 B 关闭）")
                return
            self._clear_shop_pending_buy()
            if not self._shop_open and getattr(self.keymap, "auto_toggle_shop", True):
                self._set_shop_open(True, viewport, announce=False)
                self._wait_ui_transition(220)
            self._trigger_action("shop_refresh", "shop_refresh", "d", viewport)
            self._emit_rumble(0.18, 0.22, 35)
            self._set_toast("刷新商店 (D)")
        elif button == GamepadButton.Y:
            self._clear_shop_pending_buy()
            if not self._shop_open and getattr(self.keymap, "auto_toggle_shop", True):
                self._set_shop_open(True, viewport, announce=False)
                self._wait_ui_transition(220)
            self._trigger_shop_upgrade(viewport)
            self._emit_rumble(0.25, 0.30, 45)
            self._set_toast(
                "升级商店等级 (G)"
                if self.keymap.use_native_keyboard_shortcuts
                else "升级商店等级"
            )
        elif button == GamepadButton.RS:
            self._clear_shop_pending_buy()
            if not self._shop_open and getattr(self.keymap, "auto_toggle_shop", True):
                self._set_shop_open(True, viewport, announce=False)
                self._wait_ui_transition(220)
            self._trigger_action("shop_freeze", "shop_freeze", "s", viewport)
            self._emit_rumble(0.18, 0.22, 35)
            self._set_toast("切换商店冻结 (S)")
        elif button == GamepadButton.LS:
            self._toggle_emote_panel(viewport)
        elif button == GamepadButton.START and self.keymap.ready_hold_ms == 0:
            self._trigger_battle_ready(viewport)

    def _switch_battle_prep_zone(
        self, delta: int, viewport: ViewportRect
    ) -> None:
        fsm_state = self.deploy_fsm.state
        if fsm_state == DeployState.HOLDING_UNIT:
            # 抓取干员态下在整备区与棋盘之间快速跳区
            target = (
                FocusZone.BOARD
                if self._focus_zone != FocusZone.BOARD
                else FocusZone.BENCH
            )
            self._enter_battle_prep_zone(target, viewport)
            return

        if fsm_state == DeployState.HOLDING_EQUIP:
            # 抓取装备态下在 装备栏 <-> 棋盘 <-> 整备区 之间循环切换（兼顾穿戴与挪动装备槽位）
            if self._focus_zone not in _HOLDING_EQUIP_ZONE_ORDER:
                self._enter_battle_prep_zone(FocusZone.BOARD, viewport)
                return
            eq_idx = _HOLDING_EQUIP_ZONE_ORDER.index(self._focus_zone)
            next_eq_zone = _HOLDING_EQUIP_ZONE_ORDER[
                (eq_idx + delta) % len(_HOLDING_EQUIP_ZONE_ORDER)
            ]
            self._enter_battle_prep_zone(next_eq_zone, viewport)
            return

        if fsm_state == DeployState.HOLDING_STRATEGY:
            # 抓取策略态下在 策略 -> 棋盘 -> 整备区 -> 商店 之间循环切换
            if self._focus_zone not in _HOLDING_STRATEGY_ZONE_ORDER:
                self._enter_battle_prep_zone(FocusZone.BOARD, viewport)
                return
            st_idx = _HOLDING_STRATEGY_ZONE_ORDER.index(self._focus_zone)
            next_st_zone = _HOLDING_STRATEGY_ZONE_ORDER[
                (st_idx + delta) % len(_HOLDING_STRATEGY_ZONE_ORDER)
            ]
            self._enter_battle_prep_zone(next_st_zone, viewport)
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
        if self._navigate_top_info_or_teammates(direction, viewport):
            return

        is_holding = self.deploy_fsm.state in (
            DeployState.HOLDING_UNIT,
            DeployState.HOLDING_EQUIP,
        )
        is_holding_unit = self.deploy_fsm.state == DeployState.HOLDING_UNIT
        is_holding_strategy = self.deploy_fsm.state == DeployState.HOLDING_STRATEGY
        shop_above_board = (
            self._shop_mapper.zone.start_v < self._board_mapper.quad.top_left[1]
        )
        equip_vertical = self._equip_mapper.is_vertical
        equip_on_right = self._equip_mapper.zone.start_u >= 0.5

        if self._focus_zone == FocusZone.SHOP:
            old_idx = self._shop_idx
            new_idx, exit_dir = self._shop_mapper.move(
                self._shop_idx, direction, wrap=True
            )
            self._shop_idx = new_idx
            if new_idx != old_idx or exit_dir is not None:
                self._clear_shop_pending_buy()
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
                if not equip_vertical and not is_holding_unit and not is_holding_strategy:
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
                if not equip_vertical and not is_holding_unit and not is_holding_strategy:
                    self._enter_battle_prep_zone(FocusZone.EQUIPMENT, viewport)
                    self._equip_idx = self._equip_mapper.nearest_slot(cu, cv)
                else:
                    self._enter_battle_prep_zone(FocusZone.BENCH, viewport)
                    self._bench_idx = self._bench_mapper.nearest_slot(cu, cv)
            elif exit_dir == "up" and shop_above_board and not is_holding:
                cu, cv = self._board_mapper.cell_center(self._board_row, self._board_col)
                self._enter_battle_prep_zone(FocusZone.SHOP, viewport)
                self._shop_idx = self._shop_mapper.nearest_slot(cu, cv)
            elif exit_dir == "up" and not shop_above_board and not is_holding and not is_holding_strategy:
                cu, _ = self._board_mapper.cell_center(self._board_row, self._board_col)
                self._top_info_idx = self._nearest_top_info_idx(cu)
                self._enter_battle_prep_zone(FocusZone.TOP_INFO, viewport)
            elif exit_dir == "right" and equip_vertical and equip_on_right and not is_holding_unit and not is_holding_strategy:
                cu, cv = self._board_mapper.cell_center(self._board_row, self._board_col)
                self._enter_battle_prep_zone(FocusZone.EQUIPMENT, viewport)
                self._equip_idx = self._equip_mapper.nearest_slot(cu, cv)
            elif exit_dir == "right" and not is_holding:
                self._focus_zone = FocusZone.STRATEGY
            elif exit_dir == "left" and equip_vertical and not equip_on_right and not is_holding_unit and not is_holding_strategy:
                cu, cv = self._board_mapper.cell_center(self._board_row, self._board_col)
                self._enter_battle_prep_zone(FocusZone.EQUIPMENT, viewport)
                self._equip_idx = self._equip_mapper.nearest_slot(cu, cv)
            elif exit_dir == "left" and self._has_teammates and not is_holding_strategy:
                _, cv = self._board_mapper.cell_center(self._board_row, self._board_col)
                self._teammate_focus_idx = self._nearest_teammate_idx(cv)
                self._selected_teammate = self._teammate_focus_idx + 1
                self._focus_zone = FocusZone.TEAMMATES
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
            if self._is_shop_pending_buy_for(self._shop_idx):
                self._clear_shop_pending_buy()
                u, v = self._shop_mapper.slot_center(self._shop_idx)
                sx, sy = viewport.norm_to_screen(u, v)
                self.hid.double_click(sx, sy)
                self._emit_rumble(0.25, 0.35, 45)
                self._set_toast(
                    f"已购买商店 [{self._shop_idx + 1}/{self._shop_mapper.count}] 卡牌"
                )
            else:
                # 首次按 A 仅进入本地预选确认态，严禁发送鼠标单击（防触发角色详情抢焦点）
                self._shop_pending_buy_idx = self._shop_idx
                self._shop_pending_buy_until = (
                    time.monotonic() + _SHOP_PENDING_BUY_TTL_SECONDS
                )
                self._emit_rumble(0.14, 0.18, 28)
                self._set_toast(
                    f"商店 [{self._shop_idx + 1}/{self._shop_mapper.count}]：再按一次 A 确认购买（B 取消 / RT+A 查详情）"
                )
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
                f"已抓取场上 [{self._board_row + 1},{self._board_col + 1}] 干员（原地按 A 可重设朝向）"
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
            return

        if self._focus_zone == FocusZone.STRATEGY:
            self.deploy_fsm.pick_strategy(viewport)
            self._set_toast("已锁定右侧策略（再按 A 直接使用，或移至干员/商店按 A 拖拽）")
            return

        if self._focus_zone == FocusZone.TOP_INFO:
            self._activate_top_info_item(viewport)
            return

        if self._focus_zone == FocusZone.TEAMMATES:
            self._teammate_cycle_pos = self._teammate_focus_idx + 1
            self._switch_teammate_view(self._teammate_focus_idx + 1, viewport)

    def _handle_holding_button(
        self, button: GamepadButton, viewport: ViewportRect
    ) -> None:
        fsm_state = self.deploy_fsm.state

        if button == GamepadButton.B:
            self.deploy_fsm.cancel(viewport)
            self._set_toast(
                "已取消策略使用"
                if fsm_state == DeployState.HOLDING_STRATEGY
                else "已取消抓取"
            )
            return

        if fsm_state == DeployState.HOLDING_STRATEGY:
            if button != GamepadButton.A:
                return
            if self._focus_zone == FocusZone.STRATEGY:
                self.deploy_fsm.click_strategy_in_place(viewport)
                self._set_toast("已直接使用右侧策略")
            elif self._focus_zone == FocusZone.BOARD:
                if self._shop_open and getattr(self.keymap, "auto_toggle_shop", True):
                    self._set_shop_open(False, viewport, announce=False)
                    self._wait_ui_transition(220)
                pos = self._board_mapper.cell_center(self._board_row, self._board_col)
                self.deploy_fsm.use_strategy_on_target(pos, viewport)
                self._set_toast(
                    f"已对场上 [{self._board_row + 1},{self._board_col + 1}] 干员释放策略"
                )
            elif self._focus_zone == FocusZone.BENCH:
                if self._shop_open and getattr(self.keymap, "auto_toggle_shop", True):
                    self._set_shop_open(False, viewport, announce=False)
                    self._wait_ui_transition(220)
                pos = self._bench_mapper.slot_center(self._bench_idx)
                self.deploy_fsm.use_strategy_on_target(pos, viewport)
                self._set_toast(f"已对整备区 [{self._bench_idx + 1}] 干员释放策略")
            elif self._focus_zone == FocusZone.SHOP:
                if not self._shop_open:
                    self._set_shop_open(True, viewport, announce=False)
                    self._wait_ui_transition(220)
                pos = self._shop_mapper.slot_center(self._shop_idx)
                self.deploy_fsm.use_strategy_on_target(pos, viewport)
                self._set_toast(
                    f"已对商店 [{self._shop_idx + 1}/{self._shop_mapper.count}] 释放策略"
                )
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
            if self._focus_zone == FocusZone.TEAMMATES:
                mate_num = self._teammate_focus_idx + 1
                if self.deploy_fsm.gift_holding(mate_num, viewport):
                    self._selected_teammate = mate_num
                    self._set_toast(f"已赠予联机队友 {mate_num}")
                return

            if fsm_state == DeployState.HOLDING_UNIT:
                if self._focus_zone == FocusZone.BENCH:
                    pos = self._bench_mapper.slot_center(self._bench_idx)
                    self.deploy_fsm.place_on_bench(
                        pos, viewport, slot_index=self._bench_idx
                    )
                    self._set_toast(f"已移至整备区 [{self._bench_idx + 1}]")
                elif self._focus_zone == FocusZone.BOARD:
                    same_board_cell = (
                        self.deploy_fsm.holding_source is not None
                        and self.deploy_fsm.holding_source.zone == "board"
                        and self.deploy_fsm.holding_source.board_cell
                        == (self._board_row, self._board_col)
                    )
                    pos = self._board_mapper.cell_center(
                        self._board_row, self._board_col
                    )
                    self.deploy_fsm.place_on_board(
                        (self._board_row, self._board_col), pos, viewport
                    )
                    if self.deploy_fsm.state == DeployState.AIMING_DIRECTION:
                        self._set_toast(
                            "原地转向：请推摇杆或十字键确定干员朝向"
                            if same_board_cell
                            else "请推摇杆或十字键确定干员朝向"
                        )
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
                elif self._focus_zone == FocusZone.EQUIPMENT:
                    same_slot = (
                        self.deploy_fsm.holding_source is not None
                        and self.deploy_fsm.holding_source.slot_index == self._equip_idx
                    )
                    pos = self._equip_mapper.slot_center(self._equip_idx)
                    self.deploy_fsm.place_on_equip(
                        pos, viewport, slot_index=self._equip_idx
                    )
                    self._set_toast(
                        "已取消装备选中"
                        if same_slot
                        else f"已移至装备栏 [{self._equip_idx + 1}]"
                    )

    # ------------------------------------------------------------------
    # HUD 状态快照构造
    # ------------------------------------------------------------------

    @staticmethod
    def _dir_label(direction: Direction) -> str:
        return {"up": "上", "down": "下", "left": "左", "right": "右"}[direction]

    def _build_focus_label(self) -> str:
        if self._free_cursor_active or self._focus_zone == FocusZone.FREE_CURSOR:
            return "自由鼠标"
        if (
            self._unit_selected
            and self._current_scene in (GameScene.BATTLE_PREP, GameScene.BATTLE_COMBAT)
            and not self.deploy_fsm.is_active
        ):
            _, tab_label, _, _ = _UNIT_DETAIL_TABS[self._unit_tab_idx % len(_UNIT_DETAIL_TABS)]
            prefix = "商店干员详情" if self._unit_detail_from_shop else "角色详情"
            return f"{prefix} · {tab_label}"
        if self._focus_zone == FocusZone.SHOP:
            suffix = (
                " · 再按A购买"
                if self._is_shop_pending_buy_for(self._shop_idx)
                else ""
            )
            return f"商店 [{self._shop_idx + 1}/{self._shop_mapper.count}]{suffix}"
        if self._focus_zone == FocusZone.BENCH:
            return f"整备区 [{self._bench_idx + 1}/{self._bench_mapper.count}]"
        if self._focus_zone == FocusZone.BOARD:
            return f"棋盘 [{self._board_row + 1},{self._board_col + 1}]"
        if self._focus_zone == FocusZone.EQUIPMENT:
            return f"装备栏 [{self._equip_idx + 1}/{self._equip_mapper.count}]"
        if self._focus_zone == FocusZone.STRATEGY:
            return (
                "右侧策略 · 已锁定"
                if self.deploy_fsm.state == DeployState.HOLDING_STRATEGY
                else "右侧策略"
            )
        if self._focus_zone == FocusZone.TOP_INFO:
            items = self._top_info_items()
            idx = max(0, min(len(items) - 1, self._top_info_idx))
            return f"顶部栏 · {items[idx][1]}"
        if self._focus_zone == FocusZone.TEAMMATES:
            return f"队友视角 [{self._teammate_focus_idx + 1}/4]"
        if self._focus_zone == FocusZone.EMOTE_PANEL:
            return f"交流表情 [{self._emote_idx + 1}/6]"
        if self._focus_zone == FocusZone.CHOICE_CARDS:
            if self._choice_eye_hidden:
                return "查看战场"
            if self._choice_count >= 5:
                prefix = "六选一"
            elif self._choice_count == 2:
                prefix = "二选一"
            elif self._choice_count == 3:
                prefix = "三选一"
            else:
                prefix = f"{self._choice_count}选一"
            suffix = " · 待确认" if self._selected_choice_idx == self._choice_idx else ""
            return f"{prefix} [{self._choice_idx + 1}/{self._choice_count}]{suffix}"
        if self._focus_zone == FocusZone.CONFIRM_DIALOG:
            idx = (
                1
                if not self._confirm_has_dont_ask and self._confirm_idx >= 2
                else self._confirm_idx
            )
            _, label, _, _, _ = self._confirm_dialog_item(idx)
            title = "回合确认" if self._confirm_has_dont_ask else "确认弹窗"
            return f"{title} · {label}"
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
        if scene == GameScene.MODAL_CONFIRM:
            idx = (
                1
                if not self._confirm_has_dont_ask and self._confirm_idx >= 2
                else self._confirm_idx
            )
            _, label, _, _, _ = self._confirm_dialog_item(idx)
            if self._confirm_has_dont_ask:
                return [
                    PromptItem(nav_key, "取消/确认/不再提示"),
                    PromptItem("A", f"点击{label}"),
                    PromptItem("Start/RT+A", "直接确认"),
                    PromptItem("X", "不再提示"),
                    PromptItem("B", "取消"),
                ]
            return [
                PromptItem(nav_key, "取消/确认"),
                PromptItem("A", f"点击{label}"),
                PromptItem("Start/RT+A", "直接确认"),
                PromptItem("B", "取消"),
            ]
        if scene == GameScene.MODAL_CHOICE:
            if self._choice_eye_hidden:
                return [PromptItem("松开 LT", "展开决策弹窗")]
            confirm_label = (
                "点击白条确认"
                if self._selected_choice_idx == self._choice_idx
                else "选中卡片"
            )
            items = [
                PromptItem(nav_key, "选卡"),
                PromptItem("A", confirm_label),
                PromptItem("RT+A", "一键确认"),
                PromptItem("按住 LT", "收起看场"),
            ]
            if self._selected_choice_idx is not None:
                items.append(PromptItem("B", "重选"))
            return items
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
            if not self._unit_detail_from_shop:
                if scene == GameScene.BATTLE_PREP:
                    items.insert(1, PromptItem("X", "出售(X)"))
                    items.insert(2, PromptItem("Y", "撤回(Q)"))
                else:
                    items.insert(1, PromptItem("Y", "撤退(Q)"))
            return items

        if (
            self._focus_zone == FocusZone.EMOTE_PANEL
            and scene in (GameScene.BATTLE_PREP, GameScene.BATTLE_COMBAT)
            and not self.deploy_fsm.is_active
        ):
            return [
                PromptItem(nav_key, "2×3 选表情"),
                PromptItem("A", "发送表情"),
                PromptItem("RS/LB/RB", "滚动翻页"),
                PromptItem("B", "关闭表情"),
            ]

        if scene == GameScene.BATTLE_COMBAT:
            if self._focus_zone == FocusZone.TOP_INFO:
                return [
                    PromptItem(nav_key, "本局信息/羁绊/出怪"),
                    PromptItem("A", "查看详情"),
                    PromptItem("B", "关闭浮窗"),
                    PromptItem("下", "返回棋盘"),
                ]
            if self._focus_zone == FocusZone.TEAMMATES:
                return [
                    PromptItem(nav_key, "选队友"),
                    PromptItem("A", "切换视角"),
                    PromptItem("右/B", "返回棋盘"),
                ]
            if self._focus_zone == FocusZone.STRATEGY:
                return [
                    PromptItem("A/RT+A", "使用策略"),
                    PromptItem("X", "查看说明"),
                    PromptItem("左/上", "棋盘/出怪"),
                    PromptItem("B", "关闭说明"),
                ]
            return [
                PromptItem(nav_key, "棋盘(上:羁绊/右:策略)"),
                PromptItem("A", "查看详情"),
                PromptItem("LT+十字键", "信息/出怪/表情/队友"),
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
            same_cell = (
                self._focus_zone == FocusZone.BOARD
                and self.deploy_fsm.holding_source is not None
                and self.deploy_fsm.holding_source.zone == "board"
                and self.deploy_fsm.holding_source.board_cell
                == (self._board_row, self._board_col)
            )
            if self._focus_zone == FocusZone.TEAMMATES:
                a_lbl = "赠予队友"
            elif same_cell:
                a_lbl = "原地转向"
            else:
                a_lbl = "部署/换位"
            return [
                PromptItem(nav_key, "选目标格"),
                PromptItem("LB/RB", "整备↔棋盘"),
                PromptItem("A", a_lbl),
                PromptItem("X", "出售"),
                PromptItem("Y", f"赠队友{self._selected_teammate}"),
                PromptItem("B", "取消"),
            ]
        if fsm_state == DeployState.HOLDING_EQUIP:
            if self._focus_zone == FocusZone.EQUIPMENT:
                eq_a_label = "移动/换位"
            elif self._focus_zone == FocusZone.TEAMMATES:
                eq_a_label = "赠予队友"
            else:
                eq_a_label = "穿戴装备"
            return [
                PromptItem(nav_key, "选目标槽/干员"),
                PromptItem("LB/RB", "装备↔棋盘↔整备"),
                PromptItem("A", eq_a_label),
                PromptItem("X", "出售"),
                PromptItem("Y", f"赠队友{self._selected_teammate}"),
                PromptItem("B", "取消"),
            ]
        if fsm_state == DeployState.HOLDING_STRATEGY:
            if self._focus_zone == FocusZone.STRATEGY:
                st_a_label = "直接使用策略"
            elif self._focus_zone == FocusZone.SHOP:
                st_a_label = "拖至商店释放"
            else:
                st_a_label = "拖至干员释放"
            return [
                PromptItem(nav_key, "选干员/商店"),
                PromptItem("LB/RB", "策略↔棋盘↔整备↔商店"),
                PromptItem("A", st_a_label),
                PromptItem("B", "取消"),
            ]

        # BATTLE_PREP + IDLE
        if self._focus_zone == FocusZone.STRATEGY:
            return [
                PromptItem("A", "锁定/拖放策略"),
                PromptItem("RT+A", "直接点击使用"),
                PromptItem("X", "查看说明"),
                PromptItem("左/上/下", "棋盘/出怪/装备"),
                PromptItem("B", "关闭说明"),
            ]
        if self._focus_zone == FocusZone.TOP_INFO:
            return [
                PromptItem(nav_key, "本局信息/对称羁绊/出怪"),
                PromptItem("A", "点击查看"),
                PromptItem("B", "关闭浮窗"),
                PromptItem("LB/RB", "切区"),
                PromptItem("长按Start", "开战"),
            ]
        if self._focus_zone == FocusZone.TEAMMATES:
            return [
                PromptItem(nav_key, "上下选队友"),
                PromptItem("A", "查看队友"),
                PromptItem("右/B", "返回棋盘"),
                PromptItem("长按Start", "开战"),
            ]
        if self._focus_zone == FocusZone.SHOP:
            if self._is_shop_pending_buy_for(self._shop_idx):
                return [
                    PromptItem("A", "确认购买"),
                    PromptItem("B", "取消预选"),
                    PromptItem("RT+A", "查详情"),
                    PromptItem("X", "刷新(D)"),
                    PromptItem("Y", "升本(G)"),
                    PromptItem("RS", "冻结(S)"),
                ]
            return [
                PromptItem("A", "连按A购买"),
                PromptItem("RT+A", "查详情"),
                PromptItem("X", "刷新(D)"),
                PromptItem("Y", "升本(G)"),
                PromptItem("RS", "冻结(S)"),
                PromptItem("B", "收起商店"),
                PromptItem("长按Start", "开战"),
            ]
        if self._focus_zone == FocusZone.BENCH:
            return [
                PromptItem("A", "抓取干员"),
                PromptItem("RT+A", "查详情"),
                PromptItem("LT+LB", "开/关商店"),
                PromptItem("X", "刷新(D)"),
                PromptItem("LB/RB", "切区"),
                PromptItem("长按Start", "开战"),
            ]
        if self._focus_zone == FocusZone.BOARD:
            return [
                PromptItem("A", "抓取/转向/换位"),
                PromptItem("RT+A", "查详情"),
                PromptItem("LT+LB", "开/关商店"),
                PromptItem("LT+RB", "撤退(Q)"),
                PromptItem("LB/RB", "切区"),
                PromptItem("长按Start", "开战"),
            ]
        return [
            PromptItem("A", "抓取装备"),
            PromptItem("LT+LB", "开/关商店"),
            PromptItem("X", "刷新(D)"),
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
