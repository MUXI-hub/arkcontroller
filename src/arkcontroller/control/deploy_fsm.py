"""卫戍协议干员与装备抓取、换位、穿戴、两段式定朝向与快速出售状态机。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

from arkcontroller.control.grid_mapper import Direction

if TYPE_CHECKING:
    from arkcontroller.config.models import KeymapConfig, LayoutProfile
    from arkcontroller.input.virtual_hid import VirtualHID
    from arkcontroller.platform.window_tracker import ViewportRect

RumbleCallback = Callable[[float, float, int], None]


class DeployState(str, Enum):
    """部署交互状态机枚举。"""

    IDLE = "idle"
    HOLDING_UNIT = "holding_unit"
    HOLDING_EQUIP = "holding_equip"
    HOLDING_STRATEGY = "holding_strategy"
    AIMING_DIRECTION = "aiming_direction"


@dataclass(frozen=True)
class HoldingSource:
    """记录当前虚拟抓取源的位置与所属区域信息。"""

    zone: Literal["bench", "board", "equipment", "strategy"]
    norm_pos: tuple[float, float]
    slot_index: int | None = None
    board_cell: tuple[int, int] | None = None
    clicked_in_game: bool = False
    anchor: str = "center_16_9"


class DeployFSM:
    """卫戍协议两段式部署与拖拽交互状态机。

    采用“锁定源槽位 -> 选择目标格 -> 原子平滑拖拽”的设计，确保用户在挑选目标格时
    可以随时按 B 安全取消，或按 X 快速出售而不会误拖乱丢干员。
    """

    def __init__(
        self,
        keymap: KeymapConfig,
        layout: LayoutProfile,
        hid: VirtualHID,
        rumble_cb: RumbleCallback | None = None,
    ) -> None:
        self._keymap = keymap
        self._layout = layout
        self._hid = hid
        self._rumble_cb = rumble_cb

        self._state: DeployState = DeployState.IDLE
        self._holding_source: HoldingSource | None = None
        self._aim_origin_norm: tuple[float, float] | None = None
        self._aim_preview_dir: Direction | None = None
        self._aim_cell: tuple[int, int] | None = None

    @property
    def state(self) -> DeployState:
        return self._state

    @property
    def holding_source(self) -> HoldingSource | None:
        return self._holding_source

    @property
    def holding_source_norm(self) -> tuple[float, float] | None:
        return self._holding_source.norm_pos if self._holding_source is not None else None

    @property
    def aim_origin_norm(self) -> tuple[float, float] | None:
        return self._aim_origin_norm

    @property
    def aim_preview_dir(self) -> Direction | None:
        return self._aim_preview_dir

    @property
    def aim_cell(self) -> tuple[int, int] | None:
        return self._aim_cell

    @property
    def is_active(self) -> bool:
        return self._state != DeployState.IDLE

    def update_configs(self, keymap: KeymapConfig, layout: LayoutProfile) -> None:
        """热重载配置对象。"""
        self._keymap = keymap
        self._layout = layout

    def reset(self) -> None:
        """无副作用重置状态机至空闲态（不发送任何键鼠事件）。"""
        self._state = DeployState.IDLE
        self._holding_source = None
        self._aim_origin_norm = None
        self._aim_preview_dir = None
        self._aim_cell = None

    def _emit_rumble(self, left: float, right: float, duration_ms: int) -> None:
        if self._rumble_cb is not None and self._keymap.vibration_enabled:
            self._rumble_cb(left, right, duration_ms)

    def pick_unit(
        self,
        source_zone: Literal["bench", "board"],
        source_norm: tuple[float, float],
        viewport: ViewportRect,
        *,
        slot_index: int | None = None,
        board_cell: tuple[int, int] | None = None,
        click_on_pick: bool = False,
    ) -> bool:
        """从整备区或棋盘锁定虚拟抓取源。

        默认 ``click_on_pick=False`` 不提前单击源位置，避免触发游戏内「选取角色」引发的
        界面镜头缩放与平移，确保后续平滑拖拽起终点与 HUD 框选坐标完全对齐。
        """
        if click_on_pick:
            sx, sy = viewport.norm_to_screen(source_norm[0], source_norm[1])
            self._hid.click(sx, sy, button="left")

        self._state = DeployState.HOLDING_UNIT
        self._holding_source = HoldingSource(
            zone=source_zone,
            norm_pos=(float(source_norm[0]), float(source_norm[1])),
            slot_index=slot_index,
            board_cell=board_cell,
            clicked_in_game=bool(click_on_pick),
        )
        self._aim_origin_norm = None
        self._aim_preview_dir = None
        self._aim_cell = None

        self._emit_rumble(0.25, 0.35, 45)
        return True

    def pick_equip(
        self,
        source_norm: tuple[float, float],
        viewport: ViewportRect,
        *,
        slot_index: int | None = None,
        click_on_pick: bool = False,
    ) -> bool:
        """从侧边装备栏锁定虚拟抓取装备/法术卡，默认不提前单击以避免干扰界面缩放。"""
        if click_on_pick:
            sx, sy = viewport.norm_to_screen(source_norm[0], source_norm[1])
            self._hid.click(sx, sy, button="left")

        self._state = DeployState.HOLDING_EQUIP
        self._holding_source = HoldingSource(
            zone="equipment",
            norm_pos=(float(source_norm[0]), float(source_norm[1])),
            slot_index=slot_index,
            board_cell=None,
            clicked_in_game=bool(click_on_pick),
        )
        self._aim_origin_norm = None
        self._aim_preview_dir = None
        self._aim_cell = None

        self._emit_rumble(0.20, 0.30, 40)
        return True

    def _strategy_avatar_point(
        self, button_key: str = "strategy_avatar"
    ) -> tuple[float, float, str]:
        """返回右侧策略头像的固定 HUD 归一化坐标与锚点（支持休整/作战双态，不受 shop_open 缩放影响）。"""
        pt = self._layout.buttons.get(button_key)
        if pt is not None:
            return (float(pt.u), float(pt.v), str(pt.anchor))
        if button_key == "strategy_avatar_combat":
            return (0.9602, 0.7486, "right_edge")
        return (0.9605, 0.3243, "right_edge")

    def pick_strategy(
        self,
        viewport: ViewportRect | None = None,
        *,
        source_norm: tuple[float, float] | None = None,
        anchor: str | None = None,
        button_key: str = "strategy_avatar",
    ) -> bool:
        """锁定右侧策略头像作为抓取源，等待原地再按 A 使用或拖拽至干员/商店释放。"""
        _ = viewport
        default_u, default_v, default_anchor = self._strategy_avatar_point(button_key)
        su, sv = (
            (float(source_norm[0]), float(source_norm[1]))
            if source_norm is not None
            else (default_u, default_v)
        )
        s_anchor = str(anchor) if anchor is not None else default_anchor

        self._state = DeployState.HOLDING_STRATEGY
        self._holding_source = HoldingSource(
            zone="strategy",
            norm_pos=(su, sv),
            slot_index=None,
            board_cell=None,
            clicked_in_game=False,
            anchor=s_anchor,
        )
        self._aim_origin_norm = None
        self._aim_preview_dir = None
        self._aim_cell = None

        self._emit_rumble(0.22, 0.32, 45)
        return True

    def use_strategy_on_target(
        self,
        target_norm: tuple[float, float],
        viewport: ViewportRect,
        *,
        target_anchor: str = "center_16_9",
        button_key: str = "strategy_avatar",
    ) -> bool:
        """将右侧策略头像平滑拖拽至目标干员格、整备槽或商店卡槽完成释放。"""
        if self._state != DeployState.HOLDING_STRATEGY or self._holding_source is None:
            return False

        # 起点严格取固定右侧 HUD 策略头像坐标，绝不受 shop_open 镜头缩放影响
        su, sv, s_anchor = self._strategy_avatar_point(button_key)
        start_xy = viewport.norm_to_screen(su, sv, anchor=s_anchor)
        end_xy = viewport.norm_to_screen(
            target_norm[0], target_norm[1], anchor=target_anchor
        )
        self._hid.smooth_drag(start_xy, end_xy, release_at_end=True)

        self._emit_rumble(0.35, 0.55, 65)
        self.reset()
        return True

    def click_strategy_in_place(
        self,
        viewport: ViewportRect,
        *,
        button_key: str = "strategy_avatar",
        point: tuple[float, float, str] | None = None,
    ) -> bool:
        """原地左键直接点击右侧策略头像使用策略（兼容休整/作战双态、空闲态 RT+A、LT+RT 与抓取态二次按 A）。"""
        su, sv, s_anchor = (
            point if point is not None else self._strategy_avatar_point(button_key)
        )
        sx, sy = viewport.norm_to_screen(su, sv, anchor=s_anchor)
        self._hid.click(sx, sy, button="left")

        self._emit_rumble(0.30, 0.45, 55)
        if self._state == DeployState.HOLDING_STRATEGY:
            self.reset()
        return True

    def place_on_bench(
        self,
        target_norm: tuple[float, float],
        viewport: ViewportRect,
        *,
        slot_index: int | None = None,
    ) -> bool:
        """将当前抓取的干员平滑拖拽至目标整备区槽位（整备区换位或从棋盘拉回）。"""
        if self._state != DeployState.HOLDING_UNIT or self._holding_source is None:
            return False

        # 目标为自身原槽位时视为取消选中，不发送冗余拖拽
        if (
            self._holding_source.zone == "bench"
            and slot_index is not None
            and self._holding_source.slot_index == slot_index
        ):
            self.reset()
            return True

        start_xy = viewport.norm_to_screen(*self._holding_source.norm_pos)
        end_xy = viewport.norm_to_screen(target_norm[0], target_norm[1])
        self._hid.smooth_drag(start_xy, end_xy, release_at_end=True)

        self._emit_rumble(0.20, 0.30, 45)
        self.reset()
        return True

    def place_on_board(
        self,
        target_cell: tuple[int, int],
        target_norm: tuple[float, float],
        viewport: ViewportRect,
    ) -> bool:
        """第一段部署：将当前抓取干员拖拽至棋盘目标格（同格回环拖拽原地转向），落位后进入定朝向态。"""
        if self._state != DeployState.HOLDING_UNIT or self._holding_source is None:
            return False

        if (
            self._holding_source.zone == "board"
            and self._holding_source.board_cell == target_cell
        ):
            # 实机原地重新定朝向：原地按住左键微移出格越过拖拽阈值再拖回原格松开，唤出四向菱形定朝向轮盘
            origin_xy = viewport.norm_to_screen(target_norm[0], target_norm[1])
            offset_px = max(36, int(round(viewport.active_width * 0.035)))
            dx = -offset_px if target_norm[0] > 0.65 else offset_px
            loopback_fn = getattr(self._hid, "smooth_drag_loopback", None)
            if callable(loopback_fn):
                loopback_fn(origin_xy, (dx, 0), release_at_end=True)
            else:
                waypoint_xy = (origin_xy[0] + dx, origin_xy[1])
                self._hid.smooth_drag(origin_xy, waypoint_xy, release_at_end=False)
                self._hid.smooth_drag(waypoint_xy, origin_xy, release_at_end=True)
        else:
            start_xy = viewport.norm_to_screen(
                self._holding_source.norm_pos[0],
                self._holding_source.norm_pos[1],
                anchor=self._holding_source.anchor,
            )
            end_xy = viewport.norm_to_screen(target_norm[0], target_norm[1])
            self._hid.smooth_drag(start_xy, end_xy, release_at_end=True)

        self._state = DeployState.AIMING_DIRECTION
        self._aim_origin_norm = (float(target_norm[0]), float(target_norm[1]))
        self._aim_cell = target_cell
        self._aim_preview_dir = None

        self._emit_rumble(0.35, 0.45, 55)
        return True

    def equip_target(
        self,
        target_norm: tuple[float, float],
        viewport: ViewportRect,
    ) -> bool:
        """将当前抓取的装备平滑拖拽至目标干员（场上或整备区）完成穿戴/释放。"""
        if self._state != DeployState.HOLDING_EQUIP or self._holding_source is None:
            return False

        start_xy = viewport.norm_to_screen(*self._holding_source.norm_pos)
        end_xy = viewport.norm_to_screen(target_norm[0], target_norm[1])
        self._hid.smooth_drag(start_xy, end_xy, release_at_end=True)

        self._emit_rumble(0.35, 0.50, 60)
        self.reset()
        return True

    def place_on_equip(
        self,
        target_norm: tuple[float, float],
        viewport: ViewportRect,
        *,
        slot_index: int | None = None,
    ) -> bool:
        """将当前抓取的装备在装备区槽位间平滑拖拽挪动/换位（如同整备区干员换位）。"""
        if self._state != DeployState.HOLDING_EQUIP or self._holding_source is None:
            return False

        # 目标为自身原装备槽位时视为取消选中，不发送冗余拖拽
        if (
            self._holding_source.zone == "equipment"
            and slot_index is not None
            and self._holding_source.slot_index == slot_index
        ):
            self.reset()
            return True

        start_xy = viewport.norm_to_screen(*self._holding_source.norm_pos)
        end_xy = viewport.norm_to_screen(target_norm[0], target_norm[1])
        self._hid.smooth_drag(start_xy, end_xy, release_at_end=True)

        self._emit_rumble(0.20, 0.30, 45)
        self.reset()
        return True

    def set_preview_direction(self, direction: Direction | None) -> None:
        """更新定朝向态下 HUD 十字轮盘的实时预选高亮方向。"""
        if self._state == DeployState.AIMING_DIRECTION:
            self._aim_preview_dir = direction

    def confirm_direction(self, direction: Direction, viewport: ViewportRect) -> bool:
        """第二段部署：以落位点为圆心向指定方向滑动定朝向，完成部署并微震动。"""
        if self._state != DeployState.AIMING_DIRECTION or self._aim_origin_norm is None:
            return False

        origin_xy = viewport.norm_to_screen(*self._aim_origin_norm)
        distance_px = max(
            48, int(round(viewport.active_width * self._keymap.flick_distance_ratio))
        )
        self._hid.directional_flick(origin_xy, direction, distance_px)

        self._emit_rumble(0.45, 0.65, 70)
        self.reset()
        return True

    def cancel(self, viewport: ViewportRect | None = None) -> bool:
        """安全中断当前抓取或定朝向流程。"""
        if self._state == DeployState.IDLE:
            return False

        if (
            self._state == DeployState.AIMING_DIRECTION
            and viewport is not None
            and self._aim_origin_norm is not None
        ):
            # 实机验证：官方 PC 端不识别鼠标右键撤销，需左键点击定朝向菱形左上角的「× 点击取消」按钮
            cancel_u = max(0.04, min(0.96, self._aim_origin_norm[0] - 0.10))
            cancel_v = max(0.04, min(0.96, self._aim_origin_norm[1] - 0.15))
            cancel_xy = viewport.norm_to_screen(cancel_u, cancel_v)
            self._hid.click(cancel_xy[0], cancel_xy[1], button="left")

        self._emit_rumble(0.15, 0.15, 30)
        self.reset()
        return True

    def sell_holding(self, viewport: ViewportRect) -> bool:
        """快速出售当前抓取的干员或装备。

        开启原生快捷键时：先选中源槽位再触发官方出售键（默认 X）；
        关闭原生快捷键时：从抓取源平滑拖拽至右下角金黄色 sell_dropzone 热区。
        """
        if (
            self._state not in (DeployState.HOLDING_UNIT, DeployState.HOLDING_EQUIP)
            or self._holding_source is None
        ):
            return False

        if self._keymap.use_native_keyboard_shortcuts:
            if not self._holding_source.clicked_in_game:
                start_xy = viewport.norm_to_screen(*self._holding_source.norm_pos)
                self._hid.click(start_xy[0], start_xy[1], button="left")
                wait_fn = getattr(self._hid, "wait_ms", None)
                if callable(wait_fn):
                    wait_fn(220)
            shortcuts = getattr(self._keymap, "keyboard_shortcuts", {}) or {}
            if self._holding_source.zone == "board":
                sell_key = shortcuts.get("sell_board", "x")
            else:
                sell_key = shortcuts.get("sell_bench", "x")
            self._hid.tap_key(sell_key)
        else:
            sell_pt = self._layout.buttons.get("sell_dropzone")
            if sell_pt is None:
                return False
            start_xy = viewport.norm_to_screen(*self._holding_source.norm_pos)
            end_xy = viewport.norm_to_screen(sell_pt.u, sell_pt.v, sell_pt.anchor)
            self._hid.smooth_drag(start_xy, end_xy, release_at_end=True)

        self._emit_rumble(0.30, 0.40, 55)
        self.reset()
        return True
